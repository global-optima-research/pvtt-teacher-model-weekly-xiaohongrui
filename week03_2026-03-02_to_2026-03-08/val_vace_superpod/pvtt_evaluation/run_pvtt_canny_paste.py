"""
PVTT 数据集批量推理：Canny 首帧粘贴（实验 10 工作流）

工作流：
1. 从 easy_new.json 读取所有任务
2. 对每个任务：
   a. 加载视频帧（从预提取的 video_frames/）
   b. 加载掩码（从预提取的 masks/）
   c. 加载 RGBA 商品参考图（从 output_dino_rgba/）
   d. 从 RGBA 参考图生成 Canny 边缘（无需 rembg，已有 alpha 通道）
   e. GrowMask(10px) + BlockifyMask(32px)
   f. 首帧 mask 区域嵌入白色 Canny 线条 + 中性色填充
   g. VACE 推理（ref image + canny-embedded frames + mask + prompt）
   h. mask 合成 + 保存结果

输出结构（每个任务）：
    {save_dir}/
    ├── canny_paste_white.mp4                # 合成后的目标视频
    ├── canny_paste_white_comparison.jpg     # [原视频首帧|VACE输入首帧|ref img|target首帧] + prompt
    ├── canny_paste_white_showcase.jpg       # [target首帧|target尾帧]
    ├── ref_canny.png                        # 参考图 Canny 边缘图
    └── experiment.log

用法：
    python run_pvtt_canny_paste.py \\
        --dataset_root ../../samples/pvtt_evaluation_datasets \\
        --output_root ../../experiments/results/1.3B/pvtt_canny_paste \\
        --model_size 1.3B
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

# 添加 baseline 工具库路径
_BASELINE_DIR = Path(__file__).parent.parent / "baseline" / "wan2.1-vace"
sys.path.insert(0, str(_BASELINE_DIR))

from utils import (
    setup_logger,
    load_vace_pipeline,
    load_precise_masks,
    grow_masks,
    blockify_masks,
    filter_supported_kwargs,
    process_pipeline_output,
    composite_with_mask,
    _load_font,
)
from diffsynth.utils.data import save_video


# =============================================================================
# Canny 生成与嵌入（从 exp_canny_paste.py 复用核心逻辑）
# =============================================================================

def generate_canny(image, low_threshold=50, high_threshold=150,
                   bg_threshold=240, blur_ksize=3):
    """从图像生成 Canny 边缘图。

    输入为 RGBA 时使用 alpha 通道精确分离前景。
    输入为 RGB 时回退到亮度阈值去背景。
    """
    arr = np.array(image)

    if arr.ndim == 2:
        gray = arr.copy()
        rgb = np.stack([arr] * 3, axis=-1)
    elif arr.shape[2] == 4:
        alpha = arr[:, :, 3]
        rgb = arr[:, :, :3]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray[alpha < 127] = 0
    else:
        rgb = arr[:, :, :3]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    if bg_threshold < 255:
        bg_mask = np.all(rgb > bg_threshold, axis=-1)
        gray[bg_mask] = 0

    if blur_ksize > 0:
        k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    return cv2.Canny(gray, low_threshold, high_threshold)


def _paste_canny_in_mask_region(frame_arr, mask_bool, canny_edges, line_color):
    """在单帧的 mask 区域内粘贴 Canny 边缘线条（居中、保持纵横比）。"""
    rows = np.any(mask_bool, axis=1)
    cols = np.any(mask_bool, axis=0)
    if not np.any(rows) or not np.any(cols):
        return frame_arr

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    bbox_h, bbox_w = rmax - rmin + 1, cmax - cmin + 1

    ch, cw = canny_edges.shape[:2]
    if ch == 0 or cw == 0:
        return frame_arr
    scale = min(bbox_w / cw, bbox_h / ch)
    new_w = max(1, int(cw * scale))
    new_h = max(1, int(ch * scale))
    resized = cv2.resize(canny_edges, (new_w, new_h),
                         interpolation=cv2.INTER_NEAREST)

    y_off = rmin + (bbox_h - new_h) // 2
    x_off = cmin + (bbox_w - new_w) // 2

    y_s, x_s = max(0, y_off), max(0, x_off)
    y_e = min(y_off + new_h, frame_arr.shape[0])
    x_e = min(x_off + new_w, frame_arr.shape[1])
    cy_s, cx_s = y_s - y_off, x_s - x_off
    cy_e, cx_e = y_e - y_off, x_e - x_off

    roi_canny = resized[cy_s:cy_e, cx_s:cx_e]
    roi_mask = mask_bool[y_s:y_e, x_s:x_e]
    edge_pixels = (roi_canny > 127) & roi_mask

    for c in range(3):
        channel = frame_arr[y_s:y_e, x_s:x_e, c]
        channel[edge_pixels] = line_color[c]
    return frame_arr


def create_canny_frames(frames, masks, canny_edges, fill_value=128,
                        line_color=(255, 255, 255)):
    """创建嵌入 Canny 边缘的输入帧（仅首帧嵌入）。"""
    result = []
    for i, (frame, mask) in enumerate(zip(frames, masks)):
        arr = np.array(frame).copy()
        mb = np.array(mask.convert("L")) > 127
        arr[mb] = fill_value
        if i == 0:
            arr = _paste_canny_in_mask_region(arr, mb, canny_edges, line_color)
        result.append(Image.fromarray(arr))
    return result


# =============================================================================
# 视频帧加载（从预提取的帧目录或 mp4）
# =============================================================================

def load_frames_from_dir(frame_dir, width, height, num_frames):
    """从帧目录加载视频帧，缩放至目标分辨率。"""
    frame_files = sorted(
        list(Path(frame_dir).glob("*.png")) +
        list(Path(frame_dir).glob("*.jpg"))
    )
    if len(frame_files) == 0:
        raise FileNotFoundError(f"帧目录为空: {frame_dir}")

    # 如果帧数不足，复制最后一帧；如果过多，截取前 num_frames 帧
    if len(frame_files) < num_frames:
        # 用最后一帧填充
        while len(frame_files) < num_frames:
            frame_files.append(frame_files[-1])
    frame_files = frame_files[:num_frames]

    return [Image.open(f).convert("RGB").resize((width, height))
            for f in frame_files]


def load_frames_from_mp4(mp4_path, width, height, num_frames):
    """从 mp4 直接加载视频帧（备用方案，当帧未预提取时）。"""
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {mp4_path}")

    frames = []
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb).resize((width, height))
        frames.append(pil_frame)
    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"视频无帧: {mp4_path}")

    # 填充不足的帧
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())

    return frames


def load_masks_from_dir(mask_dir, width, height, num_frames):
    """从 mask 目录加载并二值化。"""
    mask_files = sorted(
        list(Path(mask_dir).glob("*.png")) +
        list(Path(mask_dir).glob("*.jpg"))
    )
    if len(mask_files) == 0:
        raise FileNotFoundError(f"mask 目录为空: {mask_dir}")

    # 帧数对齐
    if len(mask_files) < num_frames:
        while len(mask_files) < num_frames:
            mask_files.append(mask_files[-1])
    mask_files = mask_files[:num_frames]

    masks = []
    for f in mask_files:
        m = Image.open(f).convert("L").resize((width, height), Image.NEAREST)
        m = m.point(lambda p: 255 if p > 127 else 0)
        masks.append(m.convert("RGB"))
    return masks


# =============================================================================
# 可视化
# =============================================================================

def _save_video_file(frames, path, fps=16):
    np_frames = [np.array(f) if isinstance(f, Image.Image) else f
                 for f in frames]
    save_video(np_frames, str(path), fps=fps, quality=5)


def save_four_panel_comparison(output_dir, orig_first, vace_input_first,
                                ref_img, target_first, width, height,
                                prompt, logger):
    """[原视频首帧 | mask video首帧 | ref img(VACE输入) | target首帧] 四列对比 + prompt。"""
    ref_resized = ref_img.convert("RGB").resize((width, height), Image.BICUBIC)
    tgt = Image.fromarray(target_first) if isinstance(target_first, np.ndarray) \
        else target_first

    # 底部 prompt 文字区域
    prompt_font = _load_font(max(16, height // 35))
    total_w = width * 4
    chars_per_line = max(1, total_w // (prompt_font.size * 2 // 3 + 1))
    lines = []
    prompt_text = f"Prompt: {prompt}"
    while prompt_text:
        if len(prompt_text) <= chars_per_line:
            lines.append(prompt_text)
            break
        cut = prompt_text.rfind(" ", 0, chars_per_line + 1)
        if cut <= 0:
            cut = chars_per_line
        lines.append(prompt_text[:cut])
        prompt_text = prompt_text[cut:].lstrip()
    line_height = int(prompt_font.size * 1.4)
    prompt_area_h = len(lines) * line_height + 20

    canvas = Image.new("RGB", (total_w, height + prompt_area_h), (0, 0, 0))
    canvas.paste(orig_first, (0, 0))
    canvas.paste(vace_input_first, (width, 0))
    canvas.paste(ref_resized, (width * 2, 0))
    canvas.paste(tgt, (width * 3, 0))

    draw = ImageDraw.Draw(canvas)
    label_font = _load_font(max(18, height // 30))
    labels = ["Source Video", "VACE Input (Canny)", "Reference Image", "Target Output"]
    for i, label in enumerate(labels):
        draw.text((i * width + 10, 10), label,
                  fill="white", font=label_font, stroke_width=2, stroke_fill="black")

    # 绘制 prompt 文字
    y = height + 10
    for line in lines:
        draw.text((10, y), line, fill="white", font=prompt_font)
        y += line_height

    path = output_dir / "canny_paste_white_comparison.jpg"
    canvas.save(path, quality=95)
    logger.info(f"  四列对比图: {path.name}")


def save_showcase(output_dir, name, comp_frames, logger):
    """首帧 + 末帧展示。"""
    first = Image.fromarray(comp_frames[0])
    last = Image.fromarray(comp_frames[-1])
    w, h = first.size
    canvas = Image.new("RGB", (w * 2, h))
    canvas.paste(first, (0, 0))
    canvas.paste(last, (w, 0))
    draw = ImageDraw.Draw(canvas)
    font = _load_font(max(24, h // 20))
    draw.text((10, 10), "First Frame",
              fill="white", font=font, stroke_width=2, stroke_fill="black")
    draw.text((w + 10, 10), "Last Frame",
              fill="white", font=font, stroke_width=2, stroke_fill="black")
    path = output_dir / f"{name}_showcase.jpg"
    canvas.save(path, quality=95)


# =============================================================================
# 单任务处理
# =============================================================================

def process_single_task(entry, pipe, pipe_base_kwargs_template, args,
                        dataset_root, output_root, logger):
    """处理单个 PVTT 任务。

    返回 True/False 表示成功/失败。
    """
    task_id = entry["id"]
    video_name = entry["video_name"]
    save_dir = entry["save_dir"]
    target_prompt = entry["target_prompt"]
    negative_prompt = entry["negative_prompt"]
    ref_image_id = entry["inference_image_id"]
    width = entry["video_width"]
    height = entry["video_height"]

    output_dir = output_root / save_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"")
    logger.info(f"{'='*60}")
    logger.info(f"任务: {task_id}")
    logger.info(f"  视频: {video_name} ({width}x{height})")
    logger.info(f"  参考图: {ref_image_id}")
    logger.info(f"  输出: {save_dir}")
    logger.info(f"{'='*60}")

    try:
        # ------------------------------------------------------------------
        # 1. 加载视频帧
        # ------------------------------------------------------------------
        frame_dir = dataset_root / "video_frames" / video_name
        mp4_path = dataset_root / "videos" / f"{video_name}.mp4"

        if frame_dir.exists() and len(list(frame_dir.glob("*.jpg")) + list(frame_dir.glob("*.png"))) > 0:
            frames = load_frames_from_dir(
                frame_dir, width, height, args.num_frames)
            logger.info(f"  加载 {len(frames)} 帧（从预提取目录）")
        elif mp4_path.exists():
            frames = load_frames_from_mp4(
                mp4_path, width, height, args.num_frames)
            logger.info(f"  加载 {len(frames)} 帧（从 mp4 直接读取）")
        else:
            logger.error(f"  视频帧和 mp4 均不存在，跳过")
            return False

        # ------------------------------------------------------------------
        # 2. 加载 mask
        # ------------------------------------------------------------------
        mask_dir = dataset_root / "masks" / video_name
        if not mask_dir.exists() or len(list(mask_dir.glob("*.png"))) == 0:
            logger.error(f"  mask 目录不存在或为空: {mask_dir}，跳过")
            return False

        precise_masks = load_masks_from_dir(
            mask_dir, width, height, args.num_frames)
        logger.info(f"  加载 {len(precise_masks)} 个 mask")

        # ------------------------------------------------------------------
        # 3. 加载参考商品图（优先 RGBA，回退 JPG）
        # ------------------------------------------------------------------
        ref_stem = Path(ref_image_id).stem  # e.g., "handfan_2"
        rgba_path = dataset_root / "product_images" / "output_dino_rgba" / f"{ref_stem}.png"
        jpg_path = dataset_root / "product_images" / ref_image_id

        if rgba_path.exists():
            ref_original = Image.open(rgba_path)  # RGBA for canny
            logger.info(f"  参考图: {rgba_path.name} (RGBA)")
        elif jpg_path.exists():
            ref_original = Image.open(jpg_path).convert("RGB")
            logger.info(f"  参考图: {jpg_path.name} (RGB, 无 RGBA 可用)")
        else:
            logger.error(f"  参考图不存在: {rgba_path} / {jpg_path}，跳过")
            return False

        # 用于 VACE 推理的 RGB 参考图（RGBA 需合成到白底，避免透明区域变黑）
        if ref_original.mode == "RGBA":
            white_bg = Image.new("RGB", ref_original.size, (255, 255, 255))
            white_bg.paste(ref_original, mask=ref_original.split()[3])
            reference = white_bg.resize((width, height), Image.BICUBIC)
        else:
            reference = ref_original.convert("RGB").resize(
                (width, height), Image.BICUBIC)

        # ------------------------------------------------------------------
        # 4. 生成 Canny 边缘
        # ------------------------------------------------------------------
        canny_edges = generate_canny(
            ref_original, args.canny_low, args.canny_high,
            bg_threshold=args.bg_threshold, blur_ksize=args.blur_ksize)
        logger.info(f"  Canny: {canny_edges.shape}, "
                    f"边缘像素={np.count_nonzero(canny_edges)}")

        # 保存 Canny 图
        Image.fromarray(canny_edges).save(output_dir / "ref_canny.png")

        # ------------------------------------------------------------------
        # 5. Mask 预处理：GrowMask + BlockifyMask
        # ------------------------------------------------------------------
        grown = grow_masks(precise_masks, pixels=args.grow_pixels)
        processed_masks = blockify_masks(grown, block_size=args.block_size)

        # ------------------------------------------------------------------
        # 6. 创建 Canny 首帧输入
        # ------------------------------------------------------------------
        canny_frames = create_canny_frames(
            frames, processed_masks, canny_edges,
            fill_value=args.fill_value,
            line_color=(255, 255, 255),
        )

        # 记录 VACE 输入首帧（含 Canny 线条）用于后续对比图
        vace_input_first = canny_frames[0]

        # ------------------------------------------------------------------
        # 7. VACE 推理
        # ------------------------------------------------------------------
        kwargs = dict(pipe_base_kwargs_template)
        kwargs.update({
            "prompt": target_prompt,
            "negative_prompt": negative_prompt,
            "vace_video": canny_frames,
            "vace_video_mask": processed_masks,
            "vace_reference_image": reference,
            "height": height,
            "width": width,
        })
        # 重新过滤参数
        kwargs = filter_supported_kwargs(pipe.__call__, kwargs)

        logger.info(f"  VACE 推理中...")
        t0 = time.time()
        video_data = pipe(**kwargs)
        generated = process_pipeline_output(video_data)
        t1 = time.time()
        logger.info(f"  推理完成 ({t1 - t0:.1f}s)")

        # ------------------------------------------------------------------
        # 8. 保存结果
        # ------------------------------------------------------------------
        # mask 合成
        composited = composite_with_mask(frames, generated, processed_masks)

        # 最终视频
        video_path = output_dir / "canny_paste_white.mp4"
        _save_video_file(composited, video_path, fps=args.fps)
        logger.info(f"  已保存: {video_path.name}")

        # 四列对比图 + prompt
        save_four_panel_comparison(
            output_dir, frames[0], vace_input_first,
            reference, composited[0], width, height,
            target_prompt, logger)

        # 首末帧展示
        save_showcase(output_dir, "canny_paste_white", composited, logger)

        # 释放显存
        del video_data, generated, composited, canny_frames
        torch.cuda.empty_cache()

        return True

    except Exception as e:
        logger.error(f"  处理失败: {e}", exc_info=True)
        return False


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PVTT 数据集批量推理：Canny 首帧粘贴（实验 10）")

    # 路径
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="pvtt_evaluation_datasets 根目录")
    parser.add_argument("--output_root", type=str, required=True,
                        help="输出根目录")
    parser.add_argument("--json_path", type=str, default=None,
                        help="easy_new.json 路径")

    # 模型
    parser.add_argument("--model_size", type=str, default="1.3B",
                        choices=["1.3B", "14B"])

    # 视频参数
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=16)

    # VACE 参数
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=7.5)

    # Canny 参数
    parser.add_argument("--fill_value", type=int, default=128)
    parser.add_argument("--grow_pixels", type=int, default=10)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--canny_low", type=int, default=50)
    parser.add_argument("--canny_high", type=int, default=150)
    parser.add_argument("--bg_threshold", type=int, default=240)
    parser.add_argument("--blur_ksize", type=int, default=3)

    # 控制
    parser.add_argument("--task_ids", type=str, default=None,
                        help="仅运行指定任务（逗号分隔 ID），默认运行全部")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="从第几个任务开始（用于断点续跑）")
    parser.add_argument("--max_tasks", type=int, default=None,
                        help="最多运行几个任务（用于测试）")
    parser.add_argument("--skip_existing", action="store_true", default=False,
                        help="跳过已有输出视频的任务")

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    json_path = Path(args.json_path) if args.json_path else \
        dataset_root / "edit_prompt" / "easy_new.json"

    output_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("pvtt_canny_paste", output_root)

    logger.info("=" * 60)
    logger.info("PVTT 批量推理：Canny 首帧粘贴（实验 10 工作流）")
    logger.info("=" * 60)
    logger.info(f"数据集根目录:  {dataset_root}")
    logger.info(f"输出根目录:    {output_root}")
    logger.info(f"模型大小:      {args.model_size}")
    logger.info(f"帧数:          {args.num_frames}")
    logger.info(f"推理步数:      {args.num_inference_steps}")
    logger.info(f"CFG Scale:     {args.cfg_scale}")
    logger.info(f"Canny 阈值:    low={args.canny_low}, high={args.canny_high}")
    logger.info(f"填充值:        {args.fill_value}")
    logger.info(f"Mask 处理:     GrowMask({args.grow_pixels}px) + "
                f"BlockifyMask({args.block_size}px)")
    logger.info("")

    # ------------------------------------------------------------------
    # 1. 加载任务列表
    # ------------------------------------------------------------------
    with open(json_path, "r", encoding="utf-8") as f:
        all_entries = json.load(f)
    logger.info(f"共 {len(all_entries)} 个任务")

    # 过滤
    entries = all_entries
    if args.task_ids:
        filter_ids = set(args.task_ids.split(","))
        entries = [e for e in entries if e["id"] in filter_ids]
        logger.info(f"按 ID 过滤后: {len(entries)} 个任务")

    entries = entries[args.start_idx:]
    if args.max_tasks:
        entries = entries[:args.max_tasks]

    logger.info(f"本次运行: {len(entries)} 个任务 "
                f"(从第 {args.start_idx} 个开始)")

    # ------------------------------------------------------------------
    # 2. 加载 Pipeline（一次加载）
    # ------------------------------------------------------------------
    logger.info("")
    logger.info(f"加载 Wan2.1-VACE-{args.model_size} Pipeline...")
    pipe = load_vace_pipeline(
        model_size=args.model_size, device="cuda", torch_dtype=torch.bfloat16)
    logger.info("Pipeline 加载完成")

    # 构建共享的 pipeline 参数模板
    pipe_base_kwargs_template = {
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
        "tiled": True,
    }

    # ------------------------------------------------------------------
    # 3. 逐任务处理
    # ------------------------------------------------------------------
    success_count = 0
    fail_count = 0
    skip_count = 0
    total_start = time.time()

    for idx, entry in enumerate(entries):
        task_id = entry["id"]
        save_dir = entry["save_dir"]

        # 跳过已完成的任务
        if args.skip_existing:
            existing_video = output_root / save_dir / "canny_paste_white.mp4"
            if existing_video.exists():
                logger.info(f"[{idx+1}/{len(entries)}] 跳过已完成: {task_id}")
                skip_count += 1
                continue

        logger.info(f"")
        logger.info(f"[{idx+1}/{len(entries)}] 开始处理: {task_id}")

        ok = process_single_task(
            entry, pipe, pipe_base_kwargs_template, args,
            dataset_root, output_root, logger)

        if ok:
            success_count += 1
        else:
            fail_count += 1

    # ------------------------------------------------------------------
    # 4. 汇总
    # ------------------------------------------------------------------
    total_elapsed = time.time() - total_start
    hours = int(total_elapsed // 3600)
    minutes = int((total_elapsed % 3600) // 60)
    seconds = int(total_elapsed % 60)

    logger.info("")
    logger.info("=" * 60)
    logger.info("PVTT 批量推理完成")
    logger.info(f"  成功: {success_count}")
    logger.info(f"  跳过: {skip_count}")
    logger.info(f"  失败: {fail_count}")
    logger.info(f"  总耗时: {hours}h {minutes}m {seconds}s")
    logger.info(f"  输出目录: {output_root}")
    logger.info("=" * 60)

    # 保存汇总 JSON
    summary = {
        "total_tasks": len(entries),
        "success": success_count,
        "skipped": skip_count,
        "failed": fail_count,
        "elapsed_seconds": total_elapsed,
        "model_size": args.model_size,
        "config": {
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
            "cfg_scale": args.cfg_scale,
            "canny_low": args.canny_low,
            "canny_high": args.canny_high,
            "fill_value": args.fill_value,
            "grow_pixels": args.grow_pixels,
            "block_size": args.block_size,
            "seed": args.seed,
        }
    }
    import json as json_mod
    with open(output_root / "summary.json", "w", encoding="utf-8") as f:
        json_mod.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"汇总已保存: {output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
