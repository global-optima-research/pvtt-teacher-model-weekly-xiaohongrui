"""
PVTT 数据集批量推理：Canny 首帧粘贴 + Bbox Mask（原始输出，不合成）

与 run_pvtt_canny_paste_bbox.py 的唯一区别：
  输出视频直接是模型生成的原始视频，不经过 composite_with_mask 合成。
  用于验证"割裂感"是否来自合成步骤。

输出结构（每个任务）：
    {save_dir}/
    ├── canny_paste_bbox_raw.mp4                # 模型直接生成的视频（无合成）
    ├── canny_paste_bbox_raw_comparison.jpg     # [原视频首帧|VACE输入首帧|ref img|target首帧] + prompt
    ├── canny_paste_bbox_raw_showcase.jpg       # [target首帧|target尾帧]
    ├── ref_canny.png
    └── experiment.log

用法：
    python run_pvtt_canny_paste_bbox_raw.py \\
        --dataset_root ../../samples/pvtt_evaluation_datasets \\
        --output_root ../../experiments/results/1.3B/pvtt_canny_paste_bbox_raw \\
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
    create_bbox_masks_from_precise,
    filter_supported_kwargs,
    process_pipeline_output,
    _load_font,
)
from diffsynth.utils.data import save_video

# 复用 canny_paste 的核心逻辑
from run_pvtt_canny_paste import (
    generate_canny,
    create_canny_frames,
    load_frames_from_dir,
    load_frames_from_mp4,
    load_masks_from_dir,
)


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
    """[原视频首帧 | VACE输入首帧 | ref img(VACE输入) | target首帧] 四列对比 + prompt。"""
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
    labels = ["Source Video", "VACE Input (Canny+Bbox)", "Reference Image", "Raw Output (No Comp)"]
    for i, label in enumerate(labels):
        draw.text((i * width + 10, 10), label,
                  fill="white", font=label_font, stroke_width=2, stroke_fill="black")

    # 绘制 prompt 文字
    y = height + 10
    for line in lines:
        draw.text((10, y), line, fill="white", font=prompt_font)
        y += line_height

    path = output_dir / "canny_paste_bbox_raw_comparison.jpg"
    canvas.save(path, quality=95)
    logger.info(f"  四列对比图: {path.name}")


def save_showcase(output_dir, name, frames, logger):
    """[target首帧 | target尾帧] 展示。"""
    first = Image.fromarray(frames[0]) if isinstance(frames[0], np.ndarray) \
        else frames[0]
    last = Image.fromarray(frames[-1]) if isinstance(frames[-1], np.ndarray) \
        else frames[-1]
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
    """处理单个 PVTT 任务。返回 True/False。"""
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
        # 1. 加载视频帧
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

        # 2. 加载 mask
        mask_dir = dataset_root / "masks" / video_name
        if not mask_dir.exists() or len(list(mask_dir.glob("*.png"))) == 0:
            logger.error(f"  mask 目录不存在或为空: {mask_dir}，跳过")
            return False

        precise_masks = load_masks_from_dir(
            mask_dir, width, height, args.num_frames)
        logger.info(f"  加载 {len(precise_masks)} 个 mask")

        # 3. 加载参考商品图（优先 RGBA）
        ref_stem = Path(ref_image_id).stem
        rgba_path = dataset_root / "product_images" / "output_dino_rgba" / f"{ref_stem}.png"
        jpg_path = dataset_root / "product_images" / ref_image_id

        if rgba_path.exists():
            ref_original = Image.open(rgba_path)  # RGBA for canny
            logger.info(f"  参考图: {rgba_path.name} (RGBA)")
        elif jpg_path.exists():
            ref_original = Image.open(jpg_path).convert("RGB")
            logger.info(f"  参考图: {jpg_path.name} (RGB)")
        else:
            logger.error(f"  参考图不存在，跳过")
            return False

        # RGBA 需合成到白底，避免透明区域变黑
        if ref_original.mode == "RGBA":
            white_bg = Image.new("RGB", ref_original.size, (255, 255, 255))
            white_bg.paste(ref_original, mask=ref_original.split()[3])
            reference = white_bg.resize((width, height), Image.BICUBIC)
        else:
            reference = ref_original.convert("RGB").resize(
                (width, height), Image.BICUBIC)

        # 4. 生成 Canny 边缘
        canny_edges = generate_canny(
            ref_original, args.canny_low, args.canny_high,
            bg_threshold=args.bg_threshold, blur_ksize=args.blur_ksize)
        logger.info(f"  Canny: {canny_edges.shape}, "
                    f"边缘像素={np.count_nonzero(canny_edges)}")
        Image.fromarray(canny_edges).save(output_dir / "ref_canny.png")

        # 5. 从精确 mask 生成 Bbox mask
        bbox_masks = create_bbox_masks_from_precise(
            precise_masks,
            margin_lr=args.bbox_margin_lr,
            margin_top=args.bbox_margin_top,
            margin_bottom=args.bbox_margin_bottom,
        )
        logger.info(f"  Bbox mask 生成完成 (margin: lr={args.bbox_margin_lr}, "
                     f"top={args.bbox_margin_top}, bottom={args.bbox_margin_bottom})")

        # 6. 创建 Canny 首帧输入（在 bbox mask 区域内嵌入 Canny 线条）
        canny_frames = create_canny_frames(
            frames, bbox_masks, canny_edges,
            fill_value=args.fill_value,
            line_color=(255, 255, 255),
        )
        vace_input_first = canny_frames[0]

        # 7. VACE 推理
        kwargs = dict(pipe_base_kwargs_template)
        kwargs.update({
            "prompt": target_prompt,
            "negative_prompt": negative_prompt,
            "vace_video": canny_frames,
            "vace_video_mask": bbox_masks,
            "vace_reference_image": reference,
            "height": height,
            "width": width,
        })
        kwargs = filter_supported_kwargs(pipe.__call__, kwargs)

        logger.info(f"  VACE 推理中...")
        t0 = time.time()
        video_data = pipe(**kwargs)
        generated = process_pipeline_output(video_data)
        t1 = time.time()
        logger.info(f"  推理完成 ({t1 - t0:.1f}s)")

        # 8. 直接保存模型原始输出（不做 composite_with_mask 合成）
        video_path = output_dir / "canny_paste_bbox_raw.mp4"
        _save_video_file(generated, video_path, fps=args.fps)
        logger.info(f"  已保存: {video_path.name} (原始输出，无合成)")

        # 四列对比图 + prompt
        save_four_panel_comparison(
            output_dir, frames[0], vace_input_first,
            reference, generated[0], width, height,
            target_prompt, logger)

        # 首末帧展示
        save_showcase(output_dir, "canny_paste_bbox_raw", generated, logger)

        del video_data, generated, canny_frames
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
        description="PVTT 数据集批量推理：Canny + Bbox Mask（原始输出，不合成）")

    # 路径
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--json_path", type=str, default=None)

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
    parser.add_argument("--canny_low", type=int, default=50)
    parser.add_argument("--canny_high", type=int, default=150)
    parser.add_argument("--bg_threshold", type=int, default=240)
    parser.add_argument("--blur_ksize", type=int, default=3)

    # Bbox 参数
    parser.add_argument("--bbox_margin_lr", type=int, default=20)
    parser.add_argument("--bbox_margin_top", type=int, default=20)
    parser.add_argument("--bbox_margin_bottom", type=int, default=20)

    # 控制
    parser.add_argument("--task_ids", type=str, default=None,
                        help="仅运行指定任务（逗号分隔 ID）")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--max_tasks", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true", default=False)

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    json_path = Path(args.json_path) if args.json_path else \
        dataset_root / "edit_prompt" / "easy_new.json"

    output_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("pvtt_canny_paste_bbox_raw", output_root)

    logger.info("=" * 60)
    logger.info("PVTT 批量推理：Canny + Bbox Mask（原始输出，不合成）")
    logger.info("=" * 60)
    logger.info(f"数据集根目录:  {dataset_root}")
    logger.info(f"输出根目录:    {output_root}")
    logger.info(f"模型大小:      {args.model_size}")
    logger.info(f"帧数:          {args.num_frames}")
    logger.info(f"Canny 阈值:    low={args.canny_low}, high={args.canny_high}")
    logger.info(f"填充值:        {args.fill_value}")
    logger.info(f"Bbox 边距:     lr={args.bbox_margin_lr}, "
                f"top={args.bbox_margin_top}, bottom={args.bbox_margin_bottom}")
    logger.info("")

    # 加载任务列表
    with open(json_path, "r", encoding="utf-8") as f:
        all_entries = json.load(f)
    logger.info(f"共 {len(all_entries)} 个任务")

    entries = all_entries
    if args.task_ids:
        filter_ids = set(args.task_ids.split(","))
        entries = [e for e in entries if e["id"] in filter_ids]
        logger.info(f"按 ID 过滤后: {len(entries)} 个任务")

    entries = entries[args.start_idx:]
    if args.max_tasks:
        entries = entries[:args.max_tasks]

    logger.info(f"本次运行: {len(entries)} 个任务")

    # 加载 Pipeline
    logger.info(f"加载 Wan2.1-VACE-{args.model_size} Pipeline...")
    pipe = load_vace_pipeline(
        model_size=args.model_size, device="cuda", torch_dtype=torch.bfloat16)
    logger.info("Pipeline 加载完成")

    pipe_base_kwargs_template = {
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
        "tiled": True,
    }

    # 逐任务处理
    success_count = 0
    fail_count = 0
    skip_count = 0
    total_start = time.time()

    for idx, entry in enumerate(entries):
        task_id = entry["id"]
        save_dir = entry["save_dir"]

        if args.skip_existing:
            existing_video = output_root / save_dir / "canny_paste_bbox_raw.mp4"
            if existing_video.exists():
                logger.info(f"[{idx+1}/{len(entries)}] 跳过已完成: {task_id}")
                skip_count += 1
                continue

        logger.info(f"[{idx+1}/{len(entries)}] 开始处理: {task_id}")
        ok = process_single_task(
            entry, pipe, pipe_base_kwargs_template, args,
            dataset_root, output_root, logger)
        if ok:
            success_count += 1
        else:
            fail_count += 1

    # 汇总
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

    summary = {
        "total_tasks": len(entries),
        "success": success_count,
        "skipped": skip_count,
        "failed": fail_count,
        "elapsed_seconds": total_elapsed,
        "model_size": args.model_size,
        "note": "raw output without composite_with_mask",
        "config": {
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
            "cfg_scale": args.cfg_scale,
            "canny_low": args.canny_low,
            "canny_high": args.canny_high,
            "fill_value": args.fill_value,
            "bbox_margin_lr": args.bbox_margin_lr,
            "bbox_margin_top": args.bbox_margin_top,
            "bbox_margin_bottom": args.bbox_margin_bottom,
            "seed": args.seed,
        }
    }
    with open(output_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"汇总已保存: {output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
