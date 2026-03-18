"""
PVTT 数据集批量推理：使用 FFGo 原版模型（Wan2.2-I2V-A14B + LoRA）

工作流：
1. 从 easy_new.json 读取任务
2. 对每个任务：
   a. 加载原视频首帧和首帧 mask
   b. 加载 RGBA 干净产品图
   c. 使用 LaMa 去除首帧中被替换物体
   d. 构造 FFGo 式首帧（左=产品白底图，右=干净背景居中）
   e. 调用 FFGo 原版 pipeline（Wan2.2-I2V-A14B + LoRA）生成视频
   f. 保存结果 + 对比图

输出结构（每个任务）：
    {save_dir}/
    ├── ffgo_original.mp4                   # 生成的视频（81帧）
    ├── ffgo_original_comparison.jpg        # [原视频首帧|FFGo首帧|第1,21,41,61,81帧] + prompt
    ├── ffgo_original_ref_frame.jpg         # 输入首帧（供检查）
    └── experiment.log

用法：
    python run_pvtt_ffgo_original.py \\
        --dataset_root ../../samples/pvtt_evaluation_datasets \\
        --output_root ../../experiments/results/ffgo_original/pvtt \\
        --ffgo_root ../../FFGO-Video-Customization \\
        --model_name ../../FFGO-Video-Customization/Models/Wan2.2-I2V-A14B
"""

import argparse
import json
import sys
import time
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


# =============================================================================
# 首帧构造（复用 run_pvtt_ffgo_i2v.py 的逻辑）
# =============================================================================

def compose_ffgo_first_frame(product_rgba_pil, clean_bg_pil, width, height):
    """构造 FFGo 式首帧：左侧=产品（白底），右侧=干净背景。"""
    canvas = Image.new("RGB", (width, height), (255, 255, 255))

    left_w = width // 2
    left_h = height

    prod = product_rgba_pil.copy()
    pw, ph = prod.size
    scale = min(left_w / pw, left_h / ph) * 0.85
    new_pw = max(1, int(pw * scale))
    new_ph = max(1, int(ph * scale))
    prod_resized = prod.resize((new_pw, new_ph), Image.BICUBIC)

    x_off = (left_w - new_pw) // 2
    y_off = (left_h - new_ph) // 2

    if prod_resized.mode == "RGBA":
        left_bg = Image.new("RGB", (left_w, left_h), (255, 255, 255))
        left_bg.paste(prod_resized, (x_off, y_off), prod_resized.split()[3])
        canvas.paste(left_bg, (0, 0))
    else:
        canvas.paste(prod_resized, (x_off, y_off))

    right_w = width - left_w
    right_h = height
    bw, bh = clean_bg_pil.size
    bg_scale = min(right_w / bw, right_h / bh) * 0.90
    new_bw = max(1, int(bw * bg_scale))
    new_bh = max(1, int(bh * bg_scale))
    bg_resized = clean_bg_pil.resize((new_bw, new_bh), Image.BICUBIC)
    bg_x = left_w + (right_w - new_bw) // 2
    bg_y = (right_h - new_bh) // 2
    canvas.paste(bg_resized, (bg_x, bg_y))

    return canvas


# =============================================================================
# LaMa inpainting（复用 run_pvtt_ffgo_i2v.py 的逻辑）
# =============================================================================

_LAMA_MODEL = None
_LAMA_CKPT_PATH = None


def _get_lama_model(ckpt_path=None):
    global _LAMA_MODEL
    if _LAMA_MODEL is not None:
        return _LAMA_MODEL

    if ckpt_path:
        cache_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_dir / "big-lama.pt"
        if not target.exists():
            src = Path(ckpt_path)
            if src.exists():
                import shutil
                shutil.copy2(str(src), str(target))

    from simple_lama_inpainting import SimpleLama
    _LAMA_MODEL = SimpleLama()
    return _LAMA_MODEL


def remove_object_from_frame(frame_pil, mask_pil, method="lama",
                              dilate_pixels=3, fill_value=128):
    """从首帧中移除被掩码标记的物体。"""
    mask_arr = np.array(mask_pil.convert("L"))
    if dilate_pixels > 0:
        kernel = np.ones((dilate_pixels * 2 + 1,) * 2, np.uint8)
        mask_arr = cv2.dilate(mask_arr, kernel, iterations=1)
    mask_pil_dilated = Image.fromarray(mask_arr)

    if method == "lama":
        lama = _get_lama_model(ckpt_path=_LAMA_CKPT_PATH)
        result = lama(frame_pil.convert("RGB"), mask_pil_dilated)
        return result
    elif method == "cv2":
        frame_bgr = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        mask_uint8 = (mask_arr > 127).astype(np.uint8) * 255
        inpainted = cv2.inpaint(frame_bgr, mask_uint8, 15, cv2.INPAINT_NS)
        result = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result)
    else:
        frame_arr = np.array(frame_pil).copy()
        frame_arr[mask_arr > 127] = fill_value
        return Image.fromarray(frame_arr)


# =============================================================================
# 数据加载
# =============================================================================

def load_first_frame(frame_dir, mp4_path, width, height):
    """加载视频首帧。"""
    frame_files = sorted(
        list(Path(frame_dir).glob("*.png")) +
        list(Path(frame_dir).glob("*.jpg"))
    )
    if frame_files:
        return Image.open(frame_files[0]).convert("RGB").resize((width, height))

    if Path(mp4_path).exists():
        cap = cv2.VideoCapture(str(mp4_path))
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame_rgb).resize((width, height))

    raise FileNotFoundError(f"无法加载首帧: {frame_dir} / {mp4_path}")


def load_first_mask(mask_dir, width, height):
    """加载首帧 mask。"""
    mask_files = sorted(
        list(Path(mask_dir).glob("*.png")) +
        list(Path(mask_dir).glob("*.jpg"))
    )
    if not mask_files:
        raise FileNotFoundError(f"mask 目录为空: {mask_dir}")
    m = Image.open(mask_files[0]).convert("L").resize((width, height), Image.NEAREST)
    m = m.point(lambda p: 255 if p > 127 else 0)
    return m


# =============================================================================
# 可视化
# =============================================================================

def _load_font(size):
    for fp in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
               "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
               "C:/Windows/Fonts/arial.ttf"]:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                pass
    return ImageFont.load_default()


def save_comparison(output_dir, orig_first, ref_frame, video_frames,
                    width, height, prompt, logger):
    """[原视频首帧 | FFGo首帧 | 第1帧 | 第21帧 | 第41帧 | 第61帧 | 第81帧] + prompt。"""
    # 采样帧索引：0, 20, 40, 60, 80（即第1,21,41,61,81帧）
    sample_indices = [0, 20, 40, 60, 80]
    sample_indices = [i for i in sample_indices if i < len(video_frames)]

    n_cols = 2 + len(sample_indices)  # orig + ref + sampled frames

    # prompt 文字区域
    prompt_font = _load_font(max(14, height // 40))
    total_w = width * n_cols
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

    # 粘贴原视频首帧
    canvas.paste(orig_first.resize((width, height), Image.BICUBIC), (0, 0))
    # 粘贴 FFGo 首帧
    canvas.paste(ref_frame.resize((width, height), Image.BICUBIC), (width, 0))

    # 粘贴采样帧
    for col_idx, frame_idx in enumerate(sample_indices):
        frame = video_frames[frame_idx]
        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(frame)
        frame = frame.resize((width, height), Image.BICUBIC)
        canvas.paste(frame, ((2 + col_idx) * width, 0))

    # 标签
    draw = ImageDraw.Draw(canvas)
    label_font = _load_font(max(16, height // 35))
    labels = ["Source First Frame", "FFGo Input Frame"]
    for idx in sample_indices:
        labels.append(f"Frame {idx + 1}")

    for i, label in enumerate(labels):
        draw.text((i * width + 10, 10), label,
                  fill="white", font=label_font, stroke_width=2, stroke_fill="black")

    # prompt 文字
    y = height + 10
    for line in lines:
        draw.text((10, y), line, fill="white", font=prompt_font)
        y += line_height

    path = output_dir / "ffgo_original_comparison.jpg"
    canvas.save(path, quality=95)
    logger.info(f"  对比图: {path.name}")


# =============================================================================
# 主逻辑
# =============================================================================

def setup_logger(name, output_dir):
    """简易 logger。"""
    import logging
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(output_dir / "experiment.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def extract_frames_from_tensor(sample_tensor):
    """从 pipeline 输出 tensor (1, C, N, H, W) 提取 PIL 帧列表。"""
    # sample shape: (1, C, N, H, W), values in [0, 1]
    video = sample_tensor[0]  # (C, N, H, W)
    frames = []
    for i in range(video.shape[1]):
        frame = video[:, i].permute(1, 2, 0).cpu().numpy()  # (H, W, C)
        frame = (frame * 255).clip(0, 255).astype(np.uint8)
        frames.append(Image.fromarray(frame))
    return frames


def main():
    parser = argparse.ArgumentParser(
        description="PVTT 数据集批量推理：FFGo 原版模型")

    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--json_path", type=str, default=None)
    parser.add_argument("--ffgo_root", type=str, required=True,
                        help="FFGO-Video-Customization 目录路径")

    # FFGo 模型路径
    parser.add_argument("--model_name", type=str, default=None,
                        help="Wan2.2-I2V-A14B 模型路径")
    parser.add_argument("--lora_low", type=str, default=None,
                        help="Low-noise LoRA adapter 路径")
    parser.add_argument("--lora_high", type=str, default=None,
                        help="High-noise LoRA adapter 路径")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Pipeline config 路径")

    # 生成参数
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--video_length", type=int, default=81)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)

    # Inpainting
    parser.add_argument("--inpaint_method", type=str, default="lama",
                        choices=["lama", "cv2", "neutral_fill"])
    parser.add_argument("--dilate_pixels", type=int, default=3)
    parser.add_argument("--lama_ckpt", type=str, default=None)

    # 转场
    parser.add_argument("--prompt_prefix", type=str,
                        default="ad23r2 the camera view suddenly changes. ")
    parser.add_argument("--skip_transition_frames", type=int, default=4)

    # 控制
    parser.add_argument("--task_ids", type=str, default=None)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--max_tasks", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true", default=False)

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    ffgo_root = Path(args.ffgo_root)
    json_path = Path(args.json_path) if args.json_path else \
        dataset_root / "edit_prompt" / "easy_new.json"

    # 设置默认 FFGo 模型路径
    if args.model_name is None:
        args.model_name = str(ffgo_root / "Models" / "Wan2.2-I2V-A14B")
    if args.lora_low is None:
        args.lora_low = str(ffgo_root / "Models" / "Lora" /
                            "10_LargeMixedDatset_wan_14bLow_f81_LongCaption_StartMatch_run_r128_a128_3obj_Longrun_B4" /
                            "checkpoint-600.safetensors")
    if args.lora_high is None:
        args.lora_high = str(ffgo_root / "Models" / "Lora" /
                             "10_LargeMixedDatset_wan_14bHigh_f81_LongCaption_StartMatch_run_r128_a128_3obj_Longrun_B4" /
                             "checkpoint-600.safetensors")
    if args.config_path is None:
        args.config_path = str(ffgo_root / "VideoX-Fun" / "config" / "wan2.2" /
                               "wan_civitai_i2v.yaml")

    # LaMa ckpt
    global _LAMA_CKPT_PATH
    _LAMA_CKPT_PATH = args.lama_ckpt

    output_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("pvtt_ffgo_original", output_root)

    logger.info("=" * 60)
    logger.info("PVTT 批量推理：FFGo 原版模型（Wan2.2-I2V-A14B + LoRA）")
    logger.info("=" * 60)
    logger.info(f"数据集根目录:  {dataset_root}")
    logger.info(f"输出根目录:    {output_root}")
    logger.info(f"FFGo 根目录:   {ffgo_root}")
    logger.info(f"模型路径:      {args.model_name}")
    logger.info(f"分辨率:        {args.width}x{args.height}")
    logger.info(f"帧数:          {args.video_length}")
    logger.info(f"Prompt 前缀:   {args.prompt_prefix}")
    logger.info(f"跳过转场帧:    前 {args.skip_transition_frames} 帧")
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

    # =========================================================================
    # 加载 FFGo Pipeline
    # =========================================================================
    # 将 FFGo 代码路径加入 sys.path
    ffgo_videox_path = str(ffgo_root / "VideoX-Fun")
    ffgo_videox_examples = str(ffgo_root / "VideoX-Fun" / "examples" / "wan2.2")
    ffgo_project_root = str(ffgo_root)
    for p in [ffgo_videox_path, ffgo_videox_examples, ffgo_project_root]:
        if p not in sys.path:
            sys.path.insert(0, p)

    from single_predict import build_wan22_pipeline, infer_video

    logger.info("加载 FFGo Pipeline（Wan2.2-I2V-A14B + LoRA）...")
    pipe, vae, boundary, device = build_wan22_pipeline(
        config_path=args.config_path,
        model_name=args.model_name,
        lora_low=args.lora_low,
        lora_high=args.lora_high,
    )
    logger.info("Pipeline 加载完成")

    negative_prompt = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，"
                       "静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
                       "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
                       "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，"
                       "背景人很多，倒着走")

    sample_size = [args.height, args.width]

    # =========================================================================
    # 逐任务处理
    # =========================================================================
    success_count = 0
    fail_count = 0
    skip_count = 0
    total_start = time.time()

    for idx, entry in enumerate(entries):
        task_id = entry["id"]
        video_name = entry["video_name"]
        save_dir = entry["save_dir"]
        target_prompt = entry["target_prompt"]
        ref_image_id = entry["inference_image_id"]
        # FFGo 使用固定分辨率，不依赖原视频分辨率
        orig_width = entry["video_width"]
        orig_height = entry["video_height"]

        output_dir = output_root / save_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.skip_existing:
            if (output_dir / "ffgo_original.mp4").exists():
                logger.info(f"[{idx+1}/{len(entries)}] 跳过已完成: {task_id}")
                skip_count += 1
                continue

        logger.info(f"")
        logger.info(f"{'='*60}")
        logger.info(f"[{idx+1}/{len(entries)}] 任务: {task_id}")
        logger.info(f"  视频: {video_name} ({orig_width}x{orig_height})")
        logger.info(f"  参考图: {ref_image_id}")
        logger.info(f"{'='*60}")

        try:
            # 1. 加载原视频首帧（按原始分辨率）
            frame_dir = dataset_root / "video_frames" / video_name
            mp4_path = dataset_root / "videos" / f"{video_name}.mp4"
            first_frame = load_first_frame(frame_dir, mp4_path, orig_width, orig_height)
            logger.info(f"  加载首帧完成")

            # 2. 加载首帧 mask
            mask_dir = dataset_root / "masks" / video_name
            first_mask = load_first_mask(mask_dir, orig_width, orig_height)
            logger.info(f"  加载首帧 mask 完成")

            # 3. 加载参考商品图（RGBA）
            ref_stem = Path(ref_image_id).stem
            rgba_path = dataset_root / "product_images" / "output_dino_rgba" / f"{ref_stem}.png"
            jpg_path = dataset_root / "product_images" / ref_image_id

            if rgba_path.exists():
                product_img = Image.open(rgba_path)
                logger.info(f"  参考图: {rgba_path.name} (RGBA)")
            elif jpg_path.exists():
                product_img = Image.open(jpg_path).convert("RGB")
                logger.info(f"  参考图: {jpg_path.name} (RGB)")
            else:
                logger.error(f"  参考图不存在，跳过")
                fail_count += 1
                continue

            # 4. 从首帧移除物体
            clean_bg = remove_object_from_frame(
                first_frame, first_mask, method=args.inpaint_method,
                dilate_pixels=args.dilate_pixels)
            logger.info(f"  首帧物体移除完成 (method={args.inpaint_method})")

            # 5. 构造 FFGo 式首帧（使用 FFGo 的生成分辨率）
            ref_frame = compose_ffgo_first_frame(
                product_img, clean_bg, args.width, args.height)
            ref_frame.save(output_dir / "ffgo_original_ref_frame.jpg", quality=95)
            logger.info(f"  FFGo 首帧构造完成 ({args.width}x{args.height})")

            # 6. 构造 prompt
            full_prompt = args.prompt_prefix + target_prompt
            logger.info(f"  Prompt: {full_prompt[:100]}...")

            # 7. FFGo 推理
            logger.info(f"  FFGo 推理中...")
            t0 = time.time()

            ref_frame_path = str(output_dir / "ffgo_original_ref_frame.jpg")
            video_path = infer_video(
                pipe, vae, boundary, device,
                sample_size=sample_size,
                video_length=args.video_length,
                validation_image_start=ref_frame_path,
                prompt=full_prompt,
                save_path=str(output_dir),
                negative_prompt=negative_prompt,
                fps=args.fps,
                seed=args.seed,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
            )

            t1 = time.time()
            logger.info(f"  推理完成 ({t1 - t0:.1f}s)")

            # 8. 重命名输出视频
            generated_video_path = Path(video_path)
            final_video_path = output_dir / "ffgo_original.mp4"
            if generated_video_path.exists() and generated_video_path != final_video_path:
                generated_video_path.rename(final_video_path)

            # 9. 提取帧用于对比图
            cap = cv2.VideoCapture(str(final_video_path))
            video_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_frames.append(Image.fromarray(frame_rgb))
            cap.release()
            logger.info(f"  提取了 {len(video_frames)} 帧用于对比图")

            # 10. 对比图
            if video_frames:
                save_comparison(
                    output_dir, first_frame, ref_frame, video_frames,
                    args.width, args.height, full_prompt, logger)

            torch.cuda.empty_cache()
            success_count += 1

        except Exception as e:
            logger.error(f"  处理失败: {e}", exc_info=True)
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
        "model": "Wan2.2-I2V-A14B + FFGo LoRA",
        "config": {
            "width": args.width,
            "height": args.height,
            "video_length": args.video_length,
            "guidance_scale": args.guidance_scale,
            "num_inference_steps": args.num_inference_steps,
            "prompt_prefix": args.prompt_prefix,
            "skip_transition_frames": args.skip_transition_frames,
            "inpaint_method": args.inpaint_method,
            "seed": args.seed,
        }
    }
    with open(output_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"汇总已保存: {output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
