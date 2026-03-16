"""
PVTT 数据集批量推理：VACE I2V + FFGo 首帧参考

工作流：
1. 从 easy_new.json 读取任务
2. 对每个任务：
   a. 加载原视频首帧 + 掩码
   b. 加载产品 RGBA 参考图
   c. 构造 FFGo 式首帧：左侧=产品 RGBA（白底），右侧=原视频首帧去除物体后的画面
   d. 构造 mask 序列：首帧 mask=全0（保留首帧），后续帧 mask=全1（全部生成）
   e. VACE I2V 推理（首帧 + mask seq + prompt）
   f. 保存结果（不做合成，因为后续帧全部由模型生成）

输出结构（每个任务）：
    {save_dir}/
    ├── ffgo_i2v.mp4                      # 模型生成的目标视频
    ├── ffgo_i2v_comparison.jpg            # [原视频首帧|输入首帧ref img|target首帧] 三列
    ├── ffgo_i2v_showcase.jpg              # [target首帧|target尾帧]
    ├── ffgo_ref_frame.jpg                 # 输入给模型的首帧（用于检查）
    └── experiment.log

设计说明：
  这是 I2V 模式——首帧固定（FFGo 式拼贴），后续帧全部由 VACE 从头生成。
  mask_seq: 首帧=全0（不改变），后续帧=全1（全部重新生成）。
  不需要做 mask 合成，因为后续帧没有"原视频内容"需要保留。

  首帧去除物体的方式：
  - 默认使用 neutral fill（中性灰填充 mask 区域）
  - 可选 --inpaint_method=cv2 使用 OpenCV Telea 修复
  - 可选 --inpaint_method=lama 使用 LaMa 模型修复（需要安装 lama-cleaner）

用法：
    python run_pvtt_ffgo_i2v.py \\
        --dataset_root ../../samples/pvtt_evaluation_datasets \\
        --output_root ../../experiments/results/1.3B/pvtt_ffgo_i2v \\
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
    filter_supported_kwargs,
    process_pipeline_output,
    _load_font,
)
from diffsynth.utils.data import save_video


# =============================================================================
# 首帧构造
# =============================================================================

_LAMA_MODEL = None  # 延迟加载的全局 LaMa 模型缓存
_LAMA_CKPT_PATH = None  # 用户指定的本地 LaMa ckpt 路径


def _get_lama_model(ckpt_path=None):
    """延迟加载 LaMa 模型（全局缓存，只加载一次）。

    Args:
        ckpt_path: 本地 big-lama.pt 路径。如果提供，会预先复制/软链接到
                   torch hub 缓存目录，从而跳过从 GitHub 下载。
    """
    global _LAMA_MODEL
    if _LAMA_MODEL is None:
        # 如果指定了本地 ckpt，预先放到 torch hub cache 目录
        if ckpt_path:
            import shutil
            cache_dir = Path(torch.hub.get_dir()) / "checkpoints"
            cache_dir.mkdir(parents=True, exist_ok=True)
            target = cache_dir / "big-lama.pt"
            if not target.exists():
                src = Path(ckpt_path)
                if not src.exists():
                    raise FileNotFoundError(f"LaMa ckpt 不存在: {src}")
                # 优先软链接，节省空间
                try:
                    target.symlink_to(src.resolve())
                except OSError:
                    shutil.copy2(str(src), str(target))
        from simple_lama_inpainting import SimpleLama
        _LAMA_MODEL = SimpleLama()
    return _LAMA_MODEL


def remove_object_from_frame(frame_pil, mask_pil, method="lama",
                              fill_value=128, dilate_pixels=15):
    """从帧中移除物体（mask 区域）。

    Args:
        frame_pil:     原始帧 (PIL RGB)
        mask_pil:      物体 mask (PIL, 白色=物体区域)
        method:        "lama" (推荐) / "cv2" / "neutral_fill"
        fill_value:    neutral fill 的灰度值
        dilate_pixels: mask 膨胀像素数（扩大修复区域，避免边缘残留）

    Returns:
        PIL Image (RGB)，物体区域已被修复/填充。
    """
    frame_arr = np.array(frame_pil).copy()
    mask_arr = np.array(mask_pil.convert("L"))
    mask_bool = mask_arr > 127
    mask_uint8 = (mask_bool.astype(np.uint8)) * 255

    # 膨胀 mask，扩大修复区域以覆盖边缘残留
    if dilate_pixels > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * dilate_pixels + 1, 2 * dilate_pixels + 1))
        mask_uint8 = cv2.dilate(mask_uint8, kernel, iterations=1)

    if method == "lama":
        # LaMa inpainting（大区域修复效果最佳）
        lama = _get_lama_model(ckpt_path=_LAMA_CKPT_PATH)
        # simple-lama-inpainting 接受 PIL Image
        mask_pil_dilated = Image.fromarray(mask_uint8).convert("L")
        result = lama(frame_pil.convert("RGB"), mask_pil_dilated)
        return result.convert("RGB")
    elif method == "cv2":
        # OpenCV Navier-Stokes inpainting
        frame_bgr = cv2.cvtColor(frame_arr, cv2.COLOR_RGB2BGR)
        inpainted = cv2.inpaint(frame_bgr, mask_uint8, 15, cv2.INPAINT_NS)
        result = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result)
    else:
        # neutral fill
        frame_arr[mask_uint8 > 127] = fill_value
        return Image.fromarray(frame_arr)


def compose_ffgo_first_frame(product_rgba_pil, clean_bg_pil, width, height):
    """构造 FFGo 式首帧：左侧=产品（白底），右侧=干净背景。

    Args:
        product_rgba_pil: 产品 RGBA 图像 (PIL RGBA 或 RGB)
        clean_bg_pil:     去除物体后的首帧 (PIL RGB)
        width, height:    目标分辨率

    Returns:
        PIL Image (RGB), width x height
    """
    canvas = Image.new("RGB", (width, height), (255, 255, 255))

    # 左半部分：产品图（保持纵横比居中，白色背景）
    left_w = width // 2
    left_h = height

    prod = product_rgba_pil.copy()
    pw, ph = prod.size
    scale = min(left_w / pw, left_h / ph) * 0.85  # 留 15% 边距
    new_pw = max(1, int(pw * scale))
    new_ph = max(1, int(ph * scale))
    prod_resized = prod.resize((new_pw, new_ph), Image.BICUBIC)

    # 居中粘贴到左半部分
    x_off = (left_w - new_pw) // 2
    y_off = (left_h - new_ph) // 2

    if prod_resized.mode == "RGBA":
        # 白底 + alpha 合成
        left_bg = Image.new("RGB", (left_w, left_h), (255, 255, 255))
        left_bg.paste(prod_resized, (x_off, y_off), prod_resized.split()[3])
        canvas.paste(left_bg, (0, 0))
    else:
        canvas.paste(prod_resized, (x_off, y_off))

    # 右半部分：干净背景帧（等比缩放后居中，不拉伸变形）
    right_w = width - left_w
    right_h = height
    bw, bh = clean_bg_pil.size
    bg_scale = min(right_w / bw, right_h / bh) * 0.90  # 留 10% 边距
    new_bw = max(1, int(bw * bg_scale))
    new_bh = max(1, int(bh * bg_scale))
    bg_resized = clean_bg_pil.resize((new_bw, new_bh), Image.BICUBIC)
    bg_x = left_w + (right_w - new_bw) // 2
    bg_y = (right_h - new_bh) // 2
    canvas.paste(bg_resized, (bg_x, bg_y))

    return canvas


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
    return m.convert("RGB")


# =============================================================================
# 可视化
# =============================================================================

def _save_video_file(frames, path, fps=16):
    np_frames = [np.array(f) if isinstance(f, Image.Image) else f
                 for f in frames]
    save_video(np_frames, str(path), fps=fps, quality=5)


def save_three_panel_comparison(output_dir, orig_first, ref_frame,
                                 target_first, width, height, prompt, logger):
    """[原视频首帧 | 输入首帧(ref img) | target首帧] 三列对比 + prompt 文字。"""
    tgt = Image.fromarray(target_first) if isinstance(target_first, np.ndarray) \
        else target_first

    # 底部 prompt 文字区域
    prompt_font = _load_font(max(16, height // 35))
    total_w = width * 3
    # 自动换行：按字符宽度估算每行字符数
    chars_per_line = max(1, total_w // (prompt_font.size * 2 // 3 + 1))
    lines = []
    prompt_text = f"Prompt: {prompt}"
    while prompt_text:
        if len(prompt_text) <= chars_per_line:
            lines.append(prompt_text)
            break
        # 在 chars_per_line 附近找空格断行
        cut = prompt_text.rfind(" ", 0, chars_per_line + 1)
        if cut <= 0:
            cut = chars_per_line
        lines.append(prompt_text[:cut])
        prompt_text = prompt_text[cut:].lstrip()
    line_height = int(prompt_font.size * 1.4)
    prompt_area_h = len(lines) * line_height + 20  # 上下留 10px

    canvas = Image.new("RGB", (total_w, height + prompt_area_h), (0, 0, 0))
    canvas.paste(orig_first, (0, 0))
    canvas.paste(ref_frame, (width, 0))
    canvas.paste(tgt, (width * 2, 0))

    draw = ImageDraw.Draw(canvas)
    label_font = _load_font(max(18, height // 30))
    labels = ["Source First Frame", "FFGo Ref Frame", "Post-Transition Frame"]
    for i, label in enumerate(labels):
        draw.text((i * width + 10, 10), label,
                  fill="white", font=label_font, stroke_width=2, stroke_fill="black")

    # 绘制 prompt 文字
    y = height + 10
    for line in lines:
        draw.text((10, y), line, fill="white", font=prompt_font)
        y += line_height

    path = output_dir / "ffgo_i2v_comparison.jpg"
    canvas.save(path, quality=95)
    logger.info(f"  三列对比图: {path.name}")


def save_showcase(output_dir, name, frames_list, logger):
    """[target首帧 | target尾帧] 展示。"""
    first = frames_list[0] if isinstance(frames_list[0], Image.Image) \
        else Image.fromarray(frames_list[0])
    last = frames_list[-1] if isinstance(frames_list[-1], Image.Image) \
        else Image.fromarray(frames_list[-1])
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
# 转场检测
# =============================================================================

def _compute_ssim_gray(img_a, img_b):
    """计算两张 PIL Image 之间的灰度 SSIM（简化版，无需 skimage）。"""
    a = np.array(img_a.convert("L"), dtype=np.float64)
    b = np.array(img_b.convert("L"), dtype=np.float64)
    # 简化 SSIM：用均值和方差计算
    mu_a, mu_b = a.mean(), b.mean()
    sig_a, sig_b = a.std(), b.std()
    sig_ab = ((a - mu_a) * (b - mu_b)).mean()
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ssim = ((2 * mu_a * mu_b + c1) * (2 * sig_ab + c2)) / \
           ((mu_a ** 2 + mu_b ** 2 + c1) * (sig_a ** 2 + sig_b ** 2 + c2))
    return float(ssim)


def _detect_transition_end(generated_frames, min_discard, logger):
    """自动检测转场结束帧。

    策略：从第 min_discard 帧开始，计算相邻帧 SSIM。
    当 SSIM > 阈值（帧间变化小 = 场景稳定）时，认为转场结束。
    同时记录所有帧的 SSIM 到日志供分析。

    Returns:
        discard_count: 应丢弃的帧数
    """
    if len(generated_frames) < 3:
        return 0

    # 计算所有相邻帧 SSIM
    ssim_values = []
    for i in range(len(generated_frames) - 1):
        s = _compute_ssim_gray(generated_frames[i], generated_frames[i + 1])
        ssim_values.append(s)

    # 记录前 20 帧的 SSIM
    ssim_log = ", ".join(f"{s:.3f}" for s in ssim_values[:20])
    logger.info(f"  帧间 SSIM (前20帧): [{ssim_log}]")

    # 阈值：SSIM > 0.85 且连续 2 帧都稳定 → 转场结束
    threshold = 0.85
    for i in range(max(min_discard, 1), len(ssim_values) - 1):
        if ssim_values[i] > threshold and ssim_values[i - 1] > threshold:
            logger.info(f"  转场检测: 第 {i} 帧后场景稳定 "
                        f"(SSIM={ssim_values[i]:.3f}), 丢弃前 {i} 帧")
            return i

    # 未检测到稳定点，使用 min_discard
    logger.info(f"  转场检测: 未检测到明确稳定点，使用默认丢弃 {min_discard} 帧")
    return min_discard


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
        # 1. 加载原视频首帧
        frame_dir = dataset_root / "video_frames" / video_name
        mp4_path = dataset_root / "videos" / f"{video_name}.mp4"
        first_frame = load_first_frame(frame_dir, mp4_path, width, height)
        logger.info(f"  加载首帧完成")

        # 2. 加载首帧 mask
        mask_dir = dataset_root / "masks" / video_name
        if not mask_dir.exists() or len(list(mask_dir.glob("*.png"))) == 0:
            logger.error(f"  mask 目录不存在或为空: {mask_dir}，跳过")
            return False
        first_mask = load_first_mask(mask_dir, width, height)
        logger.info(f"  加载首帧 mask 完成")

        # 3. 加载产品 RGBA 参考图
        ref_stem = Path(ref_image_id).stem
        rgba_path = dataset_root / "product_images" / "output_dino_rgba" / f"{ref_stem}.png"
        jpg_path = dataset_root / "product_images" / ref_image_id

        if rgba_path.exists():
            product_img = Image.open(rgba_path)  # 保留 RGBA
            logger.info(f"  参考图: {rgba_path.name} (RGBA)")
        elif jpg_path.exists():
            product_img = Image.open(jpg_path).convert("RGB")
            logger.info(f"  参考图: {jpg_path.name} (RGB)")
        else:
            logger.error(f"  参考图不存在，跳过")
            return False

        # 4. 从首帧移除物体
        clean_bg = remove_object_from_frame(
            first_frame, first_mask, method=args.inpaint_method,
            fill_value=args.fill_value, dilate_pixels=args.dilate_pixels)
        logger.info(f"  首帧物体移除完成 (method={args.inpaint_method}, "
                     f"dilate={args.dilate_pixels}px)")

        # 5. 构造 FFGo 式首帧
        ref_frame = compose_ffgo_first_frame(product_img, clean_bg, width, height)
        ref_frame.save(output_dir / "ffgo_ref_frame.jpg", quality=95)
        logger.info(f"  FFGo 首帧构造完成")

        # 6. 构造 VACE 输入
        # Template: [首帧=ref_frame, 后续帧全黑/空]
        # Mask: [首帧=全0(保留), 后续帧=全1(全部生成)]
        template_frames = [ref_frame]
        mask_frames = [Image.new("RGB", (width, height), (0, 0, 0))]  # 首帧 mask=0

        for _ in range(args.num_frames - 1):
            template_frames.append(Image.new("RGB", (width, height), (128, 128, 128)))
            mask_frames.append(Image.new("RGB", (width, height), (255, 255, 255)))

        # 7. 构造 prompt
        # 场景转换提示 + 原始 target prompt
        if args.transition_prefix:
            full_prompt = f"{args.transition_prefix} {target_prompt}"
        else:
            full_prompt = target_prompt
        logger.info(f"  Prompt: {full_prompt[:100]}...")

        # 8. VACE 推理
        kwargs = dict(pipe_base_kwargs_template)
        kwargs.update({
            "prompt": full_prompt,
            "negative_prompt": negative_prompt,
            "vace_video": template_frames,
            "vace_video_mask": mask_frames,
            "height": height,
            "width": width,
        })

        # 也提供 reference image（产品图）作为额外参考
        if args.use_ref_image:
            ref_for_pipe = product_img.convert("RGB").resize(
                (width, height), Image.BICUBIC)
            kwargs["vace_reference_image"] = ref_for_pipe

        kwargs = filter_supported_kwargs(pipe.__call__, kwargs)

        logger.info(f"  VACE I2V 推理中...")
        t0 = time.time()
        video_data = pipe(**kwargs)
        generated = process_pipeline_output(video_data)
        t1 = time.time()
        logger.info(f"  推理完成 ({t1 - t0:.1f}s)")

        # 9. 自动检测转场结束帧
        # FFGo 论文的 Fc=4 仅对 LoRA 微调后的模型成立。
        # VACE 未经 LoRA 微调，转场时间不可预测，因此用 SSIM 自动检测：
        # 当连续帧之间的结构相似度稳定时（不再剧烈变化），认为转场结束。
        dc = _detect_transition_end(generated, args.discard_frames, logger)

        if dc > 0 and dc < len(generated):
            clean_frames = generated[dc:]
        else:
            clean_frames = generated

        clean_np = [np.array(f) for f in clean_frames]

        # 保存完整视频（含转场帧，供分析）
        gen_all_np = [np.array(f) for f in generated]
        full_video_path = output_dir / "ffgo_i2v_full.mp4"
        _save_video_file(generated, full_video_path, fps=args.fps)
        logger.info(f"  已保存完整视频（含转场）: {full_video_path.name}")

        # 保存裁剪后的视频（丢弃转场帧）
        video_path = output_dir / "ffgo_i2v.mp4"
        _save_video_file(clean_frames, video_path, fps=args.fps)
        logger.info(f"  已保存有效视频（去转场）: {video_path.name}")

        # 三列对比图：用转场后第一帧作为 target 首帧
        save_three_panel_comparison(
            output_dir, first_frame, ref_frame,
            clean_np[0], width, height, full_prompt, logger)

        # 首末帧展示（转场后的首帧 + 尾帧）
        save_showcase(output_dir, "ffgo_i2v", clean_np, logger)

        del video_data, generated, gen_all_np, clean_frames, clean_np
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
        description="PVTT 数据集批量推理：VACE I2V + FFGo 首帧参考")

    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--json_path", type=str, default=None)
    parser.add_argument("--model_size", type=str, default="1.3B",
                        choices=["1.3B", "14B"])
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=7.5)

    # FFGo 特有参数
    parser.add_argument("--inpaint_method", type=str, default="lama",
                        choices=["lama", "cv2", "neutral_fill"],
                        help="首帧物体移除方式: lama(推荐) / cv2(OpenCV) / neutral_fill(灰色填充)")
    parser.add_argument("--fill_value", type=int, default=128,
                        help="neutral_fill 的灰度值")
    parser.add_argument("--dilate_pixels", type=int, default=3,
                        help="物体移除前 mask 膨胀像素数（默认 3）")
    parser.add_argument("--lama_ckpt", type=str, default=None,
                        help="本地 big-lama.pt 路径（避免从 GitHub 下载）")
    parser.add_argument("--transition_prefix", type=str,
                        default="The camera view suddenly changes.",
                        help="prompt 前缀（场景转换提示词）")
    parser.add_argument("--discard_frames", type=int, default=4,
                        help="丢弃前 N 帧转场帧（默认 4，参考 FFGo 的 Fc=4）")
    parser.add_argument("--use_ref_image", action="store_true", default=True,
                        help="同时传入产品图作为 vace_reference_image（默认开启）")
    parser.add_argument("--no_ref_image", dest="use_ref_image",
                        action="store_false",
                        help="不传入 vace_reference_image")

    # 控制
    parser.add_argument("--task_ids", type=str, default=None,
                        help="仅运行指定任务（逗号分隔 ID）")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--max_tasks", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true", default=False)

    args = parser.parse_args()

    # 设置全局 LaMa ckpt 路径（供 _get_lama_model 使用）
    global _LAMA_CKPT_PATH
    _LAMA_CKPT_PATH = args.lama_ckpt

    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    json_path = Path(args.json_path) if args.json_path else \
        dataset_root / "edit_prompt" / "easy_new.json"

    output_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("pvtt_ffgo_i2v", output_root)

    logger.info("=" * 60)
    logger.info("PVTT 批量推理：VACE I2V + FFGo 首帧参考")
    logger.info("=" * 60)
    logger.info(f"数据集根目录:  {dataset_root}")
    logger.info(f"输出根目录:    {output_root}")
    logger.info(f"模型大小:      {args.model_size}")
    logger.info(f"帧数:          {args.num_frames}")
    logger.info(f"物体移除方式:  {args.inpaint_method} (膨胀 {args.dilate_pixels}px)")
    logger.info(f"丢弃转场帧:   前 {args.discard_frames} 帧")
    logger.info(f"使用 ref img:  {args.use_ref_image}")
    logger.info(f"Prompt 前缀:   {args.transition_prefix}")
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
            existing_video = output_root / save_dir / "ffgo_i2v.mp4"
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
        "config": {
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
            "cfg_scale": args.cfg_scale,
            "inpaint_method": args.inpaint_method,
            "use_ref_image": args.use_ref_image,
            "transition_prefix": args.transition_prefix,
            "seed": args.seed,
        }
    }
    with open(output_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"汇总已保存: {output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
