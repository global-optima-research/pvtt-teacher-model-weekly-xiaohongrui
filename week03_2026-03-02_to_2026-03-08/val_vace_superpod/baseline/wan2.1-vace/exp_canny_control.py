"""
实验：双参考图（商品图 + Canny 结构图）+ 中性色填充 + GrowMask + BlockifyMask → 商品替换

输入：
- 视频帧（mask 区域中性色填充，与实验七相同）
- GrowMask + BlockifyMask 处理后的 mask
- 两张参考图：商品外观图 + 该商品的 Canny 边缘图
- 文本提示

核心思路：
给 VACE 模型提供两张参考图——商品外观图提供外观/纹理信息，Canny 边缘图提供
结构/形状信息。测试模型能否同时利用外观和结构信息来生成更准确的替换结果。

由于 DiffSynth 的 vace_reference_image 可能只支持单张图，脚本会依次尝试：
1. 传入参考图列表 [商品图, Canny图]
2. 若列表不支持，则拼接为左右并排的单张参考图

输入视频帧本身不做任何修改（标准中性色填充），结构信息完全通过参考图传递。

用法：
    python exp_canny_control.py \\
        --sample_dir ../../samples/teapot \\
        --output_dir ../../experiments/results/canny_control \\
        --canny_low 100 --canny_high 200
"""

import argparse
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    setup_logger,
    load_vace_pipeline,
    load_video_frames,
    load_precise_masks,
    load_reference_image,
    neutral_fill_frames,
    grow_masks,
    blockify_masks,
    filter_supported_kwargs,
    process_pipeline_output,
    composite_with_mask,
    add_common_args,
    _load_font,
)
from diffsynth.utils.data import save_video


# =============================================================================
# Canny 边缘生成
# =============================================================================

def generate_canny(image, low_threshold=50, high_threshold=150,
                   bg_threshold=240, blur_ksize=3):
    """从图像生成 Canny 边缘图。

    推荐先用 rembg 将图像转为 RGBA（背景+阴影透明），再传入本函数；
    本函数会自动使用 alpha 通道精确分离前景，避免阴影/反光等伪边缘。
    若输入为 RGB，则回退到基于亮度阈值的背景去除（对阴影不可靠）。

    Args:
        image:          PIL Image (RGB / RGBA / L)
        low_threshold:  Canny 低阈值（默认 50）
        high_threshold: Canny 高阈值（默认 150）
        bg_threshold:   背景检测阈值，RGB 各通道均 > 此值的像素视为背景
                        并置为黑色（默认 240，设为 255 则禁用）
        blur_ksize:     高斯模糊核大小，0 = 不模糊（默认 3）
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

    # 去除近白色背景 → 置为黑色，使物体外轮廓产生强梯度
    if bg_threshold < 255:
        bg_mask = np.all(rgb > bg_threshold, axis=-1)
        gray[bg_mask] = 0

    # 高斯模糊：平滑内部纹理噪声，保留外轮廓
    if blur_ksize > 0:
        k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    return cv2.Canny(gray, low_threshold, high_threshold)


def canny_to_rgb(canny_edges):
    """将单通道 Canny 边缘图转为 RGB PIL Image（白色边缘，黑色背景）。"""
    return Image.fromarray(
        np.stack([canny_edges] * 3, axis=-1)
    )


def create_composite_ref(ref_image, canny_rgb, width, height):
    """将商品图和 Canny 图左右拼接为单张参考图 (width, height)。

    左半 = 商品外观图，右半 = Canny 结构图。
    """
    half_w = width // 2
    left = ref_image.resize((half_w, height), Image.BICUBIC)
    right = canny_rgb.resize((width - half_w, height), Image.BICUBIC)
    composite = Image.new("RGB", (width, height))
    composite.paste(left, (0, 0))
    composite.paste(right, (half_w, 0))
    return composite


# =============================================================================
# 可视化辅助
# =============================================================================

def _save_canny_image(output_dir, canny_edges, logger):
    path = output_dir / "ref_canny.png"
    Image.fromarray(canny_edges).save(path)
    logger.info(f"已保存参考图 Canny 边缘图: {path}")


def _save_ref_comparison(output_dir, ref_image, canny_rgb, actual_ref, strategy, logger):
    """保存参考图对比：[商品图 | Canny图 | 实际传入模型的参考图]。"""
    w, h = ref_image.size
    canvas = Image.new("RGB", (w * 3, h))
    canvas.paste(ref_image, (0, 0))
    canvas.paste(canny_rgb.resize((w, h), Image.BICUBIC), (w, 0))
    if isinstance(actual_ref, list):
        # 列表模式：拼两张缩略图展示
        thumb_w = w // 2
        canvas.paste(actual_ref[0].resize((thumb_w, h), Image.BICUBIC), (w * 2, 0))
        canvas.paste(actual_ref[1].resize((w - thumb_w, h), Image.BICUBIC), (w * 2 + thumb_w, 0))
    else:
        canvas.paste(actual_ref.resize((w, h), Image.BICUBIC), (w * 2, 0))
    draw = ImageDraw.Draw(canvas)
    font = _load_font(max(20, h // 25))
    labels = ["Ref Image", "Canny Edge", f"Model Input ({strategy})"]
    for i, label in enumerate(labels):
        draw.text((i * w + 10, 10), label,
                  fill="white", font=font, stroke_width=2, stroke_fill="black")
    path = output_dir / "ref_comparison.jpg"
    canvas.save(path, quality=95)
    logger.info(f"已保存参考图对比: {path}")


def _save_mask_comparison(output_dir, precise, processed, logger):
    w, h = precise[0].size
    canvas = Image.new("RGB", (w * 2, h))
    canvas.paste(precise[0].convert("RGB"), (0, 0))
    canvas.paste(processed[0].convert("RGB"), (w, 0))
    draw = ImageDraw.Draw(canvas)
    font = _load_font(max(24, h // 20))
    draw.text((10, 10), "Precise Mask",
              fill="white", font=font, stroke_width=2, stroke_fill="black")
    draw.text((w + 10, 10), "Grow+Blockify",
              fill="white", font=font, stroke_width=2, stroke_fill="black")
    path = output_dir / "mask_comparison.jpg"
    canvas.save(path, quality=95)
    logger.info(f"已保存 mask 对比图: {path}")


def _save_preprocess_comparison(output_dir, orig, mask_img, filled, logger):
    """[原始帧 | 处理后 Mask | 中性色填充帧] 第一帧。"""
    w, h = orig.size
    canvas = Image.new("RGB", (w * 3, h))
    canvas.paste(orig, (0, 0))
    canvas.paste(mask_img.convert("RGB"), (w, 0))
    canvas.paste(filled, (w * 2, 0))
    draw = ImageDraw.Draw(canvas)
    font = _load_font(max(24, h // 20))
    for i, label in enumerate(["Original", "Processed Mask", "Neutral Fill"]):
        draw.text((i * w + 10, 10), label,
                  fill="white", font=font, stroke_width=2, stroke_fill="black")
    path = output_dir / "preprocess_comparison.jpg"
    canvas.save(path, quality=95)
    logger.info(f"已保存预处理对比图: {path}")


def _save_first_frame_comparison(output_dir, name, inp, mask_img, target, logger):
    """[Input | Mask | Target] 第一帧对比。"""
    w, h = inp.size
    tgt = Image.fromarray(target) if isinstance(target, np.ndarray) else target
    canvas = Image.new("RGB", (w * 3, h))
    canvas.paste(inp, (0, 0))
    canvas.paste(mask_img.convert("RGB"), (w, 0))
    canvas.paste(tgt, (w * 2, 0))
    draw = ImageDraw.Draw(canvas)
    font = _load_font(max(24, h // 20))
    for i, label in enumerate(["Input", "Mask", "Target"]):
        draw.text((i * w + 10, 10), label,
                  fill="white", font=font, stroke_width=2, stroke_fill="black")
    path = output_dir / f"{name}_comparison.jpg"
    canvas.save(path, quality=95)
    logger.info(f"已保存第一帧对比图: {path}")


def _save_showcase(output_dir, name, comp_frames, logger):
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
    logger.info(f"已保存首末帧展示: {path}")


def _save_video_file(frames, path, fps=15):
    """保存帧列表为视频。"""
    np_frames = [np.array(f) if isinstance(f, Image.Image) else f for f in frames]
    save_video(np_frames, str(path), fps=fps, quality=5)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="双参考图（商品图 + Canny 结构图）+ GrowMask + BlockifyMask 商品替换实验",
    )
    add_common_args(parser)
    parser.add_argument("--fill_value", type=int, default=128,
                        help="中性色填充值（默认 128）")
    parser.add_argument("--grow_pixels", type=int, default=10,
                        help="Mask 膨胀像素数（默认 10）")
    parser.add_argument("--block_size", type=int, default=32,
                        help="Mask 网格对齐块大小（默认 32）")
    parser.add_argument("--canny_low", type=int, default=50,
                        help="Canny 低阈值（默认 50）")
    parser.add_argument("--canny_high", type=int, default=150,
                        help="Canny 高阈值（默认 150）")
    parser.add_argument("--bg_threshold", type=int, default=240,
                        help="背景检测阈值：RGB 各通道均 > 此值视为背景并置黑"
                             "（默认 240，255=禁用）")
    parser.add_argument("--blur_ksize", type=int, default=3,
                        help="Canny 前高斯模糊核大小（默认 3，0=不模糊）")
    parser.add_argument("--no_rembg", action="store_true", default=False,
                        help="禁用 rembg 前景分割，回退到亮度阈值去背景"
                             "（默认启用 rembg 以精确去除背景+阴影）")
    args = parser.parse_args()

    sample_dir = Path(args.sample_dir)
    output_dir = Path(args.output_dir)
    logger = setup_logger("exp_canny_control", output_dir)

    logger.info("=" * 60)
    logger.info("实验：双参考图（商品图 + Canny 结构图）→ 商品替换")
    logger.info("=" * 60)
    logger.info(f"样本目录:          {sample_dir}")
    logger.info(f"输出目录:          {output_dir}")
    logger.info(f"分辨率:            {args.width}x{args.height}")
    logger.info(f"帧数:              {args.num_frames}")
    logger.info(f"模型大小:          {args.model_size}")
    logger.info(f"种子:              {args.seed}")
    logger.info(f"填充值:            {args.fill_value}")
    logger.info(f"Mask 膨胀像素:     {args.grow_pixels}")
    logger.info(f"Mask 网格大小:     {args.block_size}")
    logger.info(f"Canny 阈值:        low={args.canny_low}, high={args.canny_high}")
    logger.info(f"背景去除阈值:      {args.bg_threshold}")
    logger.info(f"高斯模糊核:        {args.blur_ksize}")
    logger.info(f"rembg 前景分割:    {'禁用' if args.no_rembg else '启用（优先）'}")
    logger.info(f"提示词:            {args.prompt}")
    logger.info(f"负面提示词:        {args.negative_prompt}")
    logger.info("")

    # ------------------------------------------------------------------
    # 加载数据
    # ------------------------------------------------------------------
    logger.info("加载数据...")
    frames = load_video_frames(
        sample_dir / "video_frames", args.width, args.height, args.num_frames)
    precise_masks = load_precise_masks(
        sample_dir / "masks", args.width, args.height, args.num_frames)
    ref_path = sample_dir / "reference_images" / args.reference_image
    reference = load_reference_image(ref_path, args.width, args.height)
    logger.info(f"已加载 {len(frames)} 帧视频、{len(precise_masks)} 个 mask、"
                f"参考图: {ref_path.name}")

    # ------------------------------------------------------------------
    # 生成参考图 Canny 边缘
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("生成参考图 Canny 边缘图")
    logger.info("=" * 60)
    ref_original = Image.open(ref_path)

    # 前景分割：优先用 rembg 精确去除背景+阴影，回退到亮度阈值
    ref_for_canny = ref_original
    if not args.no_rembg:
        try:
            from rembg import remove
            ref_for_canny = remove(ref_original)   # → RGBA，背景+阴影透明
            logger.info("rembg 前景分割成功（背景+阴影已移除）")
        except ImportError:
            logger.warning("rembg 不可用（pip install rembg[gpu]），"
                           "回退到亮度阈值去背景")
    else:
        logger.info("已禁用 rembg，使用亮度阈值去背景")

    canny_edges = generate_canny(
        ref_for_canny, args.canny_low, args.canny_high,
        bg_threshold=args.bg_threshold, blur_ksize=args.blur_ksize)
    logger.info(f"Canny 尺寸: {canny_edges.shape}, "
                f"边缘像素: {np.count_nonzero(canny_edges)}")
    _save_canny_image(output_dir, canny_edges, logger)

    # 将 Canny 转为 RGB 并缩放至视频分辨率
    canny_rgb = canny_to_rgb(canny_edges).resize(
        (args.width, args.height), Image.BICUBIC)

    # ------------------------------------------------------------------
    # Mask 预处理：GrowMask + BlockifyMask
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Mask 预处理：GrowMask({args.grow_pixels}px) + "
                f"BlockifyMask({args.block_size}px)")
    logger.info("=" * 60)
    grown = grow_masks(precise_masks, pixels=args.grow_pixels)
    processed_masks = blockify_masks(grown, block_size=args.block_size)
    logger.info("Mask 预处理完成。")
    _save_mask_comparison(output_dir, precise_masks, processed_masks, logger)

    # ------------------------------------------------------------------
    # 中性色填充输入帧（与实验七相同，不嵌入 Canny）
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"中性色填充输入帧（fill_value={args.fill_value}）")
    logger.info("=" * 60)
    filled_frames = neutral_fill_frames(
        frames, processed_masks, fill_value=args.fill_value)
    logger.info("中性色填充完成。")
    _save_preprocess_comparison(
        output_dir, frames[0], processed_masks[0], filled_frames[0], logger)

    # ------------------------------------------------------------------
    # 加载 Pipeline
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"加载 Wan2.1-VACE-{args.model_size} Pipeline")
    logger.info("=" * 60)
    pipe = load_vace_pipeline(
        model_size=args.model_size, device="cuda", torch_dtype=torch.bfloat16)
    logger.info("Pipeline 加载完成。")

    # ------------------------------------------------------------------
    # 准备双参考图并尝试传入
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("准备双参考图（商品图 + Canny 结构图）")
    logger.info("=" * 60)

    ref_resized = reference.resize((args.width, args.height))

    # 构建基础 pipeline 参数（不含 vace_reference_image）
    base_kwargs = dict(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        vace_video=filled_frames,
        vace_video_mask=processed_masks,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        tiled=True,
    )

    # 策略 1：尝试传入参考图列表
    strategy = None
    actual_ref = None
    video_data = None

    logger.info("策略 1：尝试传入参考图列表 [商品图, Canny图]...")
    try:
        kwargs_list = dict(base_kwargs)
        kwargs_list["vace_reference_image"] = [ref_resized, canny_rgb]
        kwargs_list = filter_supported_kwargs(pipe.__call__, kwargs_list)
        video_data = pipe(**kwargs_list)
        strategy = "list"
        actual_ref = [ref_resized, canny_rgb]
        logger.info("策略 1 成功：Pipeline 接受了参考图列表。")
    except Exception as e:
        logger.info(f"策略 1 失败：{e}")
        logger.info("")

    # 策略 2：左右拼接为单张参考图
    if video_data is None:
        logger.info("策略 2：拼接为左右并排的单张参考图...")
        composite_ref = create_composite_ref(
            ref_resized, canny_rgb, args.width, args.height)
        try:
            kwargs_comp = dict(base_kwargs)
            kwargs_comp["vace_reference_image"] = composite_ref
            kwargs_comp = filter_supported_kwargs(pipe.__call__, kwargs_comp)
            video_data = pipe(**kwargs_comp)
            strategy = "composite"
            actual_ref = composite_ref
            logger.info("策略 2 成功：Pipeline 接受了拼接参考图。")
        except Exception as e:
            logger.warning(f"策略 2 失败：{e}")
            logger.warning(traceback.format_exc())

    # 策略 3：仅使用原始参考图（降级，仅当前两种都失败时）
    if video_data is None:
        logger.warning("策略 3（降级）：仅使用商品参考图（无 Canny 结构信息）...")
        kwargs_fallback = dict(base_kwargs)
        kwargs_fallback["vace_reference_image"] = ref_resized
        kwargs_fallback = filter_supported_kwargs(pipe.__call__, kwargs_fallback)
        video_data = pipe(**kwargs_fallback)
        strategy = "single_ref_only"
        actual_ref = ref_resized
        logger.warning("策略 3 完成。本次结果等同于实验七（无 Canny 结构控制）。")

    logger.info(f"最终使用策略: {strategy}")

    # 保存参考图对比
    _save_ref_comparison(
        output_dir, ref_resized, canny_rgb, actual_ref, strategy, logger)

    # ------------------------------------------------------------------
    # 处理输出
    # ------------------------------------------------------------------
    generated = process_pipeline_output(video_data)

    # 保存原始模型输出
    raw_path = output_dir / "canny_control_raw.mp4"
    _save_video_file(generated, raw_path, fps=args.fps)
    logger.info(f"已保存原始模型输出视频: {raw_path}")

    # mask 合成
    composited = composite_with_mask(frames, generated, processed_masks)

    video_path = output_dir / "canny_control.mp4"
    _save_video_file(composited, video_path, fps=args.fps)
    logger.info(f"已保存合成目标视频: {video_path}")

    _save_first_frame_comparison(
        output_dir, "canny_control",
        filled_frames[0], processed_masks[0], composited[0], logger)

    _save_showcase(output_dir, "canny_control", composited, logger)

    # ------------------------------------------------------------------
    # 完成
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("实验完成！")
    logger.info(f"参考图传入策略: {strategy}")
    logger.info(f"输出目录: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
