"""
实验：Canny 首帧粘贴（白线/黑线对比）+ 中性色填充 + GrowMask + BlockifyMask → 商品替换

输入：
- 视频帧（仅首帧 mask 区域嵌入 ref 图像的 Canny 边缘线条，其余帧纯中性色填充）
- GrowMask + BlockifyMask 处理后的 mask
- 参考商品图像 + 文本提示

两种变体（同一次运行中依次执行）：
1. 白色 Canny 线条：首帧 mask 区域中性灰背景 + 白色边缘 (255,255,255)
2. 黑色 Canny 线条：首帧 mask 区域中性灰背景 + 黑色边缘 (0,0,0)

核心思路：
仅在首帧提供目标物体的结构信息，测试模型能否从单帧结构提示传播到整个视频序列。
同时对比白色与黑色 Canny 线条的效果差异。

用法：
    python exp_canny_paste.py \\
        --sample_dir ../../samples/teapot \\
        --output_dir ../../experiments/results/canny_paste \\
        --canny_low 100 --canny_high 200
"""

import argparse
import sys
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
# Canny 边缘生成与嵌入
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
    resized = cv2.resize(canny_edges, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

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
                        line_color=(255, 255, 255), apply_to="all"):
    """创建嵌入 Canny 边缘的输入帧。

    Args:
        apply_to: 'all' = 所有帧, 'first' = 仅首帧
    """
    result = []
    for i, (frame, mask) in enumerate(zip(frames, masks)):
        arr = np.array(frame).copy()
        mb = np.array(mask.convert("L")) > 127
        arr[mb] = fill_value
        if apply_to == "all" or (apply_to == "first" and i == 0):
            arr = _paste_canny_in_mask_region(arr, mb, canny_edges, line_color)
        result.append(Image.fromarray(arr))
    return result


# =============================================================================
# 可视化辅助
# =============================================================================

def _save_canny_image(output_dir, canny_edges, logger):
    path = output_dir / "ref_canny.png"
    Image.fromarray(canny_edges).save(path)
    logger.info(f"已保存参考图 Canny 边缘图: {path}")


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


def _save_preprocess_comparison(output_dir, orig, mask_img, white_frame, black_frame, logger):
    """[原始帧 | 处理后 Mask | 白色 Canny 首帧 | 黑色 Canny 首帧]。"""
    w, h = orig.size
    canvas = Image.new("RGB", (w * 4, h))
    canvas.paste(orig, (0, 0))
    canvas.paste(mask_img.convert("RGB"), (w, 0))
    canvas.paste(white_frame, (w * 2, 0))
    canvas.paste(black_frame, (w * 3, 0))
    draw = ImageDraw.Draw(canvas)
    font = _load_font(max(20, h // 25))
    labels = ["Original", "Processed Mask", "White Canny (Frame 0)", "Black Canny (Frame 0)"]
    for i, label in enumerate(labels):
        draw.text((i * w + 10, 10), label,
                  fill="white", font=font, stroke_width=2, stroke_fill="black")
    path = output_dir / "preprocess_comparison.jpg"
    canvas.save(path, quality=95)
    logger.info(f"已保存预处理对比图: {path}")


def _save_combined_comparison(output_dir, input_white, input_black, mask_img,
                               target_white, target_black, logger):
    """2×2 网格对比图：白/黑两个变体的 VACE 输入与输出。

    布局：
        [VACE Input (白线)] [VACE Input (黑线)]
        [Target (白线)]     [Target (黑线)]
    """
    w, h = input_white.size
    tgt_w = Image.fromarray(target_white) if isinstance(target_white, np.ndarray) else target_white
    tgt_b = Image.fromarray(target_black) if isinstance(target_black, np.ndarray) else target_black

    canvas = Image.new("RGB", (w * 2, h * 2))
    canvas.paste(input_white, (0, 0))
    canvas.paste(input_black, (w, 0))
    canvas.paste(tgt_w, (0, h))
    canvas.paste(tgt_b, (w, h))

    draw = ImageDraw.Draw(canvas)
    font = _load_font(max(24, h // 20))
    labels = [
        (10, 10, "VACE Input (White Canny)"),
        (w + 10, 10, "VACE Input (Black Canny)"),
        (10, h + 10, "Target (White)"),
        (w + 10, h + 10, "Target (Black)"),
    ]
    for x, y, label in labels:
        draw.text((x, y), label,
                  fill="white", font=font, stroke_width=2, stroke_fill="black")

    path = output_dir / "canny_paste_comparison.jpg"
    canvas.save(path, quality=95)
    logger.info(f"已保存白/黑对比图（2×2 网格）: {path}")


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
    np_frames = [np.array(f) if isinstance(f, Image.Image) else f for f in frames]
    save_video(np_frames, str(path), fps=fps, quality=5)


# =============================================================================
# 单个变体的推理 + 保存
# =============================================================================

def _run_variant(pipe, pipe_base_kwargs, frames, processed_masks, canny_edges,
                 fill_value, line_color, variant_name, output_dir, fps, logger):
    """运行单个 Canny 变体（白/黑）的推理并保存所有输出。"""
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"变体：{variant_name}")
    logger.info("=" * 60)

    # 创建仅首帧嵌入 Canny 的输入帧
    canny_frames = create_canny_frames(
        frames, processed_masks, canny_edges,
        fill_value=fill_value,
        line_color=line_color,
        apply_to="first",
    )
    logger.info(f"已创建首帧 Canny 输入帧（线条颜色: {line_color}）")

    # 更新 vace_video
    kwargs = dict(pipe_base_kwargs)
    kwargs["vace_video"] = canny_frames

    # 推理
    logger.info("执行推理...")
    video_data = pipe(**kwargs)
    generated = process_pipeline_output(video_data)

    # 保存原始模型输出
    raw_path = output_dir / f"canny_paste_{variant_name}_raw.mp4"
    _save_video_file(generated, raw_path, fps=fps)
    logger.info(f"已保存原始模型输出视频: {raw_path}")

    # mask 合成
    composited = composite_with_mask(frames, generated, processed_masks)

    video_path = output_dir / f"canny_paste_{variant_name}.mp4"
    _save_video_file(composited, video_path, fps=fps)
    logger.info(f"已保存合成目标视频: {video_path}")

    # 首末帧展示（每个变体独立保存）
    _save_showcase(output_dir, f"canny_paste_{variant_name}", composited, logger)

    # 提取首帧用于后续合并对比图
    first_input = canny_frames[0]        # PIL Image（首帧含 Canny 线条）
    first_target = composited[0].copy()  # numpy array

    # 释放显存
    del video_data, generated, composited, canny_frames
    torch.cuda.empty_cache()

    logger.info(f"变体 {variant_name} 完成。")
    return first_input, first_target


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Canny 首帧粘贴（白线/黑线对比）+ GrowMask + BlockifyMask 商品替换实验",
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
                        help="背景亮度阈值：RGB 均 > 此值的像素视为白色背景，"
                             "在 Canny 前置零以消除低对比度边界（默认 240）")
    parser.add_argument("--blur_ksize", type=int, default=3,
                        help="Canny 前高斯模糊核大小（0 = 不模糊，默认 3）")
    parser.add_argument("--no_rembg", action="store_true", default=False,
                        help="禁用 rembg 前景分割，回退到亮度阈值去背景"
                             "（默认启用 rembg 以精确去除背景+阴影）")
    args = parser.parse_args()

    sample_dir = Path(args.sample_dir)
    output_dir = Path(args.output_dir)
    logger = setup_logger("exp_canny_paste", output_dir)

    logger.info("=" * 60)
    logger.info("实验：Canny 首帧粘贴（白线/黑线对比）→ 商品替换")
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
    logger.info(f"背景阈值:          {args.bg_threshold}")
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

    canny_edges = generate_canny(ref_for_canny, args.canny_low, args.canny_high,
                                  bg_threshold=args.bg_threshold,
                                  blur_ksize=args.blur_ksize)
    logger.info(f"Canny 尺寸: {canny_edges.shape}, "
                f"边缘像素: {np.count_nonzero(canny_edges)}")
    _save_canny_image(output_dir, canny_edges, logger)

    # ------------------------------------------------------------------
    # Mask 预处理
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
    # 预览两个变体的首帧
    # ------------------------------------------------------------------
    white_frames = create_canny_frames(
        frames, processed_masks, canny_edges,
        fill_value=args.fill_value, line_color=(255, 255, 255), apply_to="first")
    black_frames = create_canny_frames(
        frames, processed_masks, canny_edges,
        fill_value=args.fill_value, line_color=(0, 0, 0), apply_to="first")

    _save_preprocess_comparison(
        output_dir, frames[0], processed_masks[0],
        white_frames[0], black_frames[0], logger)
    del white_frames, black_frames  # 释放，后面在 _run_variant 中重新创建

    # ------------------------------------------------------------------
    # 加载 Pipeline（一次加载，两次推理）
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"加载 Wan2.1-VACE-{args.model_size} Pipeline")
    logger.info("=" * 60)
    pipe = load_vace_pipeline(
        model_size=args.model_size, device="cuda", torch_dtype=torch.bfloat16)
    logger.info("Pipeline 加载完成。")

    # 构建共享的 pipeline 参数（不含 vace_video，由各变体提供）
    pipe_base_kwargs = filter_supported_kwargs(pipe.__call__, dict(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        vace_video=[],  # placeholder, will be replaced
        vace_video_mask=processed_masks,
        vace_reference_image=reference.resize((args.width, args.height)),
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        tiled=True,
    ))

    # ------------------------------------------------------------------
    # 变体 1：白色 Canny 线条
    # ------------------------------------------------------------------
    white_input, white_target = _run_variant(
        pipe, pipe_base_kwargs, frames, processed_masks, canny_edges,
        fill_value=args.fill_value,
        line_color=(255, 255, 255),
        variant_name="white",
        output_dir=output_dir,
        fps=args.fps,
        logger=logger,
    )

    # ------------------------------------------------------------------
    # 变体 2：黑色 Canny 线条
    # ------------------------------------------------------------------
    black_input, black_target = _run_variant(
        pipe, pipe_base_kwargs, frames, processed_masks, canny_edges,
        fill_value=args.fill_value,
        line_color=(0, 0, 0),
        variant_name="black",
        output_dir=output_dir,
        fps=args.fps,
        logger=logger,
    )

    # ------------------------------------------------------------------
    # 白/黑合并对比图（2×2 网格）
    # ------------------------------------------------------------------
    _save_combined_comparison(
        output_dir, white_input, black_input, processed_masks[0],
        white_target, black_target, logger)

    # ------------------------------------------------------------------
    # 完成
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("实验完成！白色/黑色两个变体均已执行。")
    logger.info(f"输出目录: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
