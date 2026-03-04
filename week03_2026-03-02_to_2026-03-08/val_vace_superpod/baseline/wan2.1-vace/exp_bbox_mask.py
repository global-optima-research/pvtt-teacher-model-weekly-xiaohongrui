"""
实验：Bbox Mask vs 精确 Mask 对比

假设：精确 mask 的形状先验过强，导致模型忽略参考图像。
矩形 bbox mask 可减弱形状约束，给模型更多自由度以利用参考图。

本实验运行两次 pipeline 调用：
  1. Bbox mask + 参考图像
  2. 精确 mask + 参考图像（作为对照）

用法：
    python exp_bbox_mask.py \
        --sample_dir ../../samples/teapot \
        --output_dir ../../experiments/results/bbox_mask \
        --bbox_margin_lr 20 --bbox_margin_top 20 --bbox_margin_bottom 20
"""

import argparse
import sys
from pathlib import Path

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
    create_bbox_masks_from_precise,
    filter_supported_kwargs,
    process_pipeline_output,
    composite_with_mask,
    save_experiment_outputs,
    add_common_args,
    _load_font,
)


def _side_by_side(left: Image.Image, right: Image.Image,
                  label_left: str, label_right: str) -> Image.Image:
    """创建两张图并排对比的图像。"""
    w, h = left.size
    canvas = Image.new("RGB", (w * 2, h))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (w, 0))
    draw = ImageDraw.Draw(canvas)
    font = _load_font(max(24, h // 20))
    draw.text((10, 10), label_left,
              fill="white", font=font, stroke_width=2, stroke_fill="black")
    draw.text((w + 10, 10), label_right,
              fill="white", font=font, stroke_width=2, stroke_fill="black")
    return canvas


def main():
    parser = argparse.ArgumentParser(
        description="Bbox Mask vs 精确 Mask 对比实验",
    )
    add_common_args(parser)
    parser.add_argument("--bbox_margin_lr", type=int, default=20,
                        help="Bbox 左右扩展像素（默认 20）")
    parser.add_argument("--bbox_margin_top", type=int, default=20,
                        help="Bbox 上方扩展像素（默认 20）")
    parser.add_argument("--bbox_margin_bottom", type=int, default=20,
                        help="Bbox 下方扩展像素（默认 20）")
    args = parser.parse_args()

    sample_dir = Path(args.sample_dir)
    output_dir = Path(args.output_dir)
    logger = setup_logger("exp_bbox_mask", output_dir)

    logger.info("=" * 60)
    logger.info("实验：Bbox Mask vs 精确 Mask 对比")
    logger.info("=" * 60)
    logger.info(f"样本目录:    {sample_dir}")
    logger.info(f"输出目录:    {output_dir}")
    logger.info(f"分辨率:      {args.width}x{args.height}")
    logger.info(f"帧数:        {args.num_frames}")
    logger.info(f"Bbox 边距:   lr={args.bbox_margin_lr}, "
                f"top={args.bbox_margin_top}, bottom={args.bbox_margin_bottom}")
    logger.info(f"模型大小:    {args.model_size}")
    logger.info(f"种子:        {args.seed}")

    # ------------------------------------------------------------------
    # 加载数据
    # ------------------------------------------------------------------
    logger.info("加载数据...")
    frames = load_video_frames(
        sample_dir / "video_frames", args.width, args.height, args.num_frames,
    )
    precise_masks = load_precise_masks(
        sample_dir / "masks", args.width, args.height, args.num_frames,
    )
    reference = load_reference_image(
        sample_dir / "reference_images" / args.reference_image,
        args.width, args.height,
    )
    logger.info(f"已加载 {len(frames)} 帧、{len(precise_masks)} 个 mask")

    # 从精确 mask 生成 bbox mask
    logger.info("生成 bbox mask...")
    bbox_masks = create_bbox_masks_from_precise(
        precise_masks,
        margin_lr=args.bbox_margin_lr,
        margin_top=args.bbox_margin_top,
        margin_bottom=args.bbox_margin_bottom,
    )

    # 保存 mask 对比图（第一帧）
    mask_comp = _side_by_side(
        precise_masks[0], bbox_masks[0], "Precise Mask", "Bbox Mask",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_comp.save(output_dir / "mask_comparison.jpg", quality=95)
    logger.info("已保存 mask 对比图（第 0 帧）。")

    # ------------------------------------------------------------------
    # 加载 Pipeline
    # ------------------------------------------------------------------
    logger.info(f"加载 Wan2.1-VACE-{args.model_size} pipeline...")
    pipe = load_vace_pipeline(
        model_size=args.model_size, device="cuda", torch_dtype=torch.bfloat16,
    )
    logger.info("Pipeline 加载完成。")

    # ------------------------------------------------------------------
    # 实验组 1：Bbox mask + 参考图像
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 50)
    logger.info("实验组 1: Bbox Mask + 参考图像")
    logger.info("=" * 50)

    pipe_kwargs = dict(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        vace_video=[np.array(f) for f in frames],
        vace_video_mask=[np.array(m) for m in bbox_masks],
        vace_reference_image=np.array(reference),
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
    )
    pipe_kwargs = filter_supported_kwargs(pipe.__call__, pipe_kwargs)

    video_data = pipe(**pipe_kwargs)
    generated_bbox = process_pipeline_output(video_data)
    composited_bbox = composite_with_mask(frames, generated_bbox, bbox_masks)

    save_experiment_outputs(
        output_dir, "bbox_mask", composited_bbox, frames, bbox_masks,
        fps=args.fps, logger=logger,
    )

    # ------------------------------------------------------------------
    # 对照组 2：精确 mask + 参考图像
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 50)
    logger.info("对照组 2: 精确 Mask + 参考图像")
    logger.info("=" * 50)

    pipe_kwargs2 = dict(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        vace_video=[np.array(f) for f in frames],
        vace_video_mask=[np.array(m) for m in precise_masks],
        vace_reference_image=np.array(reference),
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
    )
    pipe_kwargs2 = filter_supported_kwargs(pipe.__call__, pipe_kwargs2)

    video_data2 = pipe(**pipe_kwargs2)
    generated_precise = process_pipeline_output(video_data2)
    composited_precise = composite_with_mask(
        frames, generated_precise, precise_masks,
    )

    save_experiment_outputs(
        output_dir, "precise_mask", composited_precise, frames, precise_masks,
        fps=args.fps, logger=logger,
    )

    # ------------------------------------------------------------------
    # 并排对比（中间帧）
    # ------------------------------------------------------------------
    mid = len(composited_bbox) // 2
    comparison = _side_by_side(
        Image.fromarray(composited_precise[mid]),
        Image.fromarray(composited_bbox[mid]),
        "Precise Mask", "Bbox Mask",
    )
    comparison.save(output_dir / "side_by_side_comparison.jpg", quality=95)
    logger.info(f"已保存并排对比图（第 {mid} 帧）。")

    # ------------------------------------------------------------------
    # 汇总
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("实验完成！")
    logger.info(f"输出目录: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
