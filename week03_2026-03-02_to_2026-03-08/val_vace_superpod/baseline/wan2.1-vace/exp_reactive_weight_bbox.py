"""
实验：Reactive 流权重调整（Bbox Mask）

同时使用两种策略：
  1. Bbox mask（减弱形状先验）
  2. Reactive 权重缩放（降低 mask 区域像素强度）

用法：
    python exp_reactive_weight_bbox.py \
        --sample_dir ../../samples/teapot \
        --output_dir ../../experiments/results/reactive_weight_bbox \
        --weights "0.0,0.3,0.5,0.7,1.0" \
        --bbox_margin_lr 20 --bbox_margin_top 20 --bbox_margin_bottom 20
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    setup_logger,
    load_vace_pipeline,
    load_video_frames,
    load_precise_masks,
    load_reference_image,
    create_bbox_masks_from_precise,
    apply_reactive_weight,
    filter_supported_kwargs,
    process_pipeline_output,
    composite_with_mask,
    save_experiment_outputs,
    save_weight_grid,
    add_common_args,
)


def main():
    parser = argparse.ArgumentParser(
        description="Reactive 流权重调整实验（Bbox Mask）",
    )
    add_common_args(parser)
    parser.add_argument(
        "--weights", type=str, default="0.0",
        help="逗号分隔的权重值，例如 '0.0,0.3,0.5,0.7,1.0'（默认 '0.0'）",
    )
    parser.add_argument("--bbox_margin_lr", type=int, default=20,
                        help="Bbox 左右扩展像素（默认 20）")
    parser.add_argument("--bbox_margin_top", type=int, default=20,
                        help="Bbox 上方扩展像素（默认 20）")
    parser.add_argument("--bbox_margin_bottom", type=int, default=20,
                        help="Bbox 下方扩展像素（默认 20）")
    args = parser.parse_args()

    weights = [float(w.strip()) for w in args.weights.split(",")]
    sample_dir = Path(args.sample_dir)
    output_dir = Path(args.output_dir)
    logger = setup_logger("exp_rw_bbox", output_dir)

    logger.info("=" * 60)
    logger.info("实验：Reactive 流权重调整（Bbox Mask）")
    logger.info("=" * 60)
    logger.info(f"样本目录:    {sample_dir}")
    logger.info(f"输出目录:    {output_dir}")
    logger.info(f"分辨率:      {args.width}x{args.height}")
    logger.info(f"帧数:        {args.num_frames}")
    logger.info(f"权重列表:    {weights}")
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

    # 从精确 mask 生成 bbox mask
    logger.info("生成 bbox mask...")
    bbox_masks = create_bbox_masks_from_precise(
        precise_masks,
        margin_lr=args.bbox_margin_lr,
        margin_top=args.bbox_margin_top,
        margin_bottom=args.bbox_margin_bottom,
    )
    logger.info(f"已加载 {len(frames)} 帧，已生成 {len(bbox_masks)} 个 bbox mask")

    # 保存 bbox mask 样例
    output_dir.mkdir(parents=True, exist_ok=True)
    bbox_masks[0].save(output_dir / "bbox_mask_frame0.png")

    # ------------------------------------------------------------------
    # 加载 Pipeline
    # ------------------------------------------------------------------
    logger.info(f"加载 Wan2.1-VACE-{args.model_size} pipeline...")
    pipe = load_vace_pipeline(
        model_size=args.model_size, device="cuda", torch_dtype=torch.bfloat16,
    )
    logger.info("Pipeline 加载完成。")

    # ------------------------------------------------------------------
    # 权重扫描
    # ------------------------------------------------------------------
    mid_frames = {}

    for weight in weights:
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"测试 Bbox Reactive Weight = {weight:.1f}")
        logger.info("=" * 60)

        weighted_frames = apply_reactive_weight(frames, bbox_masks, weight)

        pipe_kwargs = dict(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            vace_video=weighted_frames,
            vace_video_mask=[np.array(m) for m in bbox_masks],
            vace_reference_image=np.array(reference),
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.cfg_scale,
            embedded_cfg_scale=6.0,
            seed=args.seed,
            tiled=True,
        )
        pipe_kwargs = filter_supported_kwargs(pipe.__call__, pipe_kwargs)

        video_data = pipe(**pipe_kwargs)
        generated = process_pipeline_output(video_data)

        if len(generated) != args.num_frames:
            logger.warning(
                f"生成了 {len(generated)} 帧，预期 {args.num_frames} 帧",
            )

        composited = composite_with_mask(frames, generated, bbox_masks)

        exp_name = f"rw_bbox_w{weight:.1f}"
        weighted_pil = [Image.fromarray(f) for f in weighted_frames]
        save_experiment_outputs(
            output_dir, exp_name, composited, weighted_pil, bbox_masks,
            fps=args.fps, logger=logger,
        )

        mid = len(composited) // 2
        mid_frames[weight] = composited[mid]

    # ------------------------------------------------------------------
    # 权重网格对比图
    # ------------------------------------------------------------------
    if len(weights) > 1:
        save_weight_grid(
            output_dir, "rw_bbox", mid_frames,
            args.width, args.height, logger=logger,
        )

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
