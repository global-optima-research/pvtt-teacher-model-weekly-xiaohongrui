"""
实验：Reactive 流权重调整（精确 Mask）

策略：对 mask 区域的 vace_video 像素强度进行缩放，
降低 reactive 流的主导作用，使参考图像更容易生效。

  weight = 1.0  -->  原始行为（参考图往往被忽略）
  weight = 0.0  -->  完全置零（运动可能丢失）
  weight = 0.3-0.7  -->  甜点区间搜索

用法：
    python exp_reactive_weight.py \
        --sample_dir ../../samples/teapot \
        --output_dir ../../experiments/results/reactive_weight \
        --weights "0.0,0.3,0.5,0.7,1.0"
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
        description="Reactive 流权重调整实验（精确 Mask）",
    )
    add_common_args(parser)
    parser.add_argument(
        "--weights", type=str, default="0.0",
        help="逗号分隔的权重值，例如 '0.0,0.3,0.5,0.7,1.0'（默认 '0.0'）",
    )
    args = parser.parse_args()

    weights = [float(w.strip()) for w in args.weights.split(",")]
    sample_dir = Path(args.sample_dir)
    output_dir = Path(args.output_dir)
    logger = setup_logger("exp_reactive_weight", output_dir)

    logger.info("=" * 60)
    logger.info("实验：Reactive 流权重调整（精确 Mask）")
    logger.info("=" * 60)
    logger.info(f"样本目录:  {sample_dir}")
    logger.info(f"输出目录:  {output_dir}")
    logger.info(f"分辨率:    {args.width}x{args.height}")
    logger.info(f"帧数:      {args.num_frames}")
    logger.info(f"权重列表:  {weights}")
    logger.info(f"模型大小:  {args.model_size}")
    logger.info(f"种子:      {args.seed}")

    # ------------------------------------------------------------------
    # 加载数据
    # ------------------------------------------------------------------
    logger.info("加载数据...")
    frames = load_video_frames(
        sample_dir / "video_frames", args.width, args.height, args.num_frames,
    )
    masks = load_precise_masks(
        sample_dir / "masks", args.width, args.height, args.num_frames,
    )
    reference = load_reference_image(
        sample_dir / "reference_images" / args.reference_image,
        args.width, args.height,
    )
    logger.info(f"已加载 {len(frames)} 帧、{len(masks)} 个 mask")

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
    mid_frames = {}  # weight -> 中间帧 np.ndarray（用于网格对比）

    for weight in weights:
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"测试 Reactive Weight = {weight:.1f}")
        logger.info("=" * 60)

        weighted_frames = apply_reactive_weight(frames, masks, weight)

        pipe_kwargs = dict(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            vace_video=weighted_frames,
            vace_video_mask=[np.array(m) for m in masks],
            vace_reference_image=np.array(reference),
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.cfg_scale,
            embedded_cfg_scale=6.0,
            seed=args.seed,
        )
        pipe_kwargs = filter_supported_kwargs(pipe.__call__, pipe_kwargs)

        video_data = pipe(**pipe_kwargs)
        generated = process_pipeline_output(video_data)
        composited = composite_with_mask(frames, generated, masks)

        exp_name = f"reactive_weight_w{weight:.1f}"
        weighted_pil = [Image.fromarray(f) for f in weighted_frames]
        save_experiment_outputs(
            output_dir, exp_name, composited, weighted_pil, masks,
            fps=args.fps, logger=logger,
        )

        mid = len(composited) // 2
        mid_frames[weight] = composited[mid]

    # ------------------------------------------------------------------
    # 权重网格对比图（仅在多个权重时生成）
    # ------------------------------------------------------------------
    if len(weights) > 1:
        save_weight_grid(
            output_dir, "reactive_weight", mid_frames,
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
