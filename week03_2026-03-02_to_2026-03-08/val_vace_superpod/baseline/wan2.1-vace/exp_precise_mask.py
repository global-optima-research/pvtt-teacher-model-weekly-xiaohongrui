"""
实验：精确 Mask + 参考图像 → 商品替换

输入：模板视频（source video）+ 精确分割 mask + 参考商品图像 + 文本提示
输出：目标视频（mask 区域内被替换为参考商品）

这是 PVTT 最基础的实验：验证 VACE 能否在 zero-shot 条件下
将参考商品填入 mask 区域。

用法：
    python exp_precise_mask.py \
        --sample_dir ../../samples/teapot \
        --output_dir ../../experiments/results/precise_mask
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    setup_logger,
    load_vace_pipeline,
    load_video_frames,
    load_precise_masks,
    load_reference_image,
    filter_supported_kwargs,
    process_pipeline_output,
    composite_with_mask,
    save_experiment_outputs,
    add_common_args,
)


def main():
    parser = argparse.ArgumentParser(
        description="精确 Mask + 参考图像商品替换实验",
    )
    add_common_args(parser)
    args = parser.parse_args()

    sample_dir = Path(args.sample_dir)
    output_dir = Path(args.output_dir)
    logger = setup_logger("exp_precise_mask", output_dir)

    logger.info("=" * 60)
    logger.info("实验：精确 Mask + 参考图像 → 商品替换")
    logger.info("=" * 60)
    logger.info(f"样本目录:  {sample_dir}")
    logger.info(f"输出目录:  {output_dir}")
    logger.info(f"分辨率:    {args.width}x{args.height}")
    logger.info(f"帧数:      {args.num_frames}")
    logger.info(f"模型大小:  {args.model_size}")
    logger.info(f"种子:      {args.seed}")
    logger.info(f"提示词:    {args.prompt}")

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
    ref_path = sample_dir / "reference_images" / args.reference_image
    reference = load_reference_image(ref_path, args.width, args.height)
    logger.info(
        f"已加载 {len(frames)} 帧视频、{len(masks)} 个 mask、"
        f"参考图: {ref_path.name}",
    )

    # ------------------------------------------------------------------
    # 加载 Pipeline
    # ------------------------------------------------------------------
    logger.info(f"加载 Wan2.1-VACE-{args.model_size} pipeline...")
    pipe = load_vace_pipeline(
        model_size=args.model_size, device="cuda", torch_dtype=torch.bfloat16,
    )
    logger.info("Pipeline 加载完成。")

    # ------------------------------------------------------------------
    # 运行实验：source video + precise mask + ref image + prompt
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("执行: source video + precise mask + reference image")
    logger.info("=" * 60)

    pipe_kwargs = dict(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        vace_video=frames,
        vace_video_mask=masks,
        vace_reference_image=reference.resize((args.width, args.height)),
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        tiled=True,
    )
    pipe_kwargs = filter_supported_kwargs(pipe.__call__, pipe_kwargs)

    # 生成
    video_data = pipe(**pipe_kwargs)
    generated = process_pipeline_output(video_data)

    # 后处理：mask 外区域保持原视频
    composited = composite_with_mask(frames, generated, masks)

    # 保存标准化输出（视频 + 对比帧 + 首末帧展示）
    save_experiment_outputs(
        output_dir, "precise_mask", composited, frames, masks,
        fps=args.fps, logger=logger,
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
