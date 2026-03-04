"""
实验：中性色填充 + GrowMask + BlockifyMask + 参考图像 → 商品替换

输入：中性色填充后的模板视频帧 + 经 Grow+Blockify 处理的 mask + 参考商品图像 + 文本提示
输出：目标视频（mask 区域内被替换为参考商品）

核心改进：
1. 将输入帧的 mask 区域填充为中性灰色，消除 reactive 流先验
2. GrowMask（膨胀）覆盖分割边界残留像素
3. BlockifyMask 对齐到 VAE 下采样网格，消除亚像素边界伪影

此配置与 ComfyUI 成功工作流的 mask 处理方式一致。

用法：
    python exp_neutral_fill_growmask.py \
        --sample_dir ../../samples/teapot \
        --output_dir ../../experiments/results/neutral_fill_growmask \
        --grow_pixels 10 --block_size 32
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
    neutral_fill_frames,
    grow_masks,
    blockify_masks,
    filter_supported_kwargs,
    process_pipeline_output,
    composite_with_mask,
    save_experiment_outputs,
    add_common_args,
    _load_font,
)


def _save_mask_comparison(output_dir, precise_masks, processed_masks, logger):
    """保存 mask 对比图：[精确 Mask | Grow+Blockify Mask]（取第一帧）。"""
    w, h = precise_masks[0].size
    canvas = Image.new("RGB", (w * 2, h))
    canvas.paste(precise_masks[0].convert("RGB"), (0, 0))
    canvas.paste(processed_masks[0].convert("RGB"), (w, 0))

    draw = ImageDraw.Draw(canvas)
    font = _load_font(max(24, h // 20))
    draw.text((10, 10), "Precise Mask",
              fill="white", font=font, stroke_width=2, stroke_fill="black")
    draw.text((w + 10, 10), "Grow+Blockify",
              fill="white", font=font, stroke_width=2, stroke_fill="black")

    path = output_dir / "mask_comparison.jpg"
    canvas.save(path, quality=95)
    logger.info(f"已保存 mask 对比图: {path}")


def _save_preprocess_vis(output_dir, frames, filled_frames, processed_masks, logger):
    """保存预处理可视化：[原始帧 | 处理后 Mask | 填充帧]（取中间帧）。"""
    mid = len(frames) // 2
    w, h = frames[mid].size
    canvas = Image.new("RGB", (w * 3, h))
    canvas.paste(frames[mid], (0, 0))
    canvas.paste(processed_masks[mid].convert("RGB"), (w, 0))
    canvas.paste(filled_frames[mid], (w * 2, 0))

    draw = ImageDraw.Draw(canvas)
    font = _load_font(max(24, h // 20))
    for i, label in enumerate(["Original", "Processed Mask", "Neutral Fill"]):
        draw.text(
            (i * w + 10, 10), label,
            fill="white", font=font, stroke_width=2, stroke_fill="black",
        )

    path = output_dir / "preprocess_comparison.jpg"
    canvas.save(path, quality=95)
    logger.info(f"已保存预处理对比图: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="中性色填充 + GrowMask + BlockifyMask 商品替换实验",
    )
    add_common_args(parser)
    parser.add_argument(
        "--fill_value", type=int, default=128,
        help="中性色填充值（默认 128，即中性灰色）",
    )
    parser.add_argument(
        "--grow_pixels", type=int, default=10,
        help="Mask 膨胀像素数（默认 10）",
    )
    parser.add_argument(
        "--block_size", type=int, default=32,
        help="Mask 网格对齐块大小（默认 32）",
    )
    args = parser.parse_args()

    sample_dir = Path(args.sample_dir)
    output_dir = Path(args.output_dir)
    logger = setup_logger("exp_neutral_fill_growmask", output_dir)

    logger.info("=" * 60)
    logger.info("实验：中性色填充 + GrowMask + BlockifyMask → 商品替换")
    logger.info("=" * 60)
    logger.info(f"样本目录:        {sample_dir}")
    logger.info(f"输出目录:        {output_dir}")
    logger.info(f"分辨率:          {args.width}x{args.height}")
    logger.info(f"帧数:            {args.num_frames}")
    logger.info(f"模型大小:        {args.model_size}")
    logger.info(f"种子:            {args.seed}")
    logger.info(f"填充值:          {args.fill_value}")
    logger.info(f"Mask 膨胀像素:   {args.grow_pixels}")
    logger.info(f"Mask 网格大小:   {args.block_size}")
    logger.info(f"提示词:          {args.prompt}")

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
    ref_path = sample_dir / "reference_images" / args.reference_image
    reference = load_reference_image(ref_path, args.width, args.height)
    logger.info(
        f"已加载 {len(frames)} 帧视频、{len(precise_masks)} 个 mask、"
        f"参考图: {ref_path.name}",
    )

    # ------------------------------------------------------------------
    # 预处理：GrowMask + BlockifyMask
    # ------------------------------------------------------------------
    logger.info(f"Mask 预处理：GrowMask({args.grow_pixels}px) + "
                f"BlockifyMask({args.block_size}px)...")
    grown_masks = grow_masks(precise_masks, pixels=args.grow_pixels)
    processed_masks = blockify_masks(grown_masks, block_size=args.block_size)
    logger.info("Mask 预处理完成。")

    # 保存 mask 对比图（精确 mask vs 处理后 mask）
    _save_mask_comparison(output_dir, precise_masks, processed_masks, logger)

    # ------------------------------------------------------------------
    # 预处理：中性色填充输入帧（使用处理后的 mask）
    # ------------------------------------------------------------------
    logger.info(f"对输入帧处理后 mask 区域进行中性色填充（fill_value={args.fill_value}）...")
    filled_frames = neutral_fill_frames(
        frames, processed_masks, fill_value=args.fill_value,
    )
    logger.info("中性色填充完成。")

    # 保存预处理可视化
    _save_preprocess_vis(output_dir, frames, filled_frames, processed_masks, logger)

    # ------------------------------------------------------------------
    # 加载 Pipeline
    # ------------------------------------------------------------------
    logger.info(f"加载 Wan2.1-VACE-{args.model_size} pipeline...")
    pipe = load_vace_pipeline(
        model_size=args.model_size, device="cuda", torch_dtype=torch.bfloat16,
    )
    logger.info("Pipeline 加载完成。")

    # ------------------------------------------------------------------
    # 运行实验：neutral fill + processed mask + ref image + prompt
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("执行: 中性色填充帧 + Grow+Blockify mask + 参考图像")
    logger.info("=" * 60)

    pipe_kwargs = dict(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        vace_video=filled_frames,
        vace_video_mask=processed_masks,
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

    # 后处理：mask 外区域保持原视频（使用处理后的 mask 合成）
    composited = composite_with_mask(frames, generated, processed_masks)

    # 保存标准化输出（视频 + 对比帧 + 首末帧展示）
    save_experiment_outputs(
        output_dir, "neutral_fill_growmask", composited, filled_frames, processed_masks,
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
