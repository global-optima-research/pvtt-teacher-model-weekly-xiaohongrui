"""
实验：中性色填充 + 精确 Mask + 参考图像 → 商品替换

输入：中性色填充后的模板视频帧 + 精确分割 mask + 参考商品图像 + 文本提示
输出：目标视频（mask 区域内被替换为参考商品）

核心改进：将输入帧的 mask 区域填充为中性灰色（默认 128），
消除 VACE reactive 流中模板物体的先验影响，使模型更多依赖参考图像
和文本提示来生成内容。

用法：
    python exp_neutral_fill.py \
        --sample_dir ../../samples/teapot \
        --output_dir ../../experiments/results/neutral_fill
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
    filter_supported_kwargs,
    process_pipeline_output,
    composite_with_mask,
    save_experiment_outputs,
    add_common_args,
    _load_font,
)


def _save_preprocess_vis(output_dir, frames, filled_frames, masks, logger):
    """保存预处理可视化：[原始帧 | Mask | 填充帧]（取中间帧）。"""
    mid = len(frames) // 2
    w, h = frames[mid].size
    canvas = Image.new("RGB", (w * 3, h))
    canvas.paste(frames[mid], (0, 0))
    canvas.paste(masks[mid].convert("RGB"), (w, 0))
    canvas.paste(filled_frames[mid], (w * 2, 0))

    draw = ImageDraw.Draw(canvas)
    font = _load_font(max(24, h // 20))
    for i, label in enumerate(["Original", "Mask", "Neutral Fill"]):
        draw.text(
            (i * w + 10, 10), label,
            fill="white", font=font, stroke_width=2, stroke_fill="black",
        )

    path = output_dir / "preprocess_comparison.jpg"
    canvas.save(path, quality=95)
    logger.info(f"已保存预处理对比图: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="中性色填充 + 精确 Mask 商品替换实验",
    )
    add_common_args(parser)
    parser.add_argument(
        "--fill_value", type=int, default=128,
        help="中性色填充值（默认 128，即中性灰色）",
    )
    args = parser.parse_args()

    sample_dir = Path(args.sample_dir)
    output_dir = Path(args.output_dir)
    logger = setup_logger("exp_neutral_fill", output_dir)

    logger.info("=" * 60)
    logger.info("实验：中性色填充 + 精确 Mask → 商品替换")
    logger.info("=" * 60)
    logger.info(f"样本目录:  {sample_dir}")
    logger.info(f"输出目录:  {output_dir}")
    logger.info(f"分辨率:    {args.width}x{args.height}")
    logger.info(f"帧数:      {args.num_frames}")
    logger.info(f"模型大小:  {args.model_size}")
    logger.info(f"种子:      {args.seed}")
    logger.info(f"填充值:    {args.fill_value}")
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
    # 预处理：中性色填充输入帧
    # ------------------------------------------------------------------
    logger.info(f"对输入帧 mask 区域进行中性色填充（fill_value={args.fill_value}）...")
    filled_frames = neutral_fill_frames(frames, masks, fill_value=args.fill_value)
    logger.info("中性色填充完成。")

    # 保存预处理可视化
    _save_preprocess_vis(output_dir, frames, filled_frames, masks, logger)

    # ------------------------------------------------------------------
    # 加载 Pipeline
    # ------------------------------------------------------------------
    logger.info(f"加载 Wan2.1-VACE-{args.model_size} pipeline...")
    pipe = load_vace_pipeline(
        model_size=args.model_size, device="cuda", torch_dtype=torch.bfloat16,
    )
    logger.info("Pipeline 加载完成。")

    # ------------------------------------------------------------------
    # 运行实验：neutral fill frames + precise mask + ref image + prompt
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("执行: 中性色填充帧 + 精确 mask + 参考图像")
    logger.info("=" * 60)

    pipe_kwargs = dict(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        vace_video=filled_frames,
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
        output_dir, "neutral_fill", composited, filled_frames, masks,
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
