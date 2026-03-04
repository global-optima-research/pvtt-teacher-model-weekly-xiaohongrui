"""
实验：1:1 复刻 ComfyUI VACE 商品替换工作流

完整复刻 ComfyUI 工作流（BV1iG4yzTEwh）的所有关键配置：

  路径 A  参考图预处理：
    LoadImage -> BiRefNetUltra(前景分割) -> InvertMask -> INPAINT_MaskedFill(neutral)
    => 参考图背景被替换为中性灰色，仅保留商品前景

  路径 B  输入帧预处理：
    LoadVideo -> Resize -> INPAINT_MaskedFill(neutral, mask=processed_mask)
    => mask 区域被填充为中性灰色

  路径 C  Mask 预处理：
    SAM2 segmentation -> GrowMask(10px, tapered) -> BlockifyMask(32px)
    => 本实验直接使用预先生成的精确 mask 代替 SAM2

  模型配置：
    Wan2.1-VACE-14B (bf16, 而非工作流的 fp8) + LightX2V 蒸馏 LoRA (rank32, strength=1.0)

  采样配置：
    steps=1, cfg=6.0, cfg_star=5.0, scheduler=dpm++_sde, shift=true

  后处理：
    生成结果 + 原始帧 + mask -> Composite

用法：
    python exp_comfyui_baseline.py \\
        --sample_dir ../../samples/teapot \\
        --output_dir ../../experiments/results/14B/comfyui_baseline \\
        --lora_path ../../models/LightX2V/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors
"""

import argparse
import sys
import traceback
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


# =============================================================================
# 参考图去背景（复刻 BiRefNetUltra + InvertMask + INPAINT_MaskedFill(neutral)）
# =============================================================================

def remove_ref_background(image, fill_value=128, model_name="birefnet-general", logger=None):
    """去除参考图背景，填充为中性色。

    复刻 ComfyUI 工作流中的数据流：
      LoadImage -> BiRefNetUltra -> InvertMask -> INPAINT_MaskedFill(neutral)

    BiRefNetUltra 提取前景 mask，反转后得到背景 mask，
    再用 neutral fill 把背景替换为中性灰色。
    最终效果：商品前景保留，背景变为灰色。

    本函数使用 rembg 库实现等效功能（rembg 内置 BiRefNet 模型）。

    Args:
        image:      PIL RGB 参考图像。
        fill_value: 背景填充灰度值（默认 128，中性灰）。
        model_name: rembg 模型名称（birefnet-general 对应 ComfyUI BiRefNetUltra）。
        logger:     日志器。

    Returns:
        去背景并填充中性色的 PIL RGB 图像。
    """
    _log = logger.info if logger else print
    _warn = logger.warning if logger else print

    try:
        from rembg import remove, new_session
    except ImportError:
        _warn("rembg 未安装，跳过参考图去背景。")
        _warn("安装方法: pip install rembg[gpu]")
        return image

    # 尝试使用 BiRefNet（与 ComfyUI BiRefNetUltra 对应）
    try:
        session = new_session(model_name)
        rgba = remove(image.convert("RGB"), session=session)
        _log(f"使用 {model_name} 模型去除参考图背景。")
    except Exception as e:
        _log(f"BiRefNet 模型加载失败 ({e})，回退到 rembg 默认模型。")
        rgba = remove(image.convert("RGB"))
        _log("使用 rembg 默认模型去除参考图背景。")

    # 创建中性色背景并合成前景
    bg = Image.new("RGB", image.size, (fill_value, fill_value, fill_value))
    if rgba.mode == "RGBA":
        bg.paste(rgba, mask=rgba.split()[3])
    else:
        bg = rgba.convert("RGB")

    return bg


# =============================================================================
# LoRA 加载（复刻 WanVideoLoraSelect + WanVideoModelLoader 的 LoRA 合并）
# =============================================================================

def load_and_merge_lora(pipe, lora_path, strength=1.0, logger=None):
    """加载 LightX2V 蒸馏 LoRA 并合并到 pipeline 的 denoising model 中。

    ComfyUI 工作流配置：
      WanVideoLoraSelect:
        lora = Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors
        strength = 1.0
        merge_loras = true

    依次尝试：
      1. DiffSynth-Studio 原生 LoRA API（如果存在）
      2. 手动 safetensors 加载 + 权重合并

    Args:
        pipe:       WanVideoPipeline 实例。
        lora_path:  LoRA safetensors 文件的绝对路径。
        strength:   LoRA 强度（默认 1.0）。
        logger:     日志器。

    Returns:
        bool: 是否成功加载。
    """
    _log = logger.info if logger else print
    _warn = logger.warning if logger else print

    lora_path = Path(lora_path)
    if not lora_path.exists():
        _warn(f"LoRA 文件不存在: {lora_path}")
        return False

    _log(f"加载 LoRA: {lora_path.name} (strength={strength})")

    # 方法 1：尝试 DiffSynth-Studio 原生 API
    for method_name in ["load_lora", "add_lora", "merge_lora"]:
        if hasattr(pipe, method_name):
            try:
                method = getattr(pipe, method_name)
                method(str(lora_path), strength=strength)
                _log(f"通过 pipe.{method_name}() 成功加载 LoRA。")
                return True
            except Exception as e:
                _log(f"pipe.{method_name}() 失败: {e}")

    # 方法 2：手动合并
    try:
        return _manual_merge_lora(pipe, lora_path, strength, logger)
    except Exception as e:
        _warn(f"手动合并 LoRA 失败: {e}")
        _warn(traceback.format_exc())
        return False


def _find_denoising_model(pipe):
    """在 pipeline 中查找 denoising model（nn.Module）。"""
    import torch.nn as nn

    # DiffSynth-Studio 常见属性名
    for attr in [
        "denoising_model", "dit", "transformer", "unet", "model",
        "dit_model", "wan_model", "diffusion_model",
    ]:
        if hasattr(pipe, attr):
            model = getattr(pipe, attr)
            if model is not None and isinstance(model, nn.Module):
                return attr, model

    # 遍历所有属性
    for attr in dir(pipe):
        if attr.startswith("_"):
            continue
        try:
            obj = getattr(pipe, attr)
            if isinstance(obj, nn.Module) and sum(1 for _ in obj.parameters()) > 1000:
                return attr, obj
        except Exception:
            continue

    return None, None


def _manual_merge_lora(pipe, lora_path, strength=1.0, logger=None):
    """手动将 LoRA 权重合并到 denoising model。

    LoRA 合并公式：W' = W + strength * (alpha / rank) * B @ A
    """
    _log = logger.info if logger else print

    from safetensors import safe_open

    # 1. 加载 LoRA state dict
    lora_state = {}
    with safe_open(str(lora_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            lora_state[key] = f.get_tensor(key)
    _log(f"LoRA 文件包含 {len(lora_state)} 个张量。")

    # 2. 定位 denoising model
    attr_name, model = _find_denoising_model(pipe)
    if model is None:
        raise RuntimeError(
            "无法在 pipeline 中找到 denoising model。"
            "请检查 DiffSynth-Studio 版本或 pipeline 结构。"
        )
    _log(f"找到 denoising model: pipe.{attr_name}")

    model_params = dict(model.named_parameters())
    model_keys = set(model_params.keys())
    _log(f"模型包含 {len(model_keys)} 个参数。")

    # 3. 解析 LoRA 对（A/B 矩阵）和 alpha 值
    lora_pairs = {}
    alpha_dict = {}
    for key, tensor in lora_state.items():
        if "alpha" in key:
            base = key.rsplit(".alpha", 1)[0]
            alpha_dict[base] = tensor.item()
        elif "lora_A" in key or "lora_down" in key:
            base = (key
                    .replace(".lora_A.weight", "")
                    .replace(".lora_down.weight", "")
                    .replace(".lora_A", "")
                    .replace(".lora_down", ""))
            lora_pairs.setdefault(base, {})["A"] = tensor
        elif "lora_B" in key or "lora_up" in key:
            base = (key
                    .replace(".lora_B.weight", "")
                    .replace(".lora_up.weight", "")
                    .replace(".lora_B", "")
                    .replace(".lora_up", ""))
            lora_pairs.setdefault(base, {})["B"] = tensor

    _log(f"识别到 {len(lora_pairs)} 个 LoRA 层对，{len(alpha_dict)} 个 alpha 值。")

    if not lora_pairs:
        raise RuntimeError("LoRA 文件中未找到有效的 LoRA 层对。")

    # 打印示例 key 帮助调试
    sample_lora_keys = list(lora_pairs.keys())[:3]
    sample_model_keys = sorted(model_keys)[:3]
    _log(f"LoRA 样例 base key: {sample_lora_keys}")
    _log(f"模型样例 key: {sample_model_keys}")

    # 4. 构建 key 映射并合并
    merged = 0
    skipped = 0

    # 预构建常见前缀映射
    prefix_combinations = [
        ("", ""),                              # 直接匹配
        ("", "model."),                        # model. 前缀
        ("", "transformer."),
        ("", "diffusion_model."),
        ("transformer.", ""),                  # 去 transformer. 前缀
        ("diffusion_model.", ""),
        ("model.", ""),
        ("transformer.", "model."),
        ("diffusion_model.", "model."),
        ("model.diffusion_model.", ""),
    ]

    for base_key, pair in lora_pairs.items():
        if "A" not in pair or "B" not in pair:
            skipped += 1
            continue

        # 尝试各种 key 映射找到目标参数
        target_key = base_key + ".weight"
        actual_key = None

        for lora_prefix, model_prefix in prefix_combinations:
            candidate = target_key
            if lora_prefix and candidate.startswith(lora_prefix):
                candidate = candidate[len(lora_prefix):]
            candidate = model_prefix + candidate
            if candidate in model_keys:
                actual_key = candidate
                break

        if actual_key is None:
            skipped += 1
            continue

        # 计算 LoRA delta: W += strength * (alpha / rank) * B @ A
        A = pair["A"].float()
        B = pair["B"].float()

        rank = A.shape[0]
        alpha = alpha_dict.get(base_key, float(rank))
        scale = alpha / rank

        param = model_params[actual_key]
        delta = (strength * scale * (B @ A)).to(device=param.device, dtype=param.dtype)

        if delta.shape != param.shape:
            skipped += 1
            continue

        with torch.no_grad():
            param.data.add_(delta)
        merged += 1

    _log(f"已合并 {merged} 个 LoRA 层，跳过 {skipped} 个。")

    if merged == 0:
        raise RuntimeError(
            f"未能合并任何 LoRA 层。LoRA 文件与模型的命名约定可能不兼容。\n"
            f"LoRA base keys 样例: {sample_lora_keys}\n"
            f"模型 keys 样例: {sample_model_keys}"
        )

    return True


# =============================================================================
# 可视化辅助
# =============================================================================

def _save_ref_comparison(output_dir, ref_original, ref_nobg, logger):
    """保存参考图预处理对比图：[原始参考图 | 去背景参考图]。"""
    w, h = ref_original.size
    canvas = Image.new("RGB", (w * 2, h))
    canvas.paste(ref_original, (0, 0))
    canvas.paste(ref_nobg, (w, 0))

    draw = ImageDraw.Draw(canvas)
    font = _load_font(max(24, h // 20))
    draw.text((10, 10), "Original Ref",
              fill="white", font=font, stroke_width=2, stroke_fill="black")
    draw.text((w + 10, 10), "No-BG Ref (neutral fill)",
              fill="white", font=font, stroke_width=2, stroke_fill="black")

    path = output_dir / "ref_comparison.jpg"
    canvas.save(path, quality=95)
    logger.info(f"已保存参考图对比: {path}")


def _save_mask_comparison(output_dir, precise_masks, processed_masks, logger):
    """保存 mask 对比图：[精确 Mask | Grow+Blockify Mask]。"""
    w, h = precise_masks[0].size
    canvas = Image.new("RGB", (w * 2, h))
    canvas.paste(precise_masks[0].convert("RGB"), (0, 0))
    canvas.paste(processed_masks[0].convert("RGB"), (w, 0))

    draw = ImageDraw.Draw(canvas)
    font = _load_font(max(24, h // 20))
    draw.text((10, 10), "Precise Mask",
              fill="white", font=font, stroke_width=2, stroke_fill="black")
    draw.text((w + 10, 10), "Grow(10)+Block(32)",
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


# =============================================================================
# 主流程
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="1:1 复刻 ComfyUI VACE 商品替换工作流",
    )
    add_common_args(parser)

    # ComfyUI 工作流特有参数
    parser.add_argument(
        "--lora_path", type=str, default=None,
        help="LightX2V 蒸馏 LoRA 路径（safetensors 格式）",
    )
    parser.add_argument(
        "--lora_strength", type=float, default=1.0,
        help="LoRA 强度（默认 1.0，与 ComfyUI 一致）",
    )
    parser.add_argument(
        "--embedded_cfg_scale", type=float, default=5.0,
        help="cfg_star / embedded_cfg_scale（默认 5.0，与 ComfyUI 一致）",
    )
    parser.add_argument(
        "--fill_value", type=int, default=128,
        help="中性色填充值（默认 128）",
    )
    parser.add_argument(
        "--grow_pixels", type=int, default=10,
        help="Mask 膨胀像素数（默认 10，与 ComfyUI GrowMask 一致）",
    )
    parser.add_argument(
        "--block_size", type=int, default=32,
        help="Mask 网格对齐块大小（默认 32，与 ComfyUI BlockifyMask 一致）",
    )
    parser.add_argument(
        "--rembg_model", type=str, default="birefnet-general",
        help="rembg 模型名称（默认 birefnet-general，对应 BiRefNetUltra）",
    )
    args = parser.parse_args()

    sample_dir = Path(args.sample_dir)
    output_dir = Path(args.output_dir)
    logger = setup_logger("exp_comfyui_baseline", output_dir)

    # 推断 LoRA 默认路径
    if args.lora_path is None:
        project_dir = Path(__file__).parent.parent.parent
        args.lora_path = str(
            project_dir / "models" / "LightX2V"
            / "Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors"
        )

    logger.info("=" * 60)
    logger.info("实验：1:1 复刻 ComfyUI VACE 商品替换工作流")
    logger.info("=" * 60)
    logger.info(f"样本目录:          {sample_dir}")
    logger.info(f"输出目录:          {output_dir}")
    logger.info(f"分辨率:            {args.width}x{args.height}")
    logger.info(f"帧数:              {args.num_frames}")
    logger.info(f"模型大小:          {args.model_size}")
    logger.info(f"种子:              {args.seed}")
    logger.info(f"LoRA 路径:         {args.lora_path}")
    logger.info(f"LoRA 强度:         {args.lora_strength}")
    logger.info(f"采样步数:          {args.num_inference_steps}")
    logger.info(f"CFG:               {args.cfg_scale}")
    logger.info(f"CFG Star:          {args.embedded_cfg_scale}")
    logger.info(f"填充值:            {args.fill_value}")
    logger.info(f"Mask 膨胀像素:     {args.grow_pixels}")
    logger.info(f"Mask 网格大小:     {args.block_size}")
    logger.info(f"参考图去背景模型:  {args.rembg_model}")
    logger.info(f"提示词:            {args.prompt}")
    logger.info(f"负面提示词:        {args.negative_prompt}")

    # ------------------------------------------------------------------
    # 1. 加载数据
    # ------------------------------------------------------------------
    logger.info("")
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
    # 2. 路径 A：参考图去背景（BiRefNetUltra → rembg）
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("路径 A：参考图去背景")
    logger.info("=" * 60)
    ref_nobg = remove_ref_background(
        reference, fill_value=args.fill_value,
        model_name=args.rembg_model, logger=logger,
    )
    _save_ref_comparison(output_dir, reference, ref_nobg, logger)

    # ------------------------------------------------------------------
    # 3. 路径 C：Mask 预处理 GrowMask(10px) + BlockifyMask(32px)
    #    （在路径 B 之前，因为填充需要使用处理后的 mask）
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"路径 C：Mask 预处理 GrowMask({args.grow_pixels}px) + "
                f"BlockifyMask({args.block_size}px)")
    logger.info("=" * 60)
    grown_masks = grow_masks(precise_masks, pixels=args.grow_pixels)
    processed_masks = blockify_masks(grown_masks, block_size=args.block_size)
    logger.info("Mask 预处理完成。")
    _save_mask_comparison(output_dir, precise_masks, processed_masks, logger)

    # ------------------------------------------------------------------
    # 4. 路径 B：输入帧中性色填充（INPAINT_MaskedFill(neutral)）
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"路径 B：输入帧 mask 区域中性色填充 (fill_value={args.fill_value})")
    logger.info("=" * 60)
    filled_frames = neutral_fill_frames(
        frames, processed_masks, fill_value=args.fill_value,
    )
    logger.info("中性色填充完成。")
    _save_preprocess_vis(output_dir, frames, filled_frames, processed_masks, logger)

    # ------------------------------------------------------------------
    # 5. 加载 Pipeline + LoRA
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"加载 Wan2.1-VACE-{args.model_size} Pipeline")
    logger.info("=" * 60)
    pipe = load_vace_pipeline(
        model_size=args.model_size, device="cuda", torch_dtype=torch.bfloat16,
    )
    logger.info("Pipeline 加载完成。")

    # 加载 LoRA
    lora_loaded = False
    if Path(args.lora_path).exists():
        logger.info("")
        logger.info("=" * 60)
        logger.info("加载 LightX2V 蒸馏 LoRA")
        logger.info("=" * 60)
        lora_loaded = load_and_merge_lora(
            pipe, args.lora_path,
            strength=args.lora_strength, logger=logger,
        )
    else:
        logger.warning(f"LoRA 文件不存在: {args.lora_path}")

    if not lora_loaded:
        logger.warning("=" * 60)
        logger.warning("LoRA 未加载！这将显著影响实验结果：")
        logger.warning("  - 蒸馏 LoRA 是 ComfyUI 工作流的核心组件")
        logger.warning("  - 无 LoRA 时 steps=1 会产生低质量结果")
        if args.num_inference_steps <= 1:
            args.num_inference_steps = 50
            logger.warning(f"  - 已自动将 steps 从 1 调整为 {args.num_inference_steps}")
        logger.warning("=" * 60)

    # ------------------------------------------------------------------
    # 6. 运行推理
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("执行推理: 中性色填充帧 + Grow+Block mask + 去背景参考图")
    logger.info(f"  steps={args.num_inference_steps}, cfg={args.cfg_scale}, "
                f"cfg_star={args.embedded_cfg_scale}, LoRA={'YES' if lora_loaded else 'NO'}")
    logger.info("=" * 60)

    pipe_kwargs = dict(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        vace_video=filled_frames,
        vace_video_mask=processed_masks,
        vace_reference_image=ref_nobg.resize((args.width, args.height)),
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.cfg_scale,
        embedded_cfg_scale=args.embedded_cfg_scale,
        seed=args.seed,
        tiled=True,
    )
    pipe_kwargs = filter_supported_kwargs(pipe.__call__, pipe_kwargs)

    video_data = pipe(**pipe_kwargs)
    generated = process_pipeline_output(video_data)

    # ------------------------------------------------------------------
    # 7. 后处理：mask 合成（mask 外保持原视频）
    # ------------------------------------------------------------------
    composited = composite_with_mask(frames, generated, processed_masks)

    # ------------------------------------------------------------------
    # 8. 保存标准化输出
    # ------------------------------------------------------------------
    save_experiment_outputs(
        output_dir, "comfyui_baseline", composited, filled_frames, processed_masks,
        fps=args.fps, logger=logger,
    )

    # ------------------------------------------------------------------
    # 汇总
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("实验完成！")
    logger.info(f"LoRA 状态: {'已加载' if lora_loaded else '未加载（结果可能不准确）'}")
    logger.info(f"输出目录: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
