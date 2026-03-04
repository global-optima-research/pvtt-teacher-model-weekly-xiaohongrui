"""
PVTT（Product Video Template Transformation）—— 共享工具库

为所有 VACE 实验提供统一的基础功能：
- Pipeline 加载（1.3B / 14B）
- 数据加载（视频帧、分割 mask、参考图像）
- Bbox mask 生成（从精确 mask 派生）
- Reactive 流权重调节
- Pipeline 输出格式统一处理
- 后处理（mask 合成、对比帧、首末帧展示）
- 日志配置
- 通用 CLI 参数解析
"""

import os
import sys
import inspect
import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

# ---- 将 DiffSynth-Studio 加入 Python 路径 ----
_DIFFSYNTH_PATH = Path(__file__).parent.parent.parent / "DiffSynth-Studio"
if str(_DIFFSYNTH_PATH) not in sys.path:
    sys.path.insert(0, str(_DIFFSYNTH_PATH))

from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.utils.data import save_video


# =============================================================================
# 日志
# =============================================================================

def setup_logger(
    name: str,
    output_dir: Path = None,
    level=logging.INFO,
) -> logging.Logger:
    """配置日志器：同时输出到控制台 + 可选的文件。

    Args:
        name:       日志器名称（通常为实验名）。
        output_dir: 若提供，同时写入 ``output_dir/experiment.log``。
        level:      日志级别。
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台输出
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # 文件输出
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(output_dir / "experiment.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# =============================================================================
# Pipeline 加载
# =============================================================================

def load_vace_pipeline(
    model_size: str = "1.3B",
    device: str = "cuda",
    torch_dtype=torch.bfloat16,
):
    """加载 Wan2.1-VACE Pipeline，支持 1.3B 和 14B。

    环境变量 ``WAN_VACE_MODEL_SIZE`` 可覆盖 *model_size* 参数。
    14B 模型会自动启用 CPU offload 以节省显存。
    """
    env_size = os.environ.get("WAN_VACE_MODEL_SIZE")
    if env_size:
        model_size = env_size

    assert model_size in ("1.3B", "14B"), f"不支持的模型大小: {model_size}"

    base_model_id = f"Wan-AI/Wan2.1-VACE-{model_size}"
    tokenizer_id = "Wan-AI/Wan2.1-T2V-1.3B"

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=device,
        model_configs=[
            ModelConfig(
                model_id=base_model_id,
                origin_file_pattern="diffusion_pytorch_model*.safetensors",
            ),
            ModelConfig(
                model_id=base_model_id,
                origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
            ),
            ModelConfig(
                model_id=base_model_id,
                origin_file_pattern="Wan2.1_VAE.pth",
            ),
        ],
        tokenizer_config=ModelConfig(
            model_id=tokenizer_id,
            origin_file_pattern="google/umt5-xxl/",
        ),
    )

    # 14B 自动开启 CPU offload
    if model_size == "14B":
        if hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        elif hasattr(pipe, "enable_sequential_cpu_offload"):
            pipe.enable_sequential_cpu_offload()

    return pipe


# =============================================================================
# 数据加载
# =============================================================================

def load_video_frames(
    video_dir: Path,
    width: int,
    height: int,
    num_frames: int,
) -> list[Image.Image]:
    """从目录加载 *num_frames* 帧视频帧，缩放至 (width, height)。"""
    frame_files = sorted(
        list(video_dir.glob("*.png")) + list(video_dir.glob("*.jpg"))
    )[:num_frames]
    if len(frame_files) < num_frames:
        raise ValueError(
            f"需要 {num_frames} 帧，但 {video_dir} 下只有 {len(frame_files)} 帧"
        )
    return [Image.open(f).convert("RGB").resize((width, height)) for f in frame_files]


def load_precise_masks(
    mask_dir: Path,
    width: int,
    height: int,
    num_frames: int,
) -> list[Image.Image]:
    """加载精确分割 mask，返回二值化的 RGB 图像（像素值 0 或 255）。"""
    mask_files = sorted(
        list(mask_dir.glob("*.png")) + list(mask_dir.glob("*.jpg"))
    )[:num_frames]
    if len(mask_files) < num_frames:
        raise ValueError(
            f"需要 {num_frames} 个 mask，但 {mask_dir} 下只有 {len(mask_files)} 个"
        )
    masks = []
    for f in mask_files:
        m = Image.open(f).convert("L").resize((width, height), Image.NEAREST)
        m = m.point(lambda p: 255 if p > 127 else 0)
        masks.append(m.convert("RGB"))
    return masks


def load_reference_image(
    ref_path: Path,
    width: int,
    height: int,
) -> Image.Image:
    """加载并缩放参考商品图像。"""
    return Image.open(ref_path).convert("RGB").resize((width, height), Image.BICUBIC)


# =============================================================================
# Bbox 工具
# =============================================================================

def get_bbox_from_mask(mask_image: Image.Image):
    """从二值 mask 中提取 bounding box。

    Returns:
        ``(x_min, y_min, x_max, y_max)``；mask 为空时返回 ``None``。
    """
    mask_array = np.array(mask_image.convert("L"))
    coords = np.argwhere(mask_array > 127)
    if len(coords) == 0:
        return None
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return (int(x_min), int(y_min), int(x_max), int(y_max))


def create_bbox_mask(
    width: int,
    height: int,
    bbox,
    margin: int = 0,
    margin_left: int | None = None,
    margin_right: int | None = None,
    margin_top: int | None = None,
    margin_bottom: int | None = None,
) -> Image.Image:
    """创建矩形 bbox mask（RGB，像素值 0/255），支持四个方向独立设置扩展边距。

    若某个方向的 margin 为 ``None``，则使用统一的 *margin* 值。
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    x_min, y_min, x_max, y_max = bbox

    ml = margin if margin_left is None else margin_left
    mr = margin if margin_right is None else margin_right
    mt = margin if margin_top is None else margin_top
    mb = margin if margin_bottom is None else margin_bottom

    x_min = max(0, x_min - ml)
    y_min = max(0, y_min - mt)
    x_max = min(width - 1, x_max + mr)
    y_max = min(height - 1, y_max + mb)

    mask[y_min : y_max + 1, x_min : x_max + 1] = 255
    return Image.fromarray(mask, mode="L").convert("RGB")


def create_bbox_masks_from_precise(
    precise_masks: list[Image.Image],
    margin_lr: int = 20,
    margin_top: int = 20,
    margin_bottom: int = 20,
) -> list[Image.Image]:
    """将精确 mask 序列转换为 bbox mask 序列。

    Args:
        precise_masks: 精确分割 mask 列表（RGB）。
        margin_lr:     左右扩展像素数。
        margin_top:    上方扩展像素数。
        margin_bottom: 下方扩展像素数。

    Returns:
        二值化 RGB bbox mask 列表。
    """
    bbox_masks = []
    width, height = precise_masks[0].size
    for pm in precise_masks:
        bbox = get_bbox_from_mask(pm)
        if bbox is None:
            bbox_mask = Image.new("RGB", (width, height), (0, 0, 0))
        else:
            bbox_mask = create_bbox_mask(
                width, height, bbox,
                margin=0,
                margin_left=margin_lr,
                margin_right=margin_lr,
                margin_top=margin_top,
                margin_bottom=margin_bottom,
            )
        # 强制二值化
        m = bbox_mask.convert("L").point(lambda p: 255 if p > 127 else 0)
        bbox_masks.append(m.convert("RGB"))
    return bbox_masks


# =============================================================================
# Reactive 流权重
# =============================================================================

def apply_reactive_weight(
    frames: list[Image.Image],
    masks: list[Image.Image],
    weight: float = 0.3,
) -> list[np.ndarray]:
    """对 mask 区域的像素强度进行缩放，模拟降低 reactive 流权重。

    ``weight = 0.0``：完全置零（reactive 流完全被抑制）。
    ``weight = 1.0``：保持原始像素不变。

    Returns:
        ``List[np.ndarray]``，dtype ``uint8``，shape ``(H, W, 3)``。
    """
    weighted_frames = []
    for frame, mask in zip(frames, masks):
        frame_arr = np.array(frame).astype(np.float32)
        mask_arr = np.array(mask.convert("L")).astype(np.float32) / 255.0
        factor = mask_arr[..., None] * weight + (1.0 - mask_arr[..., None])
        weighted = frame_arr * factor
        weighted_frames.append(np.clip(weighted, 0, 255).astype(np.uint8))
    return weighted_frames


# =============================================================================
# 输入预处理：Neutral Fill（中性色填充）
# =============================================================================

def neutral_fill_frames(
    frames: list[Image.Image],
    masks: list[Image.Image],
    fill_value: int = 128,
) -> list[Image.Image]:
    """将输入帧的 mask 区域填充为中性色（默认灰色 128）。

    用于消除 VACE reactive 流中模板物体的先验影响，
    使模型在 mask 区域更多依赖参考图像和文本提示来生成内容。

    Args:
        frames:     视频帧列表（PIL RGB）。
        masks:      对应的 mask 列表（PIL RGB，白色=需要填充的区域）。
        fill_value: 填充灰度值（默认 128，即中性灰色）。

    Returns:
        填充后的视频帧列表（PIL RGB）。
    """
    filled = []
    for frame, mask in zip(frames, masks):
        frame_arr = np.array(frame).copy()
        mask_arr = np.array(mask.convert("L")) > 127
        frame_arr[mask_arr] = fill_value
        filled.append(Image.fromarray(frame_arr))
    return filled


# =============================================================================
# Mask 预处理：GrowMask + BlockifyMask
# =============================================================================

def grow_masks(
    masks: list[Image.Image],
    pixels: int = 10,
) -> list[Image.Image]:
    """Mask 膨胀：向外扩展 mask 边缘指定像素数。

    使用 PIL MaxFilter 实现形态学膨胀，覆盖分割边界处的残留像素。

    Args:
        masks:  mask 列表（PIL RGB，白色=目标区域）。
        pixels: 扩展像素数（默认 10）。

    Returns:
        膨胀后的 mask 列表（PIL RGB，二值化）。
    """
    from PIL import ImageFilter

    grown = []
    kernel_size = 2 * pixels + 1
    for m in masks:
        m_l = m.convert("L").point(lambda p: 255 if p > 127 else 0)
        if pixels > 0:
            m_l = m_l.filter(ImageFilter.MaxFilter(kernel_size))
        grown.append(m_l.convert("RGB"))
    return grown


def blockify_masks(
    masks: list[Image.Image],
    block_size: int = 32,
) -> list[Image.Image]:
    """将 mask 对齐到 block_size × block_size 网格。

    若某个网格块中有任何白色像素，则整个块被设为白色。
    用于匹配 VAE 编码器的空间下采样率，避免亚像素 mask 边界在潜空间中产生伪影。

    Args:
        masks:      mask 列表（PIL RGB）。
        block_size: 网格块大小（默认 32，匹配常见 VAE 下采样率）。

    Returns:
        网格对齐后的 mask 列表（PIL RGB，二值化）。
    """
    blockified = []
    for m in masks:
        arr = np.array(m.convert("L"))
        h, w = arr.shape
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                ye = min(y + block_size, h)
                xe = min(x + block_size, w)
                if arr[y:ye, x:xe].max() > 127:
                    arr[y:ye, x:xe] = 255
        arr = np.where(arr > 127, 255, 0).astype(np.uint8)
        blockified.append(Image.fromarray(arr, mode="L").convert("RGB"))
    return blockified


# =============================================================================
# Pipeline 辅助
# =============================================================================

def filter_supported_kwargs(func, kwargs: dict) -> dict:
    """过滤掉 *func* 不接受的关键字参数，避免 unexpected keyword argument 报错。"""
    sig = inspect.signature(func)
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}


def process_pipeline_output(video_data) -> list[Image.Image]:
    """将 pipeline 输出统一转换为 ``List[PIL.Image]``。"""
    if hasattr(video_data, "to_numpy_images"):
        return [_frame_to_pil(f) for f in video_data.to_numpy_images()]
    if isinstance(video_data, list):
        return [_frame_to_pil(f) for f in video_data]
    raise TypeError(f"不支持的 pipeline 输出类型: {type(video_data)}")


def _frame_to_pil(frame) -> Image.Image:
    """将单帧（ndarray / PIL / Tensor）转换为 PIL Image。"""
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    if isinstance(frame, np.ndarray):
        arr = frame
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        return Image.fromarray(arr)
    if torch.is_tensor(frame):
        t = frame.detach().cpu()
        if t.ndim == 3 and t.shape[0] in (1, 3) and t.shape[-1] not in (1, 3):
            t = t.permute(1, 2, 0)
        arr = t.numpy()
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    raise TypeError(f"不支持的帧类型: {type(frame)}")


# =============================================================================
# 后处理：Mask 合成
# =============================================================================

def composite_with_mask(
    original_frames: list[Image.Image],
    generated_frames: list,
    masks: list[Image.Image],
) -> list[np.ndarray]:
    """使用 mask 将生成内容合成到原视频上。

    mask 白色（255）区域 = 使用生成帧；黑色（0）区域 = 保持原帧。

    Returns:
        ``List[np.ndarray]``，dtype ``uint8``，shape ``(H, W, 3)``。
    """
    assert len(original_frames) == len(generated_frames) == len(masks)
    composited = []
    for orig, gen, mask in zip(original_frames, generated_frames, masks):
        if isinstance(gen, np.ndarray):
            gen = Image.fromarray(gen)
        gen = gen.resize(orig.size, Image.BILINEAR)
        mask_l = mask.convert("L").resize(orig.size, Image.NEAREST)
        comp = Image.composite(gen, orig, mask_l)
        composited.append(np.array(comp, dtype=np.uint8))
    return composited


# =============================================================================
# 标准化输出
# =============================================================================

def save_experiment_outputs(
    output_dir: Path,
    name: str,
    composited_frames: list[np.ndarray],
    original_frames: list[Image.Image],
    masks: list[Image.Image],
    fps: int = 15,
    quality: int = 5,
    logger: logging.Logger = None,
):
    """保存标准化实验输出。

    在 *output_dir* 下生成三个文件：

    1. ``{name}.mp4``              —— 目标视频
    2. ``{name}_comparison.jpg``   —— [输入帧 | Mask帧 | 生成帧] 三联对比（取中间帧）
    3. ``{name}_showcase.jpg``     —— 首帧 + 末帧并排展示
    """
    _log = logger.info if logger else print
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 视频
    video_path = output_dir / f"{name}.mp4"
    save_video(composited_frames, str(video_path), fps=fps, quality=quality)
    _log(f"已保存视频: {video_path}")

    # 2. 中间帧对比图
    mid = len(composited_frames) // 2
    comp_img = _create_comparison_frame(
        original_frames[mid],
        masks[mid],
        Image.fromarray(composited_frames[mid]),
    )
    comp_path = output_dir / f"{name}_comparison.jpg"
    comp_img.save(comp_path, quality=95)
    _log(f"已保存对比帧: {comp_path}")

    # 3. 首末帧展示
    showcase = _create_showcase(
        Image.fromarray(composited_frames[0]),
        Image.fromarray(composited_frames[-1]),
    )
    showcase_path = output_dir / f"{name}_showcase.jpg"
    showcase.save(showcase_path, quality=95)
    _log(f"已保存首末帧展示: {showcase_path}")


def save_weight_grid(
    output_dir: Path,
    name: str,
    weight_frames: dict,
    width: int,
    height: int,
    logger: logging.Logger = None,
):
    """保存不同 reactive 权重的中间帧网格对比图。"""
    _log = logger.info if logger else print
    weights = sorted(weight_frames.keys())
    n = len(weights)
    grid = Image.new("RGB", (width * n, height))
    draw = ImageDraw.Draw(grid)
    font = _load_font(max(24, height // 20))

    for i, w in enumerate(weights):
        frame = Image.fromarray(weight_frames[w])
        grid.paste(frame, (i * width, 0))
        draw.text(
            (i * width + 10, 10),
            f"w={w:.1f}",
            fill="white",
            font=font,
            stroke_width=3,
            stroke_fill="black",
        )

    grid_path = output_dir / f"{name}_weight_grid.jpg"
    grid.save(grid_path, quality=95)
    _log(f"已保存权重对比网格: {grid_path}")


def _create_comparison_frame(
    input_frame: Image.Image,
    mask_frame: Image.Image,
    generated_frame: Image.Image,
) -> Image.Image:
    """创建 [输入帧 | Mask帧 | 生成帧] 三联对比图。"""
    w, h = input_frame.size
    canvas = Image.new("RGB", (w * 3, h))
    canvas.paste(input_frame, (0, 0))
    canvas.paste(mask_frame.convert("RGB"), (w, 0))
    canvas.paste(generated_frame, (w * 2, 0))

    draw = ImageDraw.Draw(canvas)
    font = _load_font(max(24, h // 20))
    for i, label in enumerate(["Input", "Mask", "Generated"]):
        draw.text(
            (i * w + 10, 10), label,
            fill="white", font=font, stroke_width=2, stroke_fill="black",
        )
    return canvas


def _create_showcase(
    first: Image.Image,
    last: Image.Image,
) -> Image.Image:
    """创建首帧 + 末帧并排展示图。"""
    w, h = first.size
    canvas = Image.new("RGB", (w * 2, h))
    canvas.paste(first, (0, 0))
    canvas.paste(last, (w, 0))

    draw = ImageDraw.Draw(canvas)
    font = _load_font(max(24, h // 20))
    draw.text(
        (10, 10), "First Frame",
        fill="white", font=font, stroke_width=2, stroke_fill="black",
    )
    draw.text(
        (w + 10, 10), "Last Frame",
        fill="white", font=font, stroke_width=2, stroke_fill="black",
    )
    return canvas


def _load_font(size: int = 30):
    """尝试加载 TrueType 字体，失败则回退到 PIL 默认字体。"""
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


# =============================================================================
# 通用 CLI 参数
# =============================================================================

def add_common_args(parser):
    """注册所有实验脚本共享的 CLI 参数。默认值与原始硬编码参数一致。"""
    parser.add_argument(
        "--sample_dir", type=str, required=True,
        help="样本数据目录（需包含 video_frames/、masks/、reference_images/ 子目录）",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="实验结果输出目录",
    )
    parser.add_argument("--width", type=int, default=480, help="视频宽度（默认 480）")
    parser.add_argument("--height", type=int, default=848, help="视频高度（默认 848）")
    parser.add_argument("--num_frames", type=int, default=49, help="帧数（默认 49）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    parser.add_argument("--fps", type=int, default=15, help="输出视频帧率（默认 15）")
    parser.add_argument(
        "--prompt", type=str,
        default="yellow rubber duck toy, product display, studio lighting",
        help="生成提示词",
    )
    parser.add_argument(
        "--negative_prompt", type=str,
        default="low quality, blurry, deformed",
        help="负面提示词",
    )
    parser.add_argument(
        "--reference_image", type=str, default="ref_rubber_duck.png",
        help="参考图像文件名（位于 sample_dir/reference_images/ 下）",
    )
    parser.add_argument(
        "--model_size", type=str, default="1.3B", choices=["1.3B", "14B"],
        help="模型大小（默认 1.3B）",
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=50,
        help="扩散采样步数（默认 50）",
    )
    parser.add_argument(
        "--cfg_scale", type=float, default=7.5,
        help="Classifier-free guidance 强度（默认 7.5）",
    )
    return parser
