"""
PVTT 数据集掩码提取：GroundingDINO + SAM2

功能：
1. 从 easy_new.json 读取所有视频条目
2. 从 mp4 中提取视频帧（保存到 video_frames/{video_name}/）
3. 使用 GroundingDINO 在首帧中检测 source_object → bbox
4. 使用 SAM2 Video Predictor 将 bbox 传播到所有帧 → 二值 mask
5. 保存 mask 到 masks/{video_name}/

GroundingDINO 后端支持两种方式（自动检测）：
  1. transformers 库（推荐，无需编译 CUDA）：pip install transformers
  2. groundingdino-py 源码：pip install groundingdino-py 或从源码 pip install -e .

前置依赖：
    pip install sam2 opencv-python-headless
    pip install transformers   # 方式 1（推荐）
    # 或 pip install groundingdino-py  # 方式 2

用法：
    # 使用 transformers 后端（推荐，无需 --gdino_config）
    python extract_masks.py \
        --dataset_root ../../samples/pvtt_evaluation_datasets \
        --sam2_checkpoint /path/to/sam2.1_hiera_large.pt \
        --sam2_config configs/sam2.1/sam2.1_hiera_l.yaml \
        --gdino_checkpoint /path/to/groundingdino_swint_ogc.pth

    # 使用源码后端（需要 --gdino_config）
    python extract_masks.py \
        --dataset_root ../../samples/pvtt_evaluation_datasets \
        --sam2_checkpoint /path/to/sam2.1_hiera_large.pt \
        --sam2_config configs/sam2.1/sam2.1_hiera_l.yaml \
        --gdino_config /path/to/GroundingDINO_SwinT_OGC.py \
        --gdino_checkpoint /path/to/groundingdino_swint_ogc.pth

    # 强制指定后端
    python extract_masks.py --gdino_backend transformers ...
    python extract_masks.py --gdino_backend source ...
"""

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------

def setup_logger(name, log_file=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# 视频帧提取
# ---------------------------------------------------------------------------

def extract_frames_from_mp4(mp4_path, output_dir, max_frames=None):
    """从 mp4 提取帧并保存为 PNG。返回实际提取的帧数。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {mp4_path}")

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and count >= max_frames:
            break
        fname = output_dir / f"{count:05d}.jpg"
        cv2.imwrite(str(fname), frame)
        count += 1
    cap.release()
    return count


# ---------------------------------------------------------------------------
# GroundingDINO 后端检测
# ---------------------------------------------------------------------------

def _detect_gdino_backend():
    """自动检测可用的 GroundingDINO 后端。优先 transformers。"""
    try:
        # 仅用于探测是否安装
        from transformers import AutoModelForZeroShotObjectDetection  # noqa: F401
        return "transformers"
    except ImportError:
        pass
    try:
        from groundingdino.util.inference import load_model  # noqa: F401
        return "source"
    except ImportError:
        pass
    raise ImportError(
        "GroundingDINO 未安装。请选择以下任一方式安装：\n"
        "  方式 1（推荐）: pip install transformers\n"
        "  方式 2（源码）: pip install groundingdino-py\n"
        "详见 scripts/README.md 中的安装指南。"
    )


# ---- transformers 后端 ----

def load_gdino_transformers(checkpoint_path, device="cuda"):
    """通过 transformers 库加载 GroundingDINO（无需编译 CUDA）。

    checkpoint_path: HuggingFace model ID（如 "IDEA-Research/grounding-dino-tiny"）
                     或本地目录路径。如果传入的是 .pth 权重文件路径，则自动使用
                     HuggingFace 上的模型 ID。
    """
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    import os

    # 如果传入的是 .pth 文件而非目录，使用 HuggingFace model ID
    model_id = checkpoint_path
    if checkpoint_path.endswith(".pth") or not os.path.isdir(checkpoint_path):
        model_id = "IDEA-Research/grounding-dino-tiny"

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    model = model.to(device)
    model.eval()
    return {"model": model, "processor": processor, "backend": "transformers"}


def _gdino_post_process_transformers(processor, outputs, input_ids, target_sizes,
                                    box_threshold, text_threshold):
    """
    兼容不同 transformers 版本的 post_process_grounded_object_detection 调用方式。

    transformers 在不同版本中该函数签名可能变化，例如：
    - 旧版：post_process_grounded_object_detection(outputs, input_ids, box_threshold=..., text_threshold=..., target_sizes=...)
    - 新版：可能不接受 box_threshold/text_threshold，而是 threshold，或参数顺序/名字变化

    这里根据签名动态组装 kwargs，尽量覆盖常见变化。
    """
    import inspect

    fn = processor.post_process_grounded_object_detection
    sig = inspect.signature(fn)
    params = sig.parameters

    # 先准备位置参数：有些版本要求 outputs/input_ids 必须作为位置参数
    pos_args = []
    kwargs = {}

    # outputs
    if "outputs" in params:
        kwargs["outputs"] = outputs
    else:
        pos_args.append(outputs)

    # input_ids（有些版本需要，有些不需要）
    if "input_ids" in params:
        kwargs["input_ids"] = input_ids
    else:
        # 旧版常见是第二个位置参数
        if len(pos_args) == 1:
            pos_args.append(input_ids)
        else:
            # 不需要就不传
            pass

    # target_sizes
    if "target_sizes" in params:
        kwargs["target_sizes"] = target_sizes
    else:
        # 个别实现用 target_size
        if "target_size" in params:
            kwargs["target_size"] = target_sizes

    # 阈值：优先 box_threshold/text_threshold；否则尝试 threshold
    if "box_threshold" in params:
        kwargs["box_threshold"] = float(box_threshold)
    if "text_threshold" in params:
        kwargs["text_threshold"] = float(text_threshold)

    if "threshold" in params and ("box_threshold" not in kwargs):
        # 使用 box_threshold 作为统一阈值（更贴近“框阈值”语义）
        kwargs["threshold"] = float(box_threshold)

    # 调用
    try:
        return fn(*pos_args, **kwargs)
    except TypeError:
        # 兜底：如果 input_ids 传错导致问题，尝试去掉 input_ids 再调用一次
        kwargs.pop("input_ids", None)
        if len(pos_args) >= 2:
            pos_args = pos_args[:1]
        return fn(*pos_args, **kwargs)


def detect_object_transformers(model_dict, image_pil, text_prompt,
                               box_threshold=0.3, text_threshold=0.25):
    """transformers 后端检测。"""
    model = model_dict["model"]
    processor = model_dict["processor"]
    device = next(model.parameters()).device

    # transformers GroundingDINO 通常需要以句号结尾的 text prompt
    if not text_prompt.strip().endswith("."):
        text_prompt = text_prompt.strip() + "."

    inputs = processor(images=image_pil, text=text_prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # target_sizes: (H, W)
    target_sizes = [image_pil.size[::-1]]

    # 兼容不同 transformers 版本
    post = _gdino_post_process_transformers(
        processor=processor,
        outputs=outputs,
        input_ids=inputs.get("input_ids", None),
        target_sizes=target_sizes,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    # 绝大多数版本返回 list[dict]，取第 0 个
    results0 = post[0] if isinstance(post, (list, tuple)) else post

    boxes = results0.get("boxes", None)      # Tensor (N,4) xyxy
    scores = results0.get("scores", None)    # Tensor (N,)
    labels = results0.get("labels", None)    # list[str] 或 Tensor

    if boxes is None or scores is None or len(boxes) == 0:
        return None, 0.0, ""

    best_idx = int(torch.argmax(scores).item())
    box = boxes[best_idx].detach().cpu().tolist()
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

    w, h = image_pil.size
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    # phrase/label 兼容：labels 可能是 list[str]，也可能是 tensor/int
    phrase = ""
    if isinstance(labels, (list, tuple)):
        phrase = str(labels[best_idx]) if best_idx < len(labels) else ""
    else:
        phrase = str(labels[best_idx].item()) if hasattr(labels, "__len__") else str(labels)

    return [x1, y1, x2, y2], float(scores[best_idx].item()), phrase


# ---- groundingdino 源码后端 ----

def load_gdino_source(config_path, checkpoint_path, device="cuda"):
    """通过 groundingdino 源码库加载模型。"""
    from groundingdino.util.inference import load_model
    model = load_model(config_path, checkpoint_path, device=device)
    model.eval()
    return {"model": model, "backend": "source"}


def detect_object_source(model_dict, image_pil, text_prompt,
                         box_threshold=0.3, text_threshold=0.25):
    """源码后端检测。"""
    from groundingdino.util.inference import predict
    import groundingdino.datasets.transforms as T

    model = model_dict["model"]
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_transformed, _ = transform(image_pil, None)
    boxes, logits, phrases = predict(
        model, image_transformed, text_prompt,
        box_threshold=box_threshold, text_threshold=text_threshold
    )

    if len(boxes) == 0:
        return None, 0.0, ""

    best_idx = int(logits.argmax().item())
    box_norm = boxes[best_idx]  # cx, cy, w, h (normalized)
    w, h = image_pil.size
    cx, cy, bw, bh = box_norm.tolist()
    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    phrase = phrases[best_idx] if isinstance(phrases, (list, tuple)) else str(phrases)
    return [x1, y1, x2, y2], float(logits[best_idx].item()), phrase


# ---- 统一接口 ----

def load_gdino_model(config_path, checkpoint_path, device="cuda", backend=None):
    """加载 GroundingDINO 模型（自动选择后端）。"""
    if backend is None:
        backend = _detect_gdino_backend()
    if backend == "transformers":
        return load_gdino_transformers(checkpoint_path, device=device)
    else:
        if not config_path:
            raise ValueError(
                "源码后端需要 --gdino_config 参数。"
                "如不想提供，请安装 transformers 库: pip install transformers"
            )
        return load_gdino_source(config_path, checkpoint_path, device=device)


def detect_object_gdino(model_dict, image_pil, text_prompt,
                        box_threshold=0.3, text_threshold=0.25):
    """使用 GroundingDINO 检测物体，返回最高置信度的 bbox [x1,y1,x2,y2] (像素坐标)。"""
    if model_dict["backend"] == "transformers":
        return detect_object_transformers(
            model_dict, image_pil, text_prompt,
            box_threshold=box_threshold, text_threshold=text_threshold
        )
    else:
        return detect_object_source(
            model_dict, image_pil, text_prompt,
            box_threshold=box_threshold, text_threshold=text_threshold
        )


# ---------------------------------------------------------------------------
# SAM2 视频分割
# ---------------------------------------------------------------------------

def run_sam2_video_segmentation(sam2_config, sam2_checkpoint, frame_dir,
                               box_xyxy, device="cuda"):
    """使用 SAM2 Video Predictor 从 bbox 传播 mask 到所有帧。

    返回: dict[frame_idx] -> np.ndarray (H, W) bool mask
    """
    from sam2.build_sam import build_sam2_video_predictor
    import os, sam2 as _sam2_pkg

    # Hydra 搜索路径是 pkg://sam2（即 sam2 包根目录），
    # 所以 config_file 必须是相对于该目录的路径，如 configs/sam2.1/sam2.1_hiera_l.yaml
    if os.path.isabs(sam2_config):
        sam2_pkg_dir = os.path.dirname(_sam2_pkg.__file__)
        sam2_config = os.path.relpath(sam2_config, sam2_pkg_dir)

    predictor = build_sam2_video_predictor(
        sam2_config, sam2_checkpoint, device=device
    )

    # autocast：cuda 才能用 bfloat16 autocast；cpu 就别开
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if str(device).startswith("cuda")
        else torch.no_grad()
    )

    with torch.inference_mode(), autocast_ctx:
        state = predictor.init_state(video_path=str(frame_dir))
        predictor.reset_state(state)

        # 在首帧添加 bbox prompt
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,
            obj_id=1,
            box=np.array(box_xyxy, dtype=np.float32),
        )

        # 前向传播
        masks = {}
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            # mask_logits: (num_obj, H, W) logits
            mask = (mask_logits[0] > 0.0).detach().cpu().numpy().squeeze()
            masks[int(frame_idx)] = mask.astype(bool)

    return masks


# ---------------------------------------------------------------------------
# 保存 mask
# ---------------------------------------------------------------------------

def save_masks(masks_dict, output_dir):
    """保存二值 mask 为 PNG（白色=目标区域，黑色=背景）。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx in sorted(masks_dict.keys()):
        mask = masks_dict[frame_idx]
        mask_uint8 = (mask.astype(np.uint8)) * 255
        Image.fromarray(mask_uint8, mode="L").save(
            output_dir / f"{frame_idx:05d}.png"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PVTT 数据集掩码提取：GroundingDINO + SAM2"
    )
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="pvtt_evaluation_datasets 根目录")
    parser.add_argument("--json_path", type=str, default=None,
                        help="easy_new.json 路径（默认: dataset_root/edit_prompt/easy_new.json）")
    parser.add_argument("--sam2_checkpoint", type=str, required=True,
                        help="SAM2 模型权重路径")
    parser.add_argument("--sam2_config", type=str, required=True,
                        help="SAM2 配置文件路径")
    parser.add_argument("--gdino_config", type=str, default=None,
                        help="GroundingDINO 配置文件路径（仅源码后端需要）")
    parser.add_argument("--gdino_checkpoint", type=str, required=True,
                        help="GroundingDINO 模型权重路径（或 HuggingFace model ID）")
    parser.add_argument("--gdino_backend", type=str, default=None,
                        choices=["transformers", "source"],
                        help="GroundingDINO 后端（默认自动检测）")
    parser.add_argument("--box_threshold", type=float, default=0.3,
                        help="GroundingDINO box 置信度阈值（默认 0.3）")
    parser.add_argument("--text_threshold", type=float, default=0.25,
                        help="GroundingDINO text 置信度阈值（默认 0.25）")
    parser.add_argument("--max_frames", type=int, default=81,
                        help="每个视频最多提取的帧数（默认 81）")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="跳过已存在 mask 的视频（默认 True）")
    parser.add_argument("--no_skip_existing", dest="skip_existing",
                        action="store_false")
    parser.add_argument("--video_names", type=str, default=None,
                        help="仅处理指定视频（逗号分隔），默认处理全部")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    json_path = Path(args.json_path) if args.json_path else \
        dataset_root / "edit_prompt" / "easy_new.json"
    video_dir = dataset_root / "videos"
    frames_root = dataset_root / "video_frames"
    masks_root = dataset_root / "masks"

    logger = setup_logger("extract_masks",
                          str(masks_root / "extract_masks.log"))

    # ------------------------------------------------------------------
    # 1. 解析 JSON，获取唯一的 (video_name, source_object) 映射
    # ------------------------------------------------------------------
    with open(json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    video_source_map = {}  # video_name -> source_object
    for entry in entries:
        vname = entry["video_name"]
        sobj = entry["source_object"]
        if vname not in video_source_map:
            video_source_map[vname] = sobj

    # 可选：仅处理指定视频
    if args.video_names:
        filter_set = set(x.strip() for x in args.video_names.split(",") if x.strip())
        video_source_map = {k: v for k, v in video_source_map.items()
                            if k in filter_set}

    logger.info(f"共 {len(video_source_map)} 个唯一视频需要处理")

    # ------------------------------------------------------------------
    # 2. 加载模型
    # ------------------------------------------------------------------
    logger.info("加载 GroundingDINO 模型...")
    gdino_model = load_gdino_model(
        args.gdino_config, args.gdino_checkpoint,
        device=args.device, backend=args.gdino_backend
    )
    logger.info(f"GroundingDINO 加载完成 (后端: {gdino_model['backend']})")

    # ------------------------------------------------------------------
    # 3. 逐视频处理
    # ------------------------------------------------------------------
    success_count = 0
    fail_count = 0
    skip_count = 0

    for vname, source_object in video_source_map.items():
        frame_out = frames_root / vname
        mask_out = masks_root / vname

        # 检查是否已存在
        if args.skip_existing and mask_out.exists():
            existing = list(mask_out.glob("*.png"))
            if len(existing) > 0:
                logger.info(f"[跳过] {vname}: 已存在 {len(existing)} 个 mask")
                skip_count += 1
                continue

        logger.info("")
        logger.info(f"{'=' * 60}")
        logger.info(f"处理视频: {vname}")
        logger.info(f"  源物体: {source_object}")
        logger.info(f"{'=' * 60}")

        mp4_path = video_dir / f"{vname}.mp4"
        if not mp4_path.exists():
            logger.error(f"  视频文件不存在: {mp4_path}")
            fail_count += 1
            continue

        try:
            # 3a. 提取视频帧
            if frame_out.exists() and len(list(frame_out.glob("*.jpg"))) > 0:
                n_frames = len(list(frame_out.glob("*.jpg")))
                logger.info(f"  帧已提取 ({n_frames} 帧)，跳过提取步骤")
            else:
                logger.info("  提取视频帧...")
                n_frames = extract_frames_from_mp4(
                    mp4_path, frame_out, max_frames=args.max_frames)
                logger.info(f"  已提取 {n_frames} 帧 → {frame_out}")

            # 如果已有帧数超过 max_frames，删除多余帧
            frame_paths = sorted(frame_out.glob("*.jpg"))
            if args.max_frames and len(frame_paths) > args.max_frames:
                for fp in frame_paths[args.max_frames:]:
                    fp.unlink()
                frame_paths = frame_paths[:args.max_frames]
                logger.info(f"  裁剪至前 {args.max_frames} 帧")
            if len(frame_paths) == 0:
                raise RuntimeError(f"未找到任何帧: {frame_out}")

            # 3b. GroundingDINO 检测
            first_frame_path = frame_paths[0]
            first_frame_pil = Image.open(first_frame_path).convert("RGB")
            logger.info(f"  GroundingDINO 检测: '{source_object}'")

            bbox, confidence, phrase = detect_object_gdino(
                gdino_model, first_frame_pil, source_object,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold
            )

            if bbox is None:
                logger.warning("  未检测到目标物体！尝试降低阈值...")
                bbox, confidence, phrase = detect_object_gdino(
                    gdino_model, first_frame_pil, source_object,
                    box_threshold=0.15, text_threshold=0.15
                )

            if bbox is None:
                logger.error("  仍未检测到目标物体，跳过此视频")
                fail_count += 1
                continue

            logger.info(f"  检测结果: bbox={bbox}, conf={confidence:.3f}, phrase='{phrase}'")

            # 保存检测结果可视化
            mask_out.parent.mkdir(parents=True, exist_ok=True)
            vis_img = first_frame_pil.copy()
            from PIL import ImageDraw
            draw = ImageDraw.Draw(vis_img)
            draw.rectangle(bbox, outline="red", width=3)
            draw.text((bbox[0], max(0, bbox[1] - 15)),
                      f"{phrase} ({confidence:.2f})", fill="red")
            vis_img.save(mask_out.parent / f"{vname}_detection.jpg")

            # 3c. SAM2 视频分割
            logger.info("  SAM2 视频分割...")
            masks = run_sam2_video_segmentation(
                args.sam2_config, args.sam2_checkpoint,
                frame_out, bbox, device=args.device
            )
            logger.info(f"  分割完成: {len(masks)} 帧 mask")

            # 3d. 保存 mask
            save_masks(masks, mask_out)
            logger.info(f"  mask 已保存 → {mask_out}")

            success_count += 1

        except Exception as e:
            logger.error(f"  处理失败: {e}", exc_info=True)
            fail_count += 1

    # ------------------------------------------------------------------
    # 4. 汇总
    # ------------------------------------------------------------------
    logger.info("")
    logger.info(f"{'=' * 60}")
    logger.info("掩码提取完成")
    logger.info(f"  成功: {success_count}")
    logger.info(f"  跳过: {skip_count}")
    logger.info(f"  失败: {fail_count}")
    logger.info(f"  帧目录: {frames_root}")
    logger.info(f"  掩码目录: {masks_root}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()