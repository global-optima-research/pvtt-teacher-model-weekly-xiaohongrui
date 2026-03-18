#!/usr/bin/env python3
"""
PVTT Evaluation Script: Comprehensive metrics for FFGo-style first-frame-guided
video generation on the PVTT dataset.

Computes three groups of metrics:
  1. FiVE-Bench Metrics  - frame-level quality vs source video
  2. Edit Success Metrics - how well the edit worked
  3. VBench Metrics       - video quality (simplified implementations)

Usage:
    python evaluate_pvtt.py \
        --generated_dir /path/to/ffgo_original/pvtt/TIMESTAMP \
        --dataset_root  /path/to/pvtt_evaluation_datasets \
        --output_csv    results.csv \
        --skip_frames   4

Requirements:
    torch, torchvision, clip (openai), lpips, transformers, opencv-python,
    scikit-image, scipy, pandas, Pillow, tqdm
"""

import argparse
import csv
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pvtt_eval")


# ===========================================================================
# Model Loading (lazy singletons)
# ===========================================================================
_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_LPIPS_MODEL = None
_DINO_MODEL = None
_DINO_TRANSFORM = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_SIZE = (480, 832)  # (H, W) common evaluation resolution


def get_clip_model():
    """Load CLIP ViT-B/32 (OpenAI)."""
    global _CLIP_MODEL, _CLIP_PREPROCESS
    if _CLIP_MODEL is None:
        import clip
        _CLIP_MODEL, _CLIP_PREPROCESS = clip.load("ViT-B/32", device=DEVICE)
        _CLIP_MODEL.eval()
        logger.info("Loaded CLIP ViT-B/32")
    return _CLIP_MODEL, _CLIP_PREPROCESS


def get_lpips_model():
    """Load LPIPS with AlexNet backbone."""
    global _LPIPS_MODEL
    if _LPIPS_MODEL is None:
        import lpips
        _LPIPS_MODEL = lpips.LPIPS(net="alex").to(DEVICE)
        _LPIPS_MODEL.eval()
        logger.info("Loaded LPIPS (AlexNet)")
    return _LPIPS_MODEL


def get_dino_model():
    """Load DINOv2 ViT-B/14 from facebook."""
    global _DINO_MODEL, _DINO_TRANSFORM
    if _DINO_MODEL is None:
        from transformers import AutoModel, AutoImageProcessor
        _DINO_MODEL = AutoModel.from_pretrained(
            "facebook/dinov2-base"
        ).to(DEVICE).eval()
        _DINO_TRANSFORM = AutoImageProcessor.from_pretrained(
            "facebook/dinov2-base"
        )
        logger.info("Loaded DINOv2 ViT-B/14 (facebook/dinov2-base)")
    return _DINO_MODEL, _DINO_TRANSFORM


# ===========================================================================
# Utility: frame extraction
# ===========================================================================

def extract_frames_from_video(video_path: str, skip_first_n: int = 0) -> List[np.ndarray]:
    """Extract frames from an mp4 file. Returns list of RGB uint8 numpy arrays."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx >= skip_first_n:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    return frames


def resize_frame(frame: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize frame to (H, W)."""
    h, w = size
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_LANCZOS4)


def frames_to_common_size(
    gen_frames: List[np.ndarray],
    src_frames: List[np.ndarray],
    size: Tuple[int, int] = EVAL_SIZE,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Resize both frame lists to a common size and truncate to the shorter length."""
    n = min(len(gen_frames), len(src_frames))
    gen_resized = [resize_frame(f, size) for f in gen_frames[:n]]
    src_resized = [resize_frame(f, size) for f in src_frames[:n]]
    return gen_resized, src_resized


def np_to_tensor(frame: np.ndarray) -> torch.Tensor:
    """Convert HWC uint8 numpy to CHW float32 tensor in [0, 1]."""
    return torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0


def np_to_lpips_tensor(frame: np.ndarray) -> torch.Tensor:
    """Convert HWC uint8 numpy to CHW float32 tensor in [-1, 1] for LPIPS."""
    return torch.from_numpy(frame).permute(2, 0, 1).float() / 127.5 - 1.0


# ===========================================================================
# Group 1: FiVE-Bench Metrics
# ===========================================================================

def compute_psnr(gen_frames: List[np.ndarray], src_frames: List[np.ndarray]) -> float:
    """Average PSNR across corresponding frame pairs."""
    from skimage.metrics import peak_signal_noise_ratio
    vals = []
    for g, s in zip(gen_frames, src_frames):
        vals.append(peak_signal_noise_ratio(s, g, data_range=255))
    return float(np.mean(vals))


def compute_ssim(gen_frames: List[np.ndarray], src_frames: List[np.ndarray]) -> float:
    """Average SSIM across corresponding frame pairs."""
    from skimage.metrics import structural_similarity
    vals = []
    for g, s in zip(gen_frames, src_frames):
        val = structural_similarity(s, g, channel_axis=2, data_range=255)
        vals.append(val)
    return float(np.mean(vals))


def compute_lpips(gen_frames: List[np.ndarray], src_frames: List[np.ndarray]) -> float:
    """Average LPIPS (AlexNet) across corresponding frame pairs. Lower is better."""
    model = get_lpips_model()
    vals = []
    with torch.no_grad():
        for g, s in zip(gen_frames, src_frames):
            gt = np_to_lpips_tensor(g).unsqueeze(0).to(DEVICE)
            st = np_to_lpips_tensor(s).unsqueeze(0).to(DEVICE)
            d = model(gt, st)
            vals.append(d.item())
    return float(np.mean(vals))


def compute_clip_text_image(frames: List[np.ndarray], text: str) -> float:
    """Average CLIP cosine similarity between frames and a text prompt."""
    import clip
    model, preprocess = get_clip_model()
    text_tok = clip.tokenize([text], truncate=True).to(DEVICE)
    vals = []
    with torch.no_grad():
        text_feat = model.encode_text(text_tok)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        for frame in frames:
            img = Image.fromarray(frame)
            img_input = preprocess(img).unsqueeze(0).to(DEVICE)
            img_feat = model.encode_image(img_input)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            sim = (img_feat * text_feat).sum().item()
            vals.append(sim)
    return float(np.mean(vals))


def compute_struct_dist(gen_frames: List[np.ndarray], src_frames: List[np.ndarray]) -> float:
    """Structural distance: mean DINO-v2 feature cosine distance between frame pairs."""
    model, transform = get_dino_model()
    vals = []
    with torch.no_grad():
        for g, s in zip(gen_frames, src_frames):
            g_pil = Image.fromarray(g)
            s_pil = Image.fromarray(s)
            g_input = transform(g_pil, return_tensors="pt").to(DEVICE)
            s_input = transform(s_pil, return_tensors="pt").to(DEVICE)
            g_feat = model(**g_input).last_hidden_state[:, 0]  # CLS token
            s_feat = model(**s_input).last_hidden_state[:, 0]
            g_feat = g_feat / g_feat.norm(dim=-1, keepdim=True)
            s_feat = s_feat / s_feat.norm(dim=-1, keepdim=True)
            cos_dist = 1.0 - (g_feat * s_feat).sum().item()
            vals.append(cos_dist)
    return float(np.mean(vals))


def compute_mfs(gen_frames: List[np.ndarray]) -> float:
    """Mean Frame Similarity: avg CLIP image similarity between consecutive frames."""
    if len(gen_frames) < 2:
        return 1.0
    model, preprocess = get_clip_model()
    feats = []
    with torch.no_grad():
        for frame in gen_frames:
            img = Image.fromarray(frame)
            img_input = preprocess(img).unsqueeze(0).to(DEVICE)
            feat = model.encode_image(img_input)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            feats.append(feat)
    sims = []
    for i in range(len(feats) - 1):
        sim = (feats[i] * feats[i + 1]).sum().item()
        sims.append(sim)
    return float(np.mean(sims))


# ===========================================================================
# Group 2: Edit Success Metrics
# ===========================================================================

def compute_clip_direction(
    gen_frames: List[np.ndarray],
    src_frames: List[np.ndarray],
    target_prompt: str,
    source_prompt: str,
) -> float:
    """
    CLIP directional similarity: measures alignment between the image edit
    direction and the text edit direction in CLIP embedding space.

    clip_dir = mean_i[ cos( (I_gen_i - I_src_i) , (T_target - T_source) ) ]
    """
    import clip
    model, preprocess = get_clip_model()
    with torch.no_grad():
        src_tok = clip.tokenize([source_prompt], truncate=True).to(DEVICE)
        tgt_tok = clip.tokenize([target_prompt], truncate=True).to(DEVICE)
        src_text_feat = model.encode_text(src_tok)
        tgt_text_feat = model.encode_text(tgt_tok)
        text_dir = tgt_text_feat - src_text_feat
        text_dir = text_dir / (text_dir.norm(dim=-1, keepdim=True) + 1e-8)

        vals = []
        for g, s in zip(gen_frames, src_frames):
            g_pil = Image.fromarray(g)
            s_pil = Image.fromarray(s)
            g_input = preprocess(g_pil).unsqueeze(0).to(DEVICE)
            s_input = preprocess(s_pil).unsqueeze(0).to(DEVICE)
            g_feat = model.encode_image(g_input)
            s_feat = model.encode_image(s_input)
            img_dir = g_feat - s_feat
            img_dir = img_dir / (img_dir.norm(dim=-1, keepdim=True) + 1e-8)
            sim = (img_dir * text_dir).sum().item()
            vals.append(sim)
    return float(np.mean(vals))


def _get_product_mask_bbox(mask_dir: str, frame_idx: int, target_h: int, target_w: int) -> Optional[Tuple[int, int, int, int]]:
    """
    Load segmentation mask for a given frame index from the mask directory.
    Returns bounding box (y1, y2, x1, x2) of the mask region, scaled to target size.
    Returns None if mask not found or empty.
    """
    mask_path = Path(mask_dir)
    mask_files = sorted(list(mask_path.glob("*.png")) + list(mask_path.glob("*.jpg")))
    if frame_idx >= len(mask_files):
        # Use the last available mask (masks may be fewer than video frames)
        if not mask_files:
            return None
        frame_idx = min(frame_idx, len(mask_files) - 1)

    mask_img = Image.open(mask_files[frame_idx]).convert("L")
    orig_w, orig_h = mask_img.size
    mask_arr = np.array(mask_img)
    mask_bin = (mask_arr > 127).astype(np.uint8)

    if mask_bin.sum() == 0:
        return None

    ys, xs = np.where(mask_bin > 0)
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1

    # Scale bounding box to target resolution
    scale_y = target_h / orig_h
    scale_x = target_w / orig_w
    y1 = int(y1 * scale_y)
    y2 = int(y2 * scale_y)
    x1 = int(x1 * scale_x)
    x2 = int(x2 * scale_x)

    # Clamp
    y1 = max(0, y1)
    y2 = min(target_h, y2)
    x1 = max(0, x1)
    x2 = min(target_w, x2)

    if y2 <= y1 or x2 <= x1:
        return None
    return (y1, y2, x1, x2)


def compute_prod_clip(
    gen_frames: List[np.ndarray],
    product_image_path: str,
    mask_dir: str,
) -> Tuple[float, float]:
    """
    ProdCLIP: CLIP similarity between generated frames' product region and reference
    product image.
    ProdPersist: 1 - normalized_std of per-frame ProdCLIP scores.

    Returns (prod_clip_mean, prod_persist).
    """
    model, preprocess = get_clip_model()

    # Load product reference image
    prod_img = Image.open(product_image_path).convert("RGB")
    with torch.no_grad():
        prod_input = preprocess(prod_img).unsqueeze(0).to(DEVICE)
        prod_feat = model.encode_image(prod_input)
        prod_feat = prod_feat / prod_feat.norm(dim=-1, keepdim=True)

    h, w = gen_frames[0].shape[:2]
    per_frame_sims = []

    # Get bounding box from first mask frame (frame 0 of source video)
    bbox_default = _get_product_mask_bbox(mask_dir, 0, h, w)

    with torch.no_grad():
        for i, frame in enumerate(gen_frames):
            # Try to get mask for this frame; fall back to first-frame bbox
            bbox = _get_product_mask_bbox(mask_dir, i, h, w)
            if bbox is None:
                bbox = bbox_default
            if bbox is None:
                # No mask available: use full frame
                crop = Image.fromarray(frame)
            else:
                y1, y2, x1, x2 = bbox
                crop = Image.fromarray(frame[y1:y2, x1:x2])

            crop_input = preprocess(crop).unsqueeze(0).to(DEVICE)
            crop_feat = model.encode_image(crop_input)
            crop_feat = crop_feat / crop_feat.norm(dim=-1, keepdim=True)
            sim = (crop_feat * prod_feat).sum().item()
            per_frame_sims.append(sim)

    mean_sim = float(np.mean(per_frame_sims))

    # ProdPersist: 1 - normalized std (std / (mean + eps))
    if len(per_frame_sims) > 1:
        std = float(np.std(per_frame_sims))
        normalized_std = std / (abs(mean_sim) + 1e-8)
        prod_persist = 1.0 - min(normalized_std, 1.0)  # Clamp to [0, 1]
    else:
        prod_persist = 1.0

    return mean_sim, prod_persist


def compute_f0_edit_ssim(
    ffgo_ref_frame_path: str,
    gen_first_frame: np.ndarray,
) -> float:
    """
    F0EditSSIM: SSIM between the right half of the FFGo input first frame
    (background reference) and the generated first frame.
    """
    from skimage.metrics import structural_similarity

    ref_img = np.array(Image.open(ffgo_ref_frame_path).convert("RGB"))
    # Right half of the reference frame
    ref_w = ref_img.shape[1]
    ref_right = ref_img[:, ref_w // 2:, :]

    # Resize generated first frame to match ref right half size
    rh, rw = ref_right.shape[:2]
    gen_resized = cv2.resize(gen_first_frame, (rw, rh), interpolation=cv2.INTER_LANCZOS4)

    return float(structural_similarity(ref_right, gen_resized, channel_axis=2, data_range=255))


def compute_edit_fidelity(gen_frames: List[np.ndarray], src_frames: List[np.ndarray]) -> float:
    """
    EditFid: CLIP feature cosine similarity between generated and source frames.
    Higher = better preservation of overall scene fidelity.
    """
    model, preprocess = get_clip_model()
    vals = []
    with torch.no_grad():
        for g, s in zip(gen_frames, src_frames):
            g_input = preprocess(Image.fromarray(g)).unsqueeze(0).to(DEVICE)
            s_input = preprocess(Image.fromarray(s)).unsqueeze(0).to(DEVICE)
            g_feat = model.encode_image(g_input)
            s_feat = model.encode_image(s_input)
            g_feat = g_feat / g_feat.norm(dim=-1, keepdim=True)
            s_feat = s_feat / s_feat.norm(dim=-1, keepdim=True)
            sim = (g_feat * s_feat).sum().item()
            vals.append(sim)
    return float(np.mean(vals))


def compute_edit_persist(gen_frames: List[np.ndarray]) -> float:
    """
    EditPersist: average SSIM between consecutive generated frames.
    Measures temporal consistency of the edited content.
    """
    from skimage.metrics import structural_similarity
    if len(gen_frames) < 2:
        return 1.0
    vals = []
    for i in range(len(gen_frames) - 1):
        val = structural_similarity(
            gen_frames[i], gen_frames[i + 1], channel_axis=2, data_range=255
        )
        vals.append(val)
    return float(np.mean(vals))


# ===========================================================================
# Group 3: VBench Metrics (simplified)
# ===========================================================================

def compute_subject_consistency(gen_frames: List[np.ndarray]) -> float:
    """
    SubjectCons: DINO feature similarity of the main subject across frames.
    Computed as mean pairwise CLS-token cosine similarity across sampled frames.
    """
    model, transform = get_dino_model()
    # Sample up to 16 evenly-spaced frames to avoid excessive computation
    indices = np.linspace(0, len(gen_frames) - 1, min(16, len(gen_frames)), dtype=int)
    feats = []
    with torch.no_grad():
        for idx in indices:
            pil_img = Image.fromarray(gen_frames[idx])
            inputs = transform(pil_img, return_tensors="pt").to(DEVICE)
            feat = model(**inputs).last_hidden_state[:, 0]
            feat = feat / feat.norm(dim=-1, keepdim=True)
            feats.append(feat)
    # Mean pairwise cosine similarity
    if len(feats) < 2:
        return 1.0
    sims = []
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            sims.append((feats[i] * feats[j]).sum().item())
    return float(np.mean(sims))


def compute_bg_consistency(gen_frames: List[np.ndarray]) -> float:
    """
    BgCons: CLIP feature similarity of background regions across frames.
    Uses full-frame CLIP features as a proxy (background dominates most frames).
    Computed as mean consecutive-pair cosine similarity.
    """
    model, preprocess = get_clip_model()
    indices = np.linspace(0, len(gen_frames) - 1, min(16, len(gen_frames)), dtype=int)
    feats = []
    with torch.no_grad():
        for idx in indices:
            img_input = preprocess(Image.fromarray(gen_frames[idx])).unsqueeze(0).to(DEVICE)
            feat = model.encode_image(img_input)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            feats.append(feat)
    if len(feats) < 2:
        return 1.0
    sims = []
    for i in range(len(feats) - 1):
        sims.append((feats[i] * feats[i + 1]).sum().item())
    return float(np.mean(sims))


def compute_temporal_flickering(gen_frames: List[np.ndarray]) -> float:
    """
    TempFlk: 1 - mean absolute pixel difference between consecutive frames.
    Higher = less flickering. Normalized to [0, 1].
    """
    if len(gen_frames) < 2:
        return 1.0
    diffs = []
    for i in range(len(gen_frames) - 1):
        diff = np.abs(
            gen_frames[i].astype(np.float32) - gen_frames[i + 1].astype(np.float32)
        ).mean() / 255.0
        diffs.append(diff)
    return float(1.0 - np.mean(diffs))


def compute_dynamic_degree(frames: List[np.ndarray]) -> float:
    """
    DynDeg: measures motion magnitude as mean optical-flow magnitude.
    Uses Farneback dense optical flow on grayscale consecutive pairs.
    """
    if len(frames) < 2:
        return 0.0
    magnitudes = []
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    for i in range(1, len(frames)):
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean()
        magnitudes.append(mag)
        prev_gray = curr_gray
    return float(np.mean(magnitudes))


# ===========================================================================
# Main evaluation per task
# ===========================================================================

def evaluate_single_task(
    gen_video_path: str,
    src_video_path: str,
    product_image_path: str,
    mask_dir: str,
    ffgo_ref_frame_path: str,
    target_prompt: str,
    source_prompt: str,
    skip_frames: int = 4,
    eval_size: Tuple[int, int] = EVAL_SIZE,
) -> Dict[str, float]:
    """Compute all metrics for a single task. Returns dict of metric_name -> value."""

    # Extract frames
    gen_frames_raw = extract_frames_from_video(gen_video_path, skip_first_n=skip_frames)
    src_frames_raw = extract_frames_from_video(src_video_path, skip_first_n=0)

    if len(gen_frames_raw) == 0:
        logger.warning(f"No generated frames after skipping {skip_frames}: {gen_video_path}")
        return {}
    if len(src_frames_raw) == 0:
        logger.warning(f"No source frames: {src_video_path}")
        return {}

    # Resize to common evaluation size
    gen_frames, src_frames = frames_to_common_size(gen_frames_raw, src_frames_raw, eval_size)

    results = {}

    # --- Group 1: FiVE-Bench ---
    logger.debug("  Computing PSNR...")
    results["PSNR"] = compute_psnr(gen_frames, src_frames)

    logger.debug("  Computing SSIM...")
    results["SSIM"] = compute_ssim(gen_frames, src_frames)

    logger.debug("  Computing LPIPS...")
    results["LPIPS"] = compute_lpips(gen_frames, src_frames)

    logger.debug("  Computing CLIP_tgt...")
    results["CLIP_tgt"] = compute_clip_text_image(gen_frames, target_prompt)

    logger.debug("  Computing StructDist...")
    results["StructDist"] = compute_struct_dist(gen_frames, src_frames)

    logger.debug("  Computing MFS...")
    results["MFS"] = compute_mfs(gen_frames)

    # --- Group 2: Edit Success ---
    logger.debug("  Computing CLIP_dir...")
    results["CLIP_dir"] = compute_clip_direction(
        gen_frames, src_frames, target_prompt, source_prompt
    )

    if os.path.exists(product_image_path) and os.path.isdir(mask_dir):
        logger.debug("  Computing ProdCLIP / ProdPersist...")
        prod_clip, prod_persist = compute_prod_clip(gen_frames, product_image_path, mask_dir)
        results["ProdCLIP"] = prod_clip
        results["ProdPersist"] = prod_persist
    else:
        logger.warning(f"  Skipping ProdCLIP (product_img={product_image_path}, mask_dir={mask_dir})")
        results["ProdCLIP"] = float("nan")
        results["ProdPersist"] = float("nan")

    if os.path.exists(ffgo_ref_frame_path):
        logger.debug("  Computing F0EditSSIM...")
        results["F0EditSSIM"] = compute_f0_edit_ssim(ffgo_ref_frame_path, gen_frames[0])
    else:
        logger.warning(f"  Skipping F0EditSSIM (ref not found: {ffgo_ref_frame_path})")
        results["F0EditSSIM"] = float("nan")

    logger.debug("  Computing EditFid...")
    results["EditFid"] = compute_edit_fidelity(gen_frames, src_frames)

    logger.debug("  Computing EditPersist...")
    results["EditPersist"] = compute_edit_persist(gen_frames)

    # --- Group 3: VBench ---
    logger.debug("  Computing SubjectCons...")
    results["SubjectCons"] = compute_subject_consistency(gen_frames)

    logger.debug("  Computing BgCons...")
    results["BgCons"] = compute_bg_consistency(gen_frames)

    logger.debug("  Computing TempFlk...")
    results["TempFlk"] = compute_temporal_flickering(gen_frames)

    logger.debug("  Computing DynDeg...")
    gen_dyn = compute_dynamic_degree(gen_frames)
    src_dyn = compute_dynamic_degree(src_frames)
    results["DynDeg_gen"] = gen_dyn
    results["DynDeg_src"] = src_dyn
    results["DynDeg_delta"] = gen_dyn - src_dyn

    return results


# ===========================================================================
# Category extraction from task ID
# ===========================================================================

def get_category(task_id: str) -> str:
    """Extract product category from task ID like '0016-bracelet1_to_bracelet3'."""
    # Remove leading digits and dash
    parts = task_id.split("-", 1)
    if len(parts) > 1:
        name_part = parts[1]
    else:
        name_part = task_id
    # Extract category: everything before the first digit in the remainder
    cat = ""
    for ch in name_part:
        if ch.isdigit():
            break
        cat += ch
    return cat.rstrip("_") if cat else "unknown"


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PVTT Evaluation: compute all metrics for FFGo video generation results."
    )
    parser.add_argument(
        "--generated_dir", type=str, required=True,
        help="Root directory of generated results (contains per-task subdirs with ffgo_original.mp4)."
    )
    parser.add_argument(
        "--dataset_root", type=str, required=True,
        help="PVTT dataset root (contains videos/, product_images/, masks/, edit_prompt/)."
    )
    parser.add_argument(
        "--json_path", type=str, default=None,
        help="Path to task JSON. Default: {dataset_root}/edit_prompt/easy_new.json"
    )
    parser.add_argument(
        "--output_csv", type=str, default=None,
        help="Output CSV path. Default: {generated_dir}/evaluation_results.csv"
    )
    parser.add_argument(
        "--output_summary", type=str, default=None,
        help="Output summary JSON path. Default: {generated_dir}/evaluation_summary.json"
    )
    parser.add_argument(
        "--video_filename", type=str, default="ffgo_original.mp4",
        help="Filename of the generated video inside each task subdir."
    )
    parser.add_argument(
        "--ref_frame_filename", type=str, default="ffgo_original_ref_frame.jpg",
        help="Filename of the FFGo input reference frame."
    )
    parser.add_argument(
        "--skip_frames", type=int, default=4,
        help="Number of transition frames to skip from the beginning of the generated video."
    )
    parser.add_argument(
        "--eval_h", type=int, default=480,
        help="Evaluation resolution height."
    )
    parser.add_argument(
        "--eval_w", type=int, default=832,
        help="Evaluation resolution width."
    )
    parser.add_argument(
        "--task_ids", type=str, default=None,
        help="Comma-separated list of task IDs to evaluate. Default: all tasks found."
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug-level logging."
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    dataset_root = Path(args.dataset_root)
    generated_dir = Path(args.generated_dir)
    json_path = Path(args.json_path) if args.json_path else dataset_root / "edit_prompt" / "easy_new.json"
    eval_size = (args.eval_h, args.eval_w)

    output_csv = args.output_csv or str(generated_dir / "evaluation_results.csv")
    output_summary = args.output_summary or str(generated_dir / "evaluation_summary.json")

    # Load task JSON
    logger.info(f"Loading tasks from: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        all_entries = json.load(f)
    logger.info(f"Total tasks in JSON: {len(all_entries)}")

    # Filter by task_ids if specified
    if args.task_ids:
        filter_ids = set(args.task_ids.split(","))
        all_entries = [e for e in all_entries if e["id"] in filter_ids]
        logger.info(f"Filtered to {len(all_entries)} tasks by --task_ids")

    # Filter to tasks that have generated results
    entries = []
    for entry in all_entries:
        save_dir = entry["save_dir"]
        video_path = generated_dir / save_dir / args.video_filename
        if video_path.exists():
            entries.append(entry)
        else:
            logger.debug(f"Skipping {entry['id']}: generated video not found at {video_path}")
    logger.info(f"Tasks with generated videos: {len(entries)} / {len(all_entries)}")

    if not entries:
        logger.error("No tasks to evaluate. Check --generated_dir and --video_filename.")
        sys.exit(1)

    # Metric column order
    metric_cols = [
        # Group 1: FiVE-Bench
        "PSNR", "SSIM", "LPIPS", "CLIP_tgt", "StructDist", "MFS",
        # Group 2: Edit Success
        "CLIP_dir", "ProdCLIP", "ProdPersist", "F0EditSSIM", "EditFid", "EditPersist",
        # Group 3: VBench
        "SubjectCons", "BgCons", "TempFlk", "DynDeg_gen", "DynDeg_src", "DynDeg_delta",
    ]

    all_results = []

    for idx, entry in enumerate(tqdm(entries, desc="Evaluating")):
        task_id = entry["id"]
        video_name = entry["video_name"]
        save_dir = entry["save_dir"]
        target_prompt = entry["target_prompt"]
        source_prompt = entry.get("source_prompt", "")

        gen_video_path = str(generated_dir / save_dir / args.video_filename)
        src_video_path = str(dataset_root / "videos" / f"{video_name}.mp4")
        mask_dir = str(dataset_root / "masks" / video_name)
        ffgo_ref_path = str(generated_dir / save_dir / args.ref_frame_filename)

        # Resolve product image path
        ref_image_id = entry.get("inference_image_id", "")
        ref_stem = Path(ref_image_id).stem
        product_rgba_path = str(dataset_root / "product_images" / "output_dino_rgba" / f"{ref_stem}.png")
        if not os.path.exists(product_rgba_path):
            # Fallback to original jpg
            product_rgba_path = str(dataset_root / "product_images" / ref_image_id)

        logger.info(f"[{idx+1}/{len(entries)}] Evaluating: {task_id}")

        if not os.path.exists(src_video_path):
            logger.warning(f"  Source video not found: {src_video_path}, skipping.")
            continue

        try:
            metrics = evaluate_single_task(
                gen_video_path=gen_video_path,
                src_video_path=src_video_path,
                product_image_path=product_rgba_path,
                mask_dir=mask_dir,
                ffgo_ref_frame_path=ffgo_ref_path,
                target_prompt=target_prompt,
                source_prompt=source_prompt,
                skip_frames=args.skip_frames,
                eval_size=eval_size,
            )
            if not metrics:
                continue

            row = {"task_id": task_id, "video_name": video_name, "category": get_category(task_id)}
            row.update(metrics)
            all_results.append(row)

            # Log key metrics
            logger.info(
                f"  PSNR={metrics.get('PSNR', 0):.2f}  SSIM={metrics.get('SSIM', 0):.4f}  "
                f"LPIPS={metrics.get('LPIPS', 0):.4f}  CLIP_tgt={metrics.get('CLIP_tgt', 0):.4f}  "
                f"CLIP_dir={metrics.get('CLIP_dir', 0):.4f}"
            )
        except Exception as e:
            logger.error(f"  Failed: {e}", exc_info=True)
            continue

    if not all_results:
        logger.error("No results computed.")
        sys.exit(1)

    # --- Write per-task CSV ---
    df = pd.DataFrame(all_results)
    col_order = ["task_id", "video_name", "category"] + [c for c in metric_cols if c in df.columns]
    df = df[col_order]
    df.to_csv(output_csv, index=False, float_format="%.6f")
    logger.info(f"Per-task results saved: {output_csv}")

    # --- Aggregations ---
    numeric_cols = [c for c in metric_cols if c in df.columns]

    # Overall
    overall = df[numeric_cols].mean().to_dict()
    overall_std = df[numeric_cols].std().to_dict()

    # Per-category
    category_agg = {}
    for cat, group in df.groupby("category"):
        cat_mean = group[numeric_cols].mean().to_dict()
        cat_std = group[numeric_cols].std().to_dict()
        cat_count = len(group)
        category_agg[cat] = {"count": cat_count, "mean": cat_mean, "std": cat_std}

    summary = {
        "num_tasks": len(all_results),
        "skip_frames": args.skip_frames,
        "eval_size": list(eval_size),
        "overall_mean": overall,
        "overall_std": overall_std,
        "per_category": category_agg,
    }

    with open(output_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Summary saved: {output_summary}")

    # --- Print summary table ---
    print("\n" + "=" * 80)
    print("PVTT EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Tasks evaluated: {len(all_results)}")
    print(f"Skip frames: {args.skip_frames}")
    print(f"Eval resolution: {eval_size[1]}x{eval_size[0]}")
    print()

    # Direction indicators
    direction = {
        "PSNR": "^", "SSIM": "^", "LPIPS": "v", "CLIP_tgt": "^",
        "StructDist": "v", "MFS": "^", "CLIP_dir": "^", "ProdCLIP": "^",
        "ProdPersist": "^", "F0EditSSIM": "^", "EditFid": "^", "EditPersist": "^",
        "SubjectCons": "^", "BgCons": "^", "TempFlk": "^",
        "DynDeg_gen": "-", "DynDeg_src": "-", "DynDeg_delta": "-",
    }

    print("--- Overall Metrics ---")
    print(f"{'Metric':<16} {'Dir':>3} {'Mean':>10} {'Std':>10}")
    print("-" * 42)
    for col in numeric_cols:
        d = direction.get(col, "")
        m = overall.get(col, float("nan"))
        s = overall_std.get(col, float("nan"))
        print(f"{col:<16} {d:>3} {m:>10.4f} {s:>10.4f}")

    print()
    print("--- Per-Category Means ---")
    cats = sorted(category_agg.keys())
    # Print header
    header = f"{'Metric':<16}"
    for cat in cats:
        header += f" {cat:>12}"
    print(header)
    print("-" * (16 + 13 * len(cats)))
    for col in numeric_cols:
        row_str = f"{col:<16}"
        for cat in cats:
            v = category_agg[cat]["mean"].get(col, float("nan"))
            row_str += f" {v:>12.4f}"
        print(row_str)

    print()
    print(f"Per-task CSV:  {output_csv}")
    print(f"Summary JSON:  {output_summary}")
    print("=" * 80)


if __name__ == "__main__":
    main()
