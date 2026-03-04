"""
Bbox 序列提取工具

从视频 mask 序列中提取 bounding box 轨迹。
如果 bbox mask 实验成功，这个工具可以：
1. 自动提取物体的 bbox 轨迹
2. 用于后续的 VideoAnyDoor 或其他 bbox-based 方法
3. 可视化 bbox 轨迹

用法：
    python scripts/extract_bbox_sequence.py \
        --mask_dir samples/teapot/masks \
        --output_dir experiments/results/bbox_tracking \
        --visualize
"""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import json


def get_bbox_from_mask(mask_image, margin=0):
    """从 mask 提取 bounding box

    Args:
        mask_image: PIL Image
        margin: 扩展边距（像素）

    Returns:
        (x_min, y_min, x_max, y_max) or None
    """
    mask_array = np.array(mask_image.convert('L'))
    coords = np.argwhere(mask_array > 127)

    if len(coords) == 0:
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # 应用 margin
    width, height = mask_image.size
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(width, x_max + margin)
    y_max = min(height, y_max + margin)

    return (int(x_min), int(y_min), int(x_max), int(y_max))


def smooth_bbox_sequence(bboxes, window_size=5):
    """平滑 bbox 序列（移动平均）

    Args:
        bboxes: List of (x_min, y_min, x_max, y_max) or None
        window_size: 平滑窗口大小

    Returns:
        List of smoothed bboxes
    """
    # 填充 None 值（使用前后帧的平均）
    filled_bboxes = []
    for i, bbox in enumerate(bboxes):
        if bbox is None:
            # 找前后最近的非 None bbox
            prev_bbox = None
            next_bbox = None

            for j in range(i-1, -1, -1):
                if bboxes[j] is not None:
                    prev_bbox = bboxes[j]
                    break

            for j in range(i+1, len(bboxes)):
                if bboxes[j] is not None:
                    next_bbox = bboxes[j]
                    break

            if prev_bbox and next_bbox:
                # 线性插值
                alpha = 0.5
                bbox = tuple(int(prev_bbox[k] * alpha + next_bbox[k] * (1-alpha))
                           for k in range(4))
            elif prev_bbox:
                bbox = prev_bbox
            elif next_bbox:
                bbox = next_bbox
            else:
                bbox = (0, 0, 0, 0)  # 没有任何 bbox

        filled_bboxes.append(bbox)

    # 移动平均平滑
    smoothed = []
    for i in range(len(filled_bboxes)):
        start = max(0, i - window_size // 2)
        end = min(len(filled_bboxes), i + window_size // 2 + 1)

        window = filled_bboxes[start:end]
        avg_bbox = tuple(
            int(sum(b[k] for b in window) / len(window))
            for k in range(4)
        )
        smoothed.append(avg_bbox)

    return smoothed


def visualize_bbox_on_frame(frame, bbox, color='red', width=3):
    """在帧上绘制 bbox

    Args:
        frame: PIL Image
        bbox: (x_min, y_min, x_max, y_max)
        color: bbox 颜色
        width: 线宽

    Returns:
        PIL Image with bbox drawn
    """
    frame_copy = frame.copy()
    draw = ImageDraw.Draw(frame_copy)

    x_min, y_min, x_max, y_max = bbox
    draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=width)

    return frame_copy


def main():
    parser = argparse.ArgumentParser(description='Extract bbox sequence from masks')
    parser.add_argument('--mask_dir', type=str, required=True,
                       help='Directory containing mask images')
    parser.add_argument('--video_dir', type=str, default=None,
                       help='Directory containing video frames (for visualization)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for bbox sequence')
    parser.add_argument('--margin', type=int, default=20,
                       help='Bbox margin expansion (pixels)')
    parser.add_argument('--smooth', action='store_true',
                       help='Apply temporal smoothing to bbox sequence')
    parser.add_argument('--smooth_window', type=int, default=5,
                       help='Smoothing window size')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization video with bbox overlay')

    args = parser.parse_args()

    # ============================================
    # 加载 masks
    # ============================================
    mask_dir = Path(args.mask_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_files = sorted(mask_dir.glob('*.png'))
    print(f"Found {len(mask_files)} masks in {mask_dir}")

    # ============================================
    # 提取 bbox 序列
    # ============================================
    print(f"Extracting bbox sequence (margin={args.margin}px)...")

    bboxes = []
    for mask_file in mask_files:
        mask = Image.open(mask_file)
        bbox = get_bbox_from_mask(mask, margin=args.margin)
        bboxes.append(bbox)

    # 统计
    valid_count = sum(1 for b in bboxes if b is not None)
    print(f"  Valid bboxes: {valid_count}/{len(bboxes)}")

    # ============================================
    # 平滑（可选）
    # ============================================
    if args.smooth:
        print(f"Smoothing bbox sequence (window={args.smooth_window})...")
        bboxes = smooth_bbox_sequence(bboxes, window_size=args.smooth_window)

    # ============================================
    # 保存 bbox 序列
    # ============================================
    bbox_json = {
        'num_frames': len(bboxes),
        'margin': args.margin,
        'smoothed': args.smooth,
        'bboxes': [
            {'frame': i, 'bbox': bbox} if bbox else {'frame': i, 'bbox': None}
            for i, bbox in enumerate(bboxes)
        ]
    }

    json_path = output_dir / 'bbox_sequence.json'
    with open(json_path, 'w') as f:
        json.dump(bbox_json, f, indent=2)

    print(f"Saved bbox sequence: {json_path}")

    # 保存为 numpy 数组（方便加载）
    bbox_array = np.array([b if b else (0, 0, 0, 0) for b in bboxes])
    npy_path = output_dir / 'bbox_sequence.npy'
    np.save(npy_path, bbox_array)
    print(f"Saved numpy array: {npy_path}")

    # ============================================
    # 可视化（可选）
    # ============================================
    if args.visualize:
        if args.video_dir is None:
            print("Warning: --video_dir not specified, skipping visualization")
        else:
            print("Generating visualization...")
            video_dir = Path(args.video_dir)
            frame_files = sorted(video_dir.glob('*.jpg'))[:len(bboxes)]

            vis_frames = []
            for frame_file, bbox in zip(frame_files, bboxes):
                frame = Image.open(frame_file).convert('RGB')

                if bbox:
                    frame_vis = visualize_bbox_on_frame(frame, bbox)
                else:
                    frame_vis = frame

                vis_frames.append(frame_vis)

            # 保存可视化视频
            from diffsynth.utils.data import save_video, VideoData

            vis_video = VideoData(
                video_frames=[np.array(f) for f in vis_frames],
                height=vis_frames[0].height,
                width=vis_frames[0].width,
                fps=15
            )

            vis_path = output_dir / 'bbox_visualization.mp4'
            save_video(vis_video, str(vis_path))
            print(f"Saved visualization: {vis_path}")

            # 保存第一帧
            vis_frames[0].save(output_dir / 'bbox_frame0.jpg')

    # ============================================
    # 统计信息
    # ============================================
    if valid_count > 0:
        bbox_widths = [b[2] - b[0] for b in bboxes if b]
        bbox_heights = [b[3] - b[1] for b in bboxes if b]

        print("\nBbox Statistics:")
        print(f"  Avg width:  {np.mean(bbox_widths):.1f}px")
        print(f"  Avg height: {np.mean(bbox_heights):.1f}px")
        print(f"  Width range:  {min(bbox_widths)}-{max(bbox_widths)}px")
        print(f"  Height range: {min(bbox_heights)}-{max(bbox_heights)}px")

    print("\nDone!")


if __name__ == "__main__":
    main()
