# """
# Grounded SAM 2 茶壶分割脚本

# 基于 experiments/logs/sam2-segmentation_2026-01-20.md 验证过的方案：
# - Grounding DINO 检测 bbox
# - SAM2 ImagePredictor 分割（效果比 VideoPredictor 的 box prompt 更好）

# 运行方式（需要在 GPU 服务器上运行）：
#     python segment_teapot.py

# 依赖：
#     pip install transformers
#     pip install sam2  # 或 segment-anything-2
# """

# import os
# import sys
# from pathlib import Path

# import numpy as np
# from PIL import Image
# import torch


# def load_grounding_dino():
#     """加载 Grounding DINO 模型"""
#     from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

#     print("Loading Grounding DINO...")
#     model_id = "IDEA-Research/grounding-dino-base"
#     processor = AutoProcessor.from_pretrained(model_id)
#     model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
#     return processor, model


# def load_sam2():
#     """加载 SAM2 模型"""
#     from sam2.build_sam import build_sam2_hf
#     from sam2.sam2_image_predictor import SAM2ImagePredictor

#     print("Loading SAM2 from HuggingFace...")
#     # 使用 sam2 库自带的 HuggingFace 加载方式
#     sam2_model = build_sam2_hf("facebook/sam2.1-hiera-large")
#     predictor = SAM2ImagePredictor(sam2_model)

#     return predictor, None, "local"


# def detect_with_grounding_dino(processor, model, image, text_prompt, device, box_threshold=0.3):
#     """使用 Grounding DINO 检测物体"""
#     inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)

#     with torch.no_grad():
#         outputs = model(**inputs)

#     results = processor.post_process_grounded_object_detection(
#         outputs,
#         inputs.input_ids,
#         box_threshold=box_threshold,
#         text_threshold=0.25,
#         target_sizes=[image.size[::-1]]  # (height, width)
#     )[0]

#     return results


# def segment_with_sam2(predictor, image, boxes, device):
#     """使用 SAM2 ImagePredictor 分割"""
#     predictor.set_image(np.array(image))

#     # 合并所有检测到的 box
#     if len(boxes) == 0:
#         return None

#     # 转换到 CPU 进行计算
#     boxes_cpu = boxes.cpu()

#     # 选择最大的 box（根据实验记录，最大框效果最好）
#     areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes_cpu]
#     largest_idx = np.argmax(areas)
#     largest_box = boxes_cpu[largest_idx]

#     print(f"Using largest box: {largest_box.tolist()}, area: {areas[largest_idx]:.0f}")

#     masks, scores, _ = predictor.predict(
#         point_coords=None,
#         point_labels=None,
#         box=largest_box.numpy(),
#         multimask_output=False
#     )

#     return masks[0]  # [H, W] boolean


# def segment_with_sam2_hf(processor, model, image, boxes, device):
#     """使用 HuggingFace 版本的 SAM2"""
#     if len(boxes) == 0:
#         return None

#     # 选择最大的 box
#     areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
#     largest_idx = np.argmax(areas)
#     largest_box = boxes[largest_idx].cpu().numpy().tolist()

#     print(f"Using largest box: {largest_box}, area: {areas[largest_idx]:.0f}")

#     inputs = processor(
#         images=image,
#         input_boxes=[[largest_box]],
#         return_tensors="pt"
#     ).to(device)

#     with torch.no_grad():
#         outputs = model(**inputs)

#     masks = processor.post_process_masks(
#         outputs.pred_masks,
#         inputs["original_sizes"],
#         inputs["reshaped_input_sizes"]
#     )

#     mask = masks[0][0, 0].cpu().numpy()
#     return mask > 0.5


# def main():
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--frames_dir", default="video_frames")
#     parser.add_argument("--output_dir", default="masks")
#     parser.add_argument("--text_prompt", default="teapot.")
#     parser.add_argument("--box_threshold", type=float, default=0.3)
#     parser.add_argument("--sample_interval", type=int, default=1,
#                         help="Process every N frames (1=all frames)")
#     args = parser.parse_args()

#     script_dir = Path(__file__).parent
#     frames_dir = script_dir / args.frames_dir
#     output_dir = script_dir / args.output_dir

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device}")

#     # 加载模型
#     gdino_processor, gdino_model = load_grounding_dino()
#     gdino_model = gdino_model.to(device)

#     sam2_result = load_sam2()
#     if sam2_result[2] == "hf":
#         sam2_processor, sam2_model, _ = sam2_result
#         sam2_model = sam2_model.to(device)
#         use_hf = True
#     else:
#         sam2_predictor, _, _ = sam2_result
#         sam2_predictor.model = sam2_predictor.model.to(device)
#         use_hf = False

#     # 获取所有帧
#     frame_files = sorted(frames_dir.glob("*.jpg"))
#     print(f"Found {len(frame_files)} frames")

#     output_dir.mkdir(parents=True, exist_ok=True)

#     # 处理每一帧
#     for i, frame_file in enumerate(frame_files):
#         if i % args.sample_interval != 0:
#             continue

#         image = Image.open(frame_file).convert("RGB")

#         # 1. Grounding DINO 检测
#         results = detect_with_grounding_dino(
#             gdino_processor, gdino_model, image,
#             args.text_prompt, device, args.box_threshold
#         )

#         boxes = results["boxes"]
#         scores = results["scores"]

#         if len(boxes) == 0:
#             print(f"Frame {i}: No detection, using previous mask or empty")
#             # 可以选择使用上一帧的 mask 或空 mask
#             mask = np.zeros((image.height, image.width), dtype=np.uint8)
#         else:
#             # 2. SAM2 分割
#             if use_hf:
#                 mask = segment_with_sam2_hf(sam2_processor, sam2_model, image, boxes, device)
#             else:
#                 mask = segment_with_sam2(sam2_predictor, image, boxes, device)

#             if mask is None:
#                 mask = np.zeros((image.height, image.width), dtype=np.uint8)
#             else:
#                 mask = mask.astype(np.uint8) * 255

#         # 保存 mask
#         mask_path = output_dir / f"{i:04d}.png"
#         Image.fromarray(mask).save(mask_path)

#         if (i + 1) % 50 == 0 or i == 0:
#             print(f"Processed frame {i + 1}/{len(frame_files)}, "
#                   f"detections: {len(boxes)}, "
#                   f"mask coverage: {mask.sum() / mask.size * 100:.1f}%")

#     print(f"\nMasks saved to {output_dir}")
#     print(f"Total: {len(list(output_dir.glob('*.png')))} masks")


# if __name__ == "__main__":
#     main()



"""
Grounded SAM 2 茶壶分割脚本

基于 experiments/logs/sam2-segmentation_2026-01-20.md 验证过的方案：
- Grounding DINO 检测 bbox
- SAM2 ImagePredictor 分割（效果比 VideoPredictor 的 box prompt 更好）

运行方式（需要在 GPU 服务器上运行）：
    python segment_teapot.py

依赖：
    pip install transformers
    pip install sam2  # 或 segment-anything-2
"""

import os
# 【新增】强制使用 HF 镜像，解决网络问题
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
from pathlib import Path
import argparse

import numpy as np
from PIL import Image
import torch


def load_grounding_dino():
    """加载 Grounding DINO 模型"""
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    print("Loading Grounding DINO...")
    model_id = "IDEA-Research/grounding-dino-base"
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    except Exception as e:
        print(f"Error loading Grounding DINO: {e}")
        print("Tip: Check your network connection or HF_ENDPOINT setting.")
        sys.exit(1)
    return processor, model


def load_sam2():
    """加载 SAM2 模型"""
    try:
        from sam2.build_sam import build_sam2_hf
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        print("Loading SAM2 from HuggingFace...")
        # 使用 sam2 库自带的 HuggingFace 加载方式
        # 注意：这里加载的是 facebook/sam2.1-hiera-large
        sam2_model = build_sam2_hf("facebook/sam2.1-hiera-large")
        predictor = SAM2ImagePredictor(sam2_model)
        
        # 返回结构: (predictor, model_obj, mode_string)
        return predictor, None, "local"
    except ImportError:
        print("Error: 'sam2' module not found. Please install via: pip install segment-anything-2")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading SAM2: {e}")
        sys.exit(1)


def detect_with_grounding_dino(processor, model, image, text_prompt, device, box_threshold=0.3):
    """使用 Grounding DINO 检测物体"""
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # 【关键修改】transformers 新版本中参数名为 threshold，旧版本为 box_threshold
    # 这里我们尝试捕获异常或者直接使用新参数名
    try:
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,     # 新版参数名
            text_threshold=0.25,
            target_sizes=[image.size[::-1]]  # (height, width)
        )[0]
    except TypeError:
        # 如果是旧版本，回退到 box_threshold
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold, # 旧版参数名
            text_threshold=0.25,
            target_sizes=[image.size[::-1]]
        )[0]

    return results


def segment_with_sam2(predictor, image, boxes, device):
    """使用 SAM2 ImagePredictor 分割"""
    predictor.set_image(np.array(image))

    # 合并所有检测到的 box
    if len(boxes) == 0:
        return None

    # 转换到 CPU 进行计算
    boxes_cpu = boxes.cpu()

    # 选择最大的 box（根据实验记录，最大框效果最好）
    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes_cpu]
    largest_idx = np.argmax(areas)
    largest_box = boxes_cpu[largest_idx]

    # debug print (optional)
    # print(f"Using largest box: {largest_box.tolist()}, area: {areas[largest_idx]:.0f}")

    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=largest_box.numpy(),
        multimask_output=False
    )

    return masks[0]  # [H, W] boolean


def segment_with_sam2_hf(processor, model, image, boxes, device):
    """使用 HuggingFace 版本的 SAM2 (备用路径)"""
    if len(boxes) == 0:
        return None

    # 选择最大的 box
    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
    largest_idx = np.argmax(areas)
    largest_box = boxes[largest_idx].cpu().numpy().tolist()

    inputs = processor(
        images=image,
        input_boxes=[[largest_box]],
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.post_process_masks(
        outputs.pred_masks,
        inputs["original_sizes"],
        inputs["reshaped_input_sizes"]
    )

    mask = masks[0][0, 0].cpu().numpy()
    return mask > 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_dir", default="video_frames")
    parser.add_argument("--output_dir", default="masks")
    parser.add_argument("--text_prompt", default="teapot.")
    parser.add_argument("--box_threshold", type=float, default=0.3)
    parser.add_argument("--sample_interval", type=int, default=1,
                        help="Process every N frames (1=all frames)")
    args = parser.parse_args()

    # 获取当前脚本所在目录
    script_dir = Path(__file__).resolve().parent
    
    # 解析输入输出路径
    frames_dir = script_dir / args.frames_dir
    output_dir = script_dir / args.output_dir

    # 检查输入目录是否存在
    if not frames_dir.exists():
        print(f"Error: Frames directory not found at {frames_dir}")
        print("Please ensure your video frames are extracted there.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. 加载模型
    gdino_processor, gdino_model = load_grounding_dino()
    gdino_model = gdino_model.to(device)

    sam2_result = load_sam2()
    
    # 判断 SAM2 加载模式
    if sam2_result[2] == "hf":
        sam2_processor, sam2_model, _ = sam2_result
        sam2_model = sam2_model.to(device)
        use_hf = True
    else:
        sam2_predictor, _, _ = sam2_result
        # 注意: SAM2ImagePredictor 内部已经处理了 device，通常在 set_image 时或者加载时
        # 这里尽量确保 predictor 的 model 在 device 上
        try:
            sam2_predictor.model = sam2_predictor.model.to(device)
        except:
            pass 
        use_hf = False

    # 2. 获取所有帧
    # 支持 jpg 和 png
    frame_files = sorted(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")))
    if not frame_files:
        print(f"No frames found in {frames_dir}")
        return
        
    print(f"Found {len(frame_files)} frames to process")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. 处理每一帧
    print("Starting segmentation loop...")
    for i, frame_file in enumerate(frame_files):
        if i % args.sample_interval != 0:
            continue

        try:
            image = Image.open(frame_file).convert("RGB")
        except Exception as e:
            print(f"Error reading frame {frame_file}: {e}")
            continue

        # 3.1. Grounding DINO 检测
        # 返回字典 {'scores': tensor, 'labels': tensor, 'boxes': tensor}
        results = detect_with_grounding_dino(
            gdino_processor, gdino_model, image,
            args.text_prompt, device, args.box_threshold
        )

        boxes = results["boxes"]

        if len(boxes) == 0:
            # print(f"Frame {i}: No detection")
            mask = np.zeros((image.height, image.width), dtype=np.uint8)
        else:
            # 3.2. SAM2 分割
            if use_hf:
                mask = segment_with_sam2_hf(sam2_processor, sam2_model, image, boxes, device)
            else:
                mask = segment_with_sam2(sam2_predictor, image, boxes, device)

            if mask is None:
                mask = np.zeros((image.height, image.width), dtype=np.uint8)
            else:
                mask = mask.astype(np.uint8) * 255

        # 保存 mask (使用与原文件名相同的名字，但在 output_dir 下，存为 png)
        mask_filename = frame_file.stem + ".png"
        mask_path = output_dir / mask_filename
        Image.fromarray(mask).save(mask_path)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"Processed frame {i + 1}/{len(frame_files)} | "
                  f"Detections: {len(boxes)} | "
                  f"Mask area: {mask.sum() / 255:.0f} px")

    print(f"\nDone! Masks saved to {output_dir}")
    print(f"Total masks: {len(list(output_dir.glob('*.png')))}")


if __name__ == "__main__":
    main()

