# PVTT 评估指标

针对 FFGo 首帧引导视频生成在 PVTT（Product Video Template Transformation）数据集上的综合评估指标。评估将生成的目标视频与原始源视频、参考产品图在三个维度上进行对比。

## 概述

生成视频由 FFGo Pipeline（Wan2.2-I2V-A14B + LoRA）产生：输入为拼贴首帧（左=白底产品图，右=去物体背景），输出 81 帧视频。前 N 帧（默认 4 帧）为转场帧，评估前会被裁掉。剩余帧与原始源视频帧进行对比评估。

## 指标分组

### 第一组：FiVE-Bench 指标（帧级质量 vs 源视频）

评估生成帧相对于源视频帧的视觉质量和结构保真度。

| 指标 | 方向 | 范围 | 说明 |
|------|------|------|------|
| **PSNR** | 越高越好 ↑ | [0, +∞) dB | 峰值信噪比。衡量像素级重建质量，逐帧计算后取平均。视频编辑场景典型范围：15-35 dB。 |
| **SSIM** | 越高越好 ↑ | [0, 1] | 结构相似性指数。衡量对应帧之间的结构、亮度和对比度相似性。1 = 完全相同。 |
| **LPIPS** | 越低越好 ↓ | [0, 1+] | 学习感知图像块相似度（AlexNet 骨干网络）。基于深度特征的感知距离，与人类主观判断高度相关。0 = 感知上完全相同。 |
| **CLIP_tgt** | 越高越好 ↑ | [-1, 1] | CLIP（ViT-B/32）生成帧与目标提示词之间的余弦相似度。衡量图文对齐程度。 |
| **StructDist** | 越低越好 ↓ | [0, 2] | 结构距离：DINOv2（ViT-B/14）CLS token 在生成帧和源帧之间的余弦距离。捕捉像素之上的高层结构相似性。0 = 结构完全相同。 |
| **MFS** | 越高越好 ↑ | [-1, 1] | 平均帧相似度：生成视频中相邻帧之间的 CLIP 图像余弦相似度均值。衡量时序连贯性。 |

### 第二组：Edit Success 指标（编辑效果评估）

评估产品替换编辑是否成功、是否忠实于目标以及时序一致性。

| 指标 | 方向 | 范围 | 说明 |
|------|------|------|------|
| **CLIP_tgt** | 越高越好 ↑ | [-1, 1] | 同第一组，衡量图文对齐程度。 |
| **CLIP_dir** | 越高越好 ↑ | [-1, 1] | CLIP 方向一致性。对每对帧（生成帧, 源帧），计算图像空间编辑向量（CLIP(gen) - CLIP(src)）与文本空间编辑向量（CLIP(target_prompt) - CLIP(source_prompt)）之间的余弦相似度。正值表示视觉变化方向与文本描述的变化方向一致。 |
| **ProdCLIP** | 越高越好 ↑ | [-1, 1] | 生成帧中产品区域（通过源掩码裁剪）与参考产品图（output_dino_rgba/中的 RGBA 图）之间的 CLIP 相似度。衡量产品外观保真度。 |
| **ProdPersist** | 越高越好 ↑ | [0, 1] | 产品时序一致性：1 - 归一化标准差（逐帧 ProdCLIP 分数的 std/mean）。越高 = 产品外观在帧间越稳定。 |
| **F0EditSSIM** | 越高越好 ↑ | [0, 1] | FFGo 输入首帧右半部分（背景/场景参考）与生成视频首帧之间的 SSIM。衡量模型对参考场景的重建能力。 |
| **EditFid** | 越高越好 ↑ | [-1, 1] | 编辑保真度：生成帧与源帧之间的 CLIP 特征余弦相似度均值。衡量编辑后整体场景外观保持程度。 |
| **EditPersist** | 越高越好 ↑ | [0, 1] | 生成视频相邻帧之间的 SSIM 均值。衡量编辑后视频的时序平滑度和一致性。 |

### 第三组：VBench 指标（视频质量）

VBench 风格视频质量指标的简化实现，无需安装完整 VBench 包。

| 指标 | 方向 | 范围 | 说明 |
|------|------|------|------|
| **SubjectCons** | 越高越好 ↑ | [-1, 1] | 主体一致性：在均匀采样的帧（最多 16 帧）之间计算 DINOv2 CLS token 成对余弦相似度均值。衡量主体外观是否在时间维度上保持一致。 |
| **BgCons** | 越高越好 ↑ | [-1, 1] | 背景一致性：相邻帧之间的 CLIP 特征余弦相似度均值。衡量背景在视频中的稳定性。 |
| **TempFlk** | 越高越好 ↑ | [0, 1] | 时序闪烁：1 - mean(|像素差|) / 255（相邻帧之间）。越高 = 闪烁越少。惩罚帧间快速亮度变化。 |
| **DynDeg_gen** | 参考值 | [0, +∞) | 生成视频的动态程度：相邻帧之间 Farneback 光流幅度均值。衡量运动量。 |
| **DynDeg_src** | 参考值 | [0, +∞) | 源视频的动态程度（相同计算方法）。 |
| **DynDeg_delta** | 参考值 | (-∞, +∞) | 差值 = DynDeg_gen - DynDeg_src。正值表示生成视频运动量大于源视频。理想情况下接近 0。 |

## 环境配置

> **请勿在本地电脑上安装！** 以下步骤在 GPU 服务器上执行。

### 前置要求

- Python 3.8+
- CUDA GPU（推荐 16GB+ 显存）
- Conda 或 virtualenv

### 安装步骤（在 GPU 服务器上）

```bash
# 激活你的环境（可以使用已有的 diffsynth 环境）
conda activate diffsynth

# 核心依赖
pip install torch torchvision  # 确保 CUDA 版本与服务器匹配

# CLIP（OpenAI）
pip install git+https://github.com/openai/CLIP.git
# 国内服务器如果 github 访问慢，可用镜像：
# pip install git+https://gitclone.com/github.com/openai/CLIP.git

# LPIPS
pip install lpips

# DINOv2（通过 HuggingFace transformers）
pip install transformers

# 其他依赖
pip install opencv-python-headless scikit-image scipy pandas Pillow tqdm

# 国内服务器设置 HuggingFace 镜像加速模型下载：
export HF_ENDPOINT=https://hf-mirror.com
```

### 模型下载

以下模型在首次使用时自动下载并缓存：

| 模型 | 大小 | 来源 |
|------|------|------|
| CLIP ViT-B/32 | ~350 MB | OpenAI，由 `clip` 包缓存 |
| LPIPS AlexNet | ~10 MB | 由 `lpips` 包缓存 |
| DINOv2 ViT-B/14 | ~350 MB | `facebook/dinov2-base`（HuggingFace） |

提前下载（避免推理时等待）：

```python
import clip; clip.load("ViT-B/32")
import lpips; lpips.LPIPS(net="alex")
from transformers import AutoModel; AutoModel.from_pretrained("facebook/dinov2-base")
```

## 使用方法

### 基本用法

```bash
python pvtt_evaluation/metrics/evaluate_pvtt.py \
    --generated_dir experiments/results/ffgo_original/pvtt/20260317_185832 \
    --dataset_root samples/pvtt_evaluation_datasets \
    --skip_frames 4
```

### 完整参数

```bash
python pvtt_evaluation/metrics/evaluate_pvtt.py \
    --generated_dir experiments/results/ffgo_original/pvtt/TIMESTAMP \
    --dataset_root samples/pvtt_evaluation_datasets \
    --json_path samples/pvtt_evaluation_datasets/edit_prompt/easy_new.json \
    --output_csv results/evaluation_results.csv \
    --output_summary results/evaluation_summary.json \
    --video_filename ffgo_original.mp4 \
    --ref_frame_filename ffgo_original_ref_frame.jpg \
    --skip_frames 4 \
    --eval_h 480 \
    --eval_w 832 \
    --task_ids "0016-bracelet1_to_bracelet3,0021-earring1_to_earring3" \
    --verbose
```

### 使用 Shell 脚本

```bash
bash scripts/run_pvtt_evaluate.sh

# 自定义设置：
GENERATED_DIR=experiments/results/ffgo_original/pvtt/20260317_185832 \
SKIP_FRAMES=4 \
    bash scripts/run_pvtt_evaluate.sh
```

## 输出格式

### 逐任务 CSV（`evaluation_results.csv`）

每行对应一个任务：

```
task_id,video_name,category,PSNR,SSIM,LPIPS,CLIP_tgt,StructDist,MFS,CLIP_dir,...
0016-bracelet1_to_bracelet3,0016-bracelet1,bracelet,18.42,0.5123,0.3456,0.2789,...
```

### 汇总 JSON（`evaluation_summary.json`）

```json
{
  "num_tasks": 14,
  "skip_frames": 4,
  "eval_size": [480, 832],
  "overall_mean": {
    "PSNR": 18.42,
    "SSIM": 0.5123,
    ...
  },
  "overall_std": { ... },
  "per_category": {
    "bracelet": { "count": 2, "mean": { ... }, "std": { ... } },
    "earring":  { "count": 2, "mean": { ... }, "std": { ... } },
    ...
  }
}
```

### 控制台汇总表

脚本结束时打印格式化汇总表，包含：
- 各指标的总体均值和标准差（附方向标识 ↑/↓）
- 按产品类别的分组统计

## 实现细节

1. **分辨率处理**：源视频和生成视频都统一缩放到公共评估分辨率（默认 832×480），确保公平对比。源视频可能是原始分辨率（如 856×480 或 480×856），生成视频为 832×480。

2. **帧对齐**：生成帧列表和源帧列表取较短长度截断。生成视频的前 N 帧转场帧通过 `--skip_frames` 跳过。

3. **产品区域提取**：使用数据集中的分割掩码（`masks/{video_name}/`）提取 bbox 来裁剪产品区域。如果某帧的掩码不可用，回退使用首帧的 bbox。

4. **VBench 简化实现**：SubjectCons、BgCons、TempFlk、DynDeg 是 VBench 指标的简化重实现，捕捉相同的评估直觉，但无需安装完整 VBench 包。SubjectCons 使用 DINOv2 成对相似度；BgCons 使用 CLIP 相邻相似度；TempFlk 使用像素级差分；DynDeg 使用 Farneback 光流。

5. **显存管理**：模型在首次使用时惰性加载并缓存为单例。帧逐个处理以控制 GPU 显存峰值。

## 指标解读指南

对于视频产品编辑任务，好的结果应当：
- **高** PSNR、SSIM（结构接近源视频）
- **低** LPIPS、StructDist（感知上接近源视频）
- **高** CLIP_tgt、CLIP_dir（编辑结果与目标文本匹配）
- **高** ProdCLIP、ProdPersist（产品外观正确且跨帧一致）
- **高** F0EditSSIM（场景从参考首帧中成功重建）
- **高** MFS、EditPersist、SubjectCons、BgCons、TempFlk（时序一致性好）
- **DynDeg_delta 接近 0**（运动量与源视频相当）
