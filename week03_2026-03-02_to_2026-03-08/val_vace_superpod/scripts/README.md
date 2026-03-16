# PVTT 实验脚本

基于 VACE 的**商品视频模板转换（Product Video Template Transformation）**实验脚本集。

## 实验列表

### 第一轮实验（基础）

| Shell 脚本 | Python 脚本 | 说明 |
|---|---|---|
| `run_precise_mask.sh` | `exp_precise_mask.py` | 精确 mask + 参考图像 → 商品替换（基础实验） |
| `run_bbox_mask.sh` | `exp_bbox_mask.py` | Bbox mask vs 精确 mask 对比实验 |
| `run_reactive_weight.sh` | `exp_reactive_weight.py` | Reactive 流权重扫描（精确 mask） |
| `run_reactive_weight_bbox.sh` | `exp_reactive_weight_bbox.py` | Reactive 流权重扫描（bbox mask） |

### 第二轮实验（Neutral Fill 预处理）

基于 ComfyUI 成功工作流分析，引入中性色填充预处理来消除 reactive 流的模板锁定效应。

| Shell 脚本 | Python 脚本 | 说明 |
|---|---|---|
| `run_neutral_fill.sh` | `exp_neutral_fill.py` | 中性色填充 + 精确 mask + 参考图像 → 商品替换 |
| `run_neutral_fill_bbox.sh` | `exp_neutral_fill_bbox.py` | 中性色填充 + bbox mask + 参考图像 → 商品替换 |
| `run_neutral_fill_growmask.sh` | `exp_neutral_fill_growmask.py` | 中性色填充 + GrowMask + BlockifyMask + 参考图像 → 商品替换 |

### 第三轮实验（ComfyUI 工作流复刻）

1:1 复刻 ComfyUI VACE 商品替换工作流，引入参考图去背景、LightX2V 蒸馏 LoRA 单步采样等完整流程。

| Shell 脚本 | Python 脚本 | 说明 |
|---|---|---|
| `run_comfyui_baseline.sh` | `exp_comfyui_baseline.py` | ComfyUI 工作流完整复刻：去背景参考图 + 中性色填充 + GrowMask+BlockifyMask + LightX2V LoRA 单步推理 |

**额外依赖**：`pip install rembg[gpu] safetensors`
**额外模型**：`models/LightX2V/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors`（LoRA 仅兼容 14B 模型）

### 第四轮实验（Canny 结构控制）

利用参考图像的 Canny 边缘作为结构控制信号，引导模型按照目标物体的轮廓生成替换内容。

| Shell 脚本 | Python 脚本 | 说明 |
|---|---|---|
| `run_canny_control.sh` | `exp_canny_control.py` | 双参考图：给 VACE 同时传入商品图 + Canny 结构图作为参考（输入帧保持标准中性色填充） |
| `run_canny_paste.sh` | `exp_canny_paste.py` | Canny 首帧粘贴：仅首帧 mask 区域嵌入 Canny 边缘，白色/黑色两个变体依次运行 |

**额外依赖**：`pip install opencv-python-headless`（通常已随 DiffSynth-Studio 安装）

## 快速开始

```bash
# 运行单个实验（默认使用 1.3B 模型）
bash scripts/run_neutral_fill.sh

# 使用 14B 模型运行
WAN_VACE_MODEL_SIZE=14B bash scripts/run_neutral_fill.sh

# 自定义参数（通过环境变量）
SEED=123 WIDTH=480 HEIGHT=848 bash scripts/run_neutral_fill_bbox.sh

# 运行 GrowMask 实验（自定义膨胀像素和网格大小）
GROW_PIXELS=15 BLOCK_SIZE=32 bash scripts/run_neutral_fill_growmask.sh

# 通过 SLURM 批量运行所有实验
sbatch experiment.sbatch
```

## 参数修改方式

有**两种方式**修改实验参数：

### 方式一：通过环境变量（推荐用于临时修改）

在运行 sh 脚本前设置环境变量，脚本会自动使用这些值覆盖默认值：

```bash
# 修改单个参数
SEED=123 bash scripts/run_neutral_fill.sh

# 修改多个参数
WAN_VACE_MODEL_SIZE=14B SEED=123 CFG_SCALE=6.0 bash scripts/run_neutral_fill.sh

# 修改提示词（注意用引号包裹含空格的值）
PROMPT="a red sports car, product display" bash scripts/run_neutral_fill.sh
```

### 方式二：直接修改 sh 文件中的 python 命令行（推荐用于固定修改）

打开 sh 文件，找到底部的 `python3` 命令，直接修改 `--参数名 参数值`。

**基本格式**：`--参数名 参数值`（用空格分隔，不需要方括号）

```bash
# 修改前（sh 文件中的 python 命令）：
python3 "$BASELINE_DIR/exp_neutral_fill.py" \
    --sample_dir "$SAMPLE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --width $WIDTH \
    --height $HEIGHT \
    --seed $SEED \
    --cfg_scale $CFG_SCALE \
    ...

# 修改后（直接写死参数值）：
python3 "$BASELINE_DIR/exp_neutral_fill.py" \
    --sample_dir "$SAMPLE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --width 480 \
    --height 848 \
    --seed 123 \
    --cfg_scale 6.0 \
    ...
```

**各参数类型的格式示例**：

```bash
# 整数参数：直接写数字
--seed 123
--num_inference_steps 30
--fill_value 128
--grow_pixels 15
--block_size 32
--width 480
--height 848
--num_frames 49

# 浮点数参数：直接写小数
--cfg_scale 6.0

# 字符串参数：用引号包裹（尤其是含空格的值）
--prompt "a red sports car, product display, studio lighting"
--negative_prompt "low quality, blurry, deformed"
--reference_image "my_product.png"
--model_size "14B"

# 逗号分隔列表（仅 reactive weight 实验）：用引号包裹，逗号分隔，不要加方括号
--weights "0.0,0.3,0.5,0.7,1.0"
```

**注意**：
- 参数值**不需要**方括号 `[]`，直接写值即可
- 每个 `--参数名 参数值` 对用空格分隔
- sh 文件中的 `$变量名` 会被环境变量覆盖；如果想固定某个值，直接写死数字或字符串
- 两种方式可以混用：部分参数用环境变量，部分参数在 sh 文件中写死

### 直接运行 Python 脚本（不通过 sh）

也可以跳过 sh 文件，直接运行 Python 脚本：

```bash
# 先 cd 到 baseline 目录
cd baseline/wan2.1-vace

# 直接运行（所有参数都在命令行指定）
python exp_neutral_fill.py \
    --sample_dir ../../samples/teapot \
    --output_dir ../../experiments/results/14B/neutral_fill/20240315_143022 \
    --width 480 \
    --height 848 \
    --num_frames 49 \
    --seed 42 \
    --prompt "yellow rubber duck toy, product display, studio lighting" \
    --reference_image ref_rubber_duck.png \
    --model_size 14B \
    --num_inference_steps 50 \
    --cfg_scale 6.0 \
    --fill_value 128

# 查看所有可用参数及默认值
python exp_neutral_fill.py --help
```

## 参数说明

### 通用参数（所有实验共享）

| 环境变量 | CLI 参数 | 默认值 | 说明 |
|---|---|---|---|
| `WAN_VACE_MODEL_SIZE` | `--model_size` | `1.3B` | 模型大小（`1.3B` 或 `14B`） |
| `SAMPLE_NAME` | `--sample_dir` | `teapot` | 样本目录名（位于 `samples/` 下） |
| `OUTPUT_DIR` | `--output_dir` | （各实验不同） | 输出目录 |
| `WIDTH` | `--width` | `480` | 视频宽度 |
| `HEIGHT` | `--height` | `848` | 视频高度 |
| `NUM_FRAMES` | `--num_frames` | `49` | 帧数 |
| `SEED` | `--seed` | `42` | 随机种子 |
| `FPS` | `--fps` | `15` | 输出视频帧率 |
| `PROMPT` | `--prompt` | `"yellow rubber duck toy, ..."` | 生成提示词 |
| `NEGATIVE_PROMPT` | `--negative_prompt` | `"low quality, blurry, deformed"` | 负面提示词 |
| `REFERENCE_IMAGE` | `--reference_image` | `ref_rubber_duck.png` | 参考图像文件名 |
| `NUM_INFERENCE_STEPS` | `--num_inference_steps` | `50` | 扩散采样步数 |
| `CFG_SCALE` | `--cfg_scale` | `7.5` | CFG 引导强度 |

### Neutral Fill 实验专有参数

| 环境变量 | CLI 参数 | 默认值 | 说明 | 适用实验 |
|---|---|---|---|---|
| `FILL_VALUE` | `--fill_value` | `128` | 中性色填充灰度值（0-255） | 全部 3 个 neutral fill 实验 |

### Bbox 实验专有参数

| 环境变量 | CLI 参数 | 默认值 | 说明 | 适用实验 |
|---|---|---|---|---|
| `BBOX_MARGIN_LR` | `--bbox_margin_lr` | `20` | Bbox 左右扩展像素 | neutral_fill_bbox, bbox_mask |
| `BBOX_MARGIN_TOP` | `--bbox_margin_top` | `20` | Bbox 上方扩展像素 | 同上 |
| `BBOX_MARGIN_BOTTOM` | `--bbox_margin_bottom` | `20` | Bbox 下方扩展像素 | 同上 |

### GrowMask 实验专有参数

| 环境变量 | CLI 参数 | 默认值 | 说明 |
|---|---|---|---|
| `GROW_PIXELS` | `--grow_pixels` | `10` | Mask 膨胀像素数 |
| `BLOCK_SIZE` | `--block_size` | `32` | Mask 网格对齐块大小 |

### Canny 结构控制实验专有参数

| 环境变量 | CLI 参数 | 默认值 | 说明 |
|---|---|---|---|
| `CANNY_LOW` | `--canny_low` | `50` | Canny 边缘检测低阈值 |
| `CANNY_HIGH` | `--canny_high` | `150` | Canny 边缘检测高阈值 |
| `NO_REMBG` | `--no_rembg` | `0`（启用） | 设为 `1` 禁用 rembg 前景分割，回退到亮度阈值去背景 |
| `BG_THRESHOLD` | `--bg_threshold` | `240` | 亮度阈值回退方案：RGB 均 > 此值视为背景（仅 rembg 不可用时生效） |
| `BLUR_KSIZE` | `--blur_ksize` | `3` | Canny 前高斯模糊核大小（0 = 不模糊） |

> **Canny 前景分割**：默认使用 rembg 神经网络精确分离产品前景，去除背景和阴影后再提取 Canny 边缘。rembg 不可用时自动回退到基于 `BG_THRESHOLD` 的亮度阈值方案。
>
> `canny_paste` 实验在同一次运行中依次产出白色线条和黑色线条两组结果，无需手动指定颜色。

### ComfyUI 复刻实验专有参数

| 环境变量 | CLI 参数 | 默认值 | 说明 |
|---|---|---|---|
| `LORA_PATH` | `--lora_path` | `models/LightX2V/...rank32.safetensors` | LightX2V 蒸馏 LoRA 路径 |
| `LORA_STRENGTH` | `--lora_strength` | `1.0` | LoRA 合并强度 |
| `EMBEDDED_CFG_SCALE` | `--embedded_cfg_scale` | `5.0` | Wan CFG_Star 参数 |
| `REMBG_MODEL` | `--rembg_model` | `birefnet-general` | 参考图去背景模型 |

> **注意**：ComfyUI 复刻实验默认 `NUM_INFERENCE_STEPS=1`（依赖 LoRA 蒸馏），`CFG_SCALE=6.0`。LoRA 仅兼容 14B 模型；使用 1.3B 模型时 LoRA 加载将失败并自动回退至 50 步采样。

### Reactive 权重实验专有参数

| 环境变量 | CLI 参数 | 默认值 | 说明 |
|---|---|---|---|
| `WEIGHTS` | `--weights` | `0.0` | 逗号分隔的权重值列表（如 `"0.0,0.3,0.7"`） |

## 输出结构

输出路径格式：`experiments/results/{模型大小}/{实验名}/{时间戳}/`

- **模型大小**（`1.3B` / `14B`）：区分不同模型的实验结果
- **时间戳**（`YYYYMMDD_HHMMSS`）：每次运行自动生成，防止结果覆盖

```
experiments/results/
  |- 14B/
  |    |- neutral_fill/
  |    |    |- 20240315_143022/            # 第一次运行
  |    |    |    |- neutral_fill.mp4
  |    |    |    |- neutral_fill_comparison.jpg
  |    |    |    |- neutral_fill_showcase.jpg
  |    |    |    |- preprocess_comparison.jpg
  |    |    |    |- experiment.log
  |    |    |- 20240316_091500/            # 第二次运行（不会覆盖）
  |    |    |    |- ...
  |    |- neutral_fill_bbox/
  |    |    |- 20240315_143522/
  |    |    |    |- neutral_fill_bbox.mp4
  |    |    |    |- neutral_fill_bbox_comparison.jpg
  |    |    |    |- neutral_fill_bbox_showcase.jpg
  |    |    |    |- preprocess_comparison.jpg
  |    |    |    |- experiment.log
  |    |- neutral_fill_growmask/
  |    |    |- 20240315_144022/
  |    |    |    |- neutral_fill_growmask.mp4
  |    |    |    |- neutral_fill_growmask_comparison.jpg
  |    |    |    |- neutral_fill_growmask_showcase.jpg
  |    |    |    |- mask_comparison.jpg
  |    |    |    |- preprocess_comparison.jpg
  |    |    |    |- experiment.log
  |    |- precise_mask/
  |    |    |- 20240315_150000/
  |    |    |    |- precise_mask.mp4
  |    |    |    |- precise_mask_comparison.jpg
  |    |    |    |- precise_mask_showcase.jpg
  |    |    |    |- experiment.log
  |    |- bbox_mask/
  |    |    |- ...
  |    |- reactive_weight/
  |    |    |- ...
  |    |- reactive_weight_bbox/
  |    |    |- ...
  |- 1.3B/
  |    |- neutral_fill/
  |    |    |- 20240315_160000/
  |    |    |    |- ...
  |    |- ...
```

## 命名规范

| 组件 | 命名模式 | 示例 |
|---|---|---|
| Python 实验脚本 | `exp_{名称}.py` | `exp_neutral_fill.py` |
| Shell 启动脚本 | `run_{名称}.sh` | `run_neutral_fill.sh` |
| 输出视频 | `{名称}.mp4` | `neutral_fill.mp4` |
| 对比帧 | `{名称}_comparison.jpg` | `neutral_fill_comparison.jpg` |
| 首末帧展示 | `{名称}_showcase.jpg` | `neutral_fill_showcase.jpg` |
| 预处理对比 | `preprocess_comparison.jpg` | — |

### 第五轮实验（PVTT 数据集评估）

在 pvtt_evaluation_datasets（199 个任务、53 个视频、35 个商品图）上批量运行实验。默认采样每种产品 3 个任务（共 23 个）。

**两步流程：**

| 步骤 | Shell 脚本 | Python 脚本 | 说明 |
|------|---|---|---|
| Step 1 | `run_pvtt_extract_masks.sh` | `pvtt_evaluation/extract_masks.py` | GroundingDINO + SAM2 从视频中提取商品掩码序列 |
| Step 2 | `run_pvtt_canny_paste.sh` | `pvtt_evaluation/run_pvtt_canny_paste.py` | Canny 首帧粘贴 + GrowMask（81帧，采样 23 任务） |
| Step 2' | `run_pvtt_canny_paste_bbox.sh` | `pvtt_evaluation/run_pvtt_canny_paste_bbox.py` | Canny 首帧粘贴 + **Bbox Mask**（81帧，采样 23 任务） |

**额外依赖**：GroundingDINO + SAM2（详细安装见下文 [GroundingDINO + SAM2 安装指南](#groundingdino--sam2-安装指南) 章节）

**SLURM 一键提交**：`sbatch pvtt_evaluation.sbatch`（自动执行 Step 1 + Step 2）

```bash
# Canny + GrowMask（默认采样 23 任务）
bash scripts/run_pvtt_canny_paste.sh

# Canny + Bbox Mask（默认采样 23 任务）
bash scripts/run_pvtt_canny_paste_bbox.sh

# 跑全部 199 个任务
SAMPLED=0 bash scripts/run_pvtt_canny_paste.sh

# 断点续跑
SKIP_EXISTING=1 bash scripts/run_pvtt_canny_paste.sh

# 仅运行指定任务
TASK_IDS="0001-handfan1_to_handfan2" bash scripts/run_pvtt_canny_paste.sh
```

### PVTT 评估专有参数

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `WAN_VACE_MODEL_SIZE` | `1.3B` | 模型大小 |
| `START_IDX` | `0` | 从第几个任务开始（断点续跑） |
| `MAX_TASKS` | （全部） | 最多运行几个任务 |
| `TASK_IDS` | （全部） | 逗号分隔的任务 ID |
| `SKIP_EXISTING` | `0` | 设为 `1` 跳过已有输出的任务 |
| `SKIP_MASK_EXTRACTION` | `0` | 设为 `1` 跳过掩码提取步骤 |
| `SKIP_INFERENCE` | `0` | 设为 `1` 跳过推理步骤 |
| `SAM2_CHECKPOINT` | `/home/hxiaoap/models/sam2.1_hiera_large.pt` | SAM2 权重路径 |
| `SAM2_CONFIG` | `configs/sam2.1/sam2.1_hiera_l.yaml` | SAM2 配置路径 |
| `GDINO_CONFIG` | `/home/hxiaoap/models/GroundingDINO_SwinT_OGC.py` | GroundingDINO 配置路径 |
| `GDINO_CHECKPOINT` | `/home/hxiaoap/models/groundingdino_swint_ogc.pth` | GroundingDINO 权重路径 |

### PVTT 输出结构

```
# Canny + GrowMask
experiments/results/{MODEL_SIZE}/pvtt_canny_paste/{TIMESTAMP}/
├── summary.json
├── easy_0001-handfan1_to_handfan2/
│   ├── canny_paste_white.mp4                    # 合成后目标视频
│   ├── canny_paste_white_comparison.jpg         # [原视频首帧|VACE输入首帧|ref img|target首帧] + prompt
│   ├── canny_paste_white_showcase.jpg           # [target首帧|target尾帧]
│   ├── ref_canny.png                            # Canny 边缘图
│   └── experiment.log
└── ...

# Canny + Bbox Mask
experiments/results/{MODEL_SIZE}/pvtt_canny_paste_bbox/{TIMESTAMP}/
├── summary.json
├── easy_0001-handfan1_to_handfan2/
│   ├── canny_paste_bbox.mp4                     # 合成后目标视频
│   ├── canny_paste_bbox_comparison.jpg          # [原视频首帧|VACE输入首帧|ref img|target首帧] + prompt
│   ├── canny_paste_bbox_showcase.jpg            # [target首帧|target尾帧]
│   ├── ref_canny.png                            # Canny 边缘图
│   └── experiment.log
└── ...
```

### 第六轮实验（PVTT 数据集 - 更多工作流对比）

在 PVTT 数据集上运行两个新工作流，采样每种产品 3 个任务（共 23 个），与第五轮实验 10（Canny 首帧粘贴）对比。

| Shell 脚本 | Python 脚本 | 说明 |
|---|---|---|
| `run_pvtt_neutral_fill_bbox.sh` | `pvtt_evaluation/run_pvtt_neutral_fill_bbox.py` | **实验 6 工作流**：中性色填充 + Bbox mask + ref img + prompt → mask 区域替换 |
| `run_pvtt_ffgo_i2v.sh` | `pvtt_evaluation/run_pvtt_ffgo_i2v.py` | **VACE I2V + FFGo 首帧**：拼贴首帧（产品+去物体背景）+ 后续帧全1 mask → I2V 生成 |

**采样策略**：8 种产品类别 x 3 个任务（handfan 仅 2 个），共 23 个任务。默认 `SAMPLED=1` 只跑采样子集。

```bash
# 实验 6：中性色填充 + Bbox mask
bash scripts/run_pvtt_neutral_fill_bbox.sh

# FFGo I2V：拼贴首帧 + I2V 生成
bash scripts/run_pvtt_ffgo_i2v.sh

# 跑全部 199 个任务（不采样）
SAMPLED=0 bash scripts/run_pvtt_neutral_fill_bbox.sh

# 自定义物体移除方式（FFGo）
INPAINT_METHOD=neutral_fill bash scripts/run_pvtt_ffgo_i2v.sh

# 添加场景转换前缀（FFGo）
TRANSITION_PREFIX="the scene transitions to" bash scripts/run_pvtt_ffgo_i2v.sh
```

#### 实验 6 (Neutral Fill Bbox) 专有参数

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `FILL_VALUE` | `128` | 中性色填充灰度值 |
| `BBOX_MARGIN_LR` | `20` | Bbox 左右扩展像素 |
| `BBOX_MARGIN_TOP` | `20` | Bbox 上方扩展像素 |
| `BBOX_MARGIN_BOTTOM` | `20` | Bbox 下方扩展像素 |

#### FFGo I2V 专有参数

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `INPAINT_METHOD` | `lama` | 首帧物体移除方式: `lama`(推荐) / `cv2` / `neutral_fill` |
| `FILL_VALUE` | `128` | neutral_fill 灰度值 |
| `DILATE_PIXELS` | `15` | 物体移除前 mask 膨胀像素数 |
| `TRANSITION_PREFIX` | `The camera view suddenly changes.` | Prompt 前缀（场景转换提示词） |
| `DISCARD_FRAMES` | `4` | 最少丢弃转场帧数（实际通过 SSIM 自动检测） |
| `USE_REF_IMAGE` | `1` | 是否同时传 VACE reference image (`0`=不传) |

**额外依赖（FFGo I2V）**：`pip install simple-lama-inpainting`（用于高质量物体去除，首次运行自动下载模型）

#### 第六轮输出结构

```
# 实验 6
experiments/results/{MODEL_SIZE}/pvtt_neutral_fill_bbox/{TIMESTAMP}/
├── summary.json
├── easy_0001-handfan1_to_handfan2/
│   ├── neutral_fill_bbox.mp4                   # 合成后目标视频
│   ├── neutral_fill_bbox_comparison.jpg        # [原始首帧|mask video首帧|ref img|target首帧]
│   ├── neutral_fill_bbox_showcase.jpg          # [target首帧|target尾帧]
│   └── experiment.log
└── ...

# FFGo I2V
experiments/results/{MODEL_SIZE}/pvtt_ffgo_i2v/{TIMESTAMP}/
├── summary.json
├── easy_0001-handfan1_to_handfan2/
│   ├── ffgo_i2v.mp4                            # 去转场后的有效视频
│   ├── ffgo_i2v_full.mp4                       # 完整视频（含转场帧，供分析）
│   ├── ffgo_i2v_comparison.jpg                 # [原视频首帧|FFGo首帧|转场后首帧]
│   ├── ffgo_i2v_showcase.jpg                   # [转场后首帧|尾帧]
│   ├── ffgo_ref_frame.jpg                      # 输入给模型的 FFGo 式拼贴首帧
│   └── experiment.log
└── ...
```

## GroundingDINO + SAM2 安装指南

## LaMa 安装指南（FFGo I2V 物体移除）

FFGo I2V 实验使用 LaMa (Large Mask Inpainting) 从首帧中移除被替换物体。安装非常简单：

```bash
pip install simple-lama-inpainting
```

- 首次运行时会自动从网络下载 LaMa 模型权重（~200MB）
- 国内服务器如果下载缓慢，可以设置 `HF_ENDPOINT=https://hf-mirror.com`
- 如果安装失败或不想用 LaMa，可以降级使用 OpenCV inpainting：`INPAINT_METHOD=cv2 bash scripts/run_pvtt_ffgo_i2v.sh`

PVTT 掩码提取（Step 1）依赖 GroundingDINO（目标检测）和 SAM2（视频分割）。以下是在 **国内服务器** 上的完整安装流程。

### 前置条件

```bash
# 确认 CUDA 和 PyTorch 已安装
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 预期输出: 2.x.x True
```

### 1. SAM2 安装（从源码，解决国内 pip 卡住问题）

由于 `pip install sam2` 需要从 GitHub 下载，国内服务器经常卡住。推荐 **从源码安装**：

```bash
# ---------- 方法 A：直接 git clone（需要能访问 GitHub）----------
cd /home/hxiaoap
git clone https://github.com/facebookresearch/sam2.git
cd sam2

# ---------- 方法 B：如果 git clone 也卡，用镜像 ----------
# 方式 B1: 使用 gitclone 镜像
git clone https://gitclone.com/github.com/facebookresearch/sam2.git
cd sam2

# 方式 B2: 使用 ghproxy 镜像
git clone https://ghproxy.com/https://github.com/facebookresearch/sam2.git
cd sam2

# 方式 B3: 本地下载 zip 后上传到服务器
# 在本地浏览器访问: https://github.com/facebookresearch/sam2/archive/refs/heads/main.zip
# scp sam2-main.zip hxiaoap@server:/home/hxiaoap/
# ssh server
# cd /home/hxiaoap && unzip sam2-main.zip && mv sam2-main sam2 && cd sam2

# ---------- 从源码安装（所有方法通用的后续步骤）----------
# 激活你的 conda 环境
conda activate diffsynth

# 安装 SAM2（editable mode，这样后续更新只需 git pull）
pip install -e ".[notebooks]"

# 验证安装
python -c "from sam2.build_sam import build_sam2_video_predictor; print('SAM2 安装成功')"
```

#### SAM2 权重下载

```bash
# 创建模型存放目录
mkdir -p /home/hxiaoap/models

# 下载 SAM2.1 Hiera Large 权重（推荐，精度最高）
# 方法 A：直接下载（如果能访问 GitHub）
wget -O /home/hxiaoap/models/sam2.1_hiera_large.pt \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# 方法 B：通过 HuggingFace 镜像下载
pip install huggingface_hub
HF_ENDPOINT=https://hf-mirror.com python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('facebook/sam2.1-hiera-large', 'sam2.1_hiera_large.pt',
                local_dir='/home/hxiaoap/models')
"

# 方法 C：手动从 HuggingFace 下载后上传
# 浏览器访问: https://huggingface.co/facebook/sam2.1-hiera-large
# 下载 sam2.1_hiera_large.pt，然后 scp 到服务器
```

#### SAM2 配置文件（sam2.1_hiera_l.yaml）

配置文件 **已经包含在 SAM2 源码中**，无需单独下载：

```bash
# 如果你是从源码安装的（推荐），配置文件在：
ls /home/hxiaoap/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml

# 设置环境变量指向它
export SAM2_CONFIG="/home/hxiaoap/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
```

**注意**：如果你的 SAM2 源码不在 `/home/hxiaoap/sam2`，请根据实际路径修改。`sam2/configs/` 目录下有多种配置：

| 配置文件 | 模型大小 | 推荐场景 |
|---------|---------|---------|
| `sam2.1_hiera_t.yaml` | Tiny | 快速测试 |
| `sam2.1_hiera_s.yaml` | Small | 资源受限 |
| `sam2.1_hiera_b+.yaml` | Base+ | 均衡 |
| `sam2.1_hiera_l.yaml` | Large | **推荐（最佳精度）** |

### 2. GroundingDINO 安装

有两种安装方式，**推荐方式 A**（通过 transformers 库，无需编译 CUDA，避免版本冲突）：

#### 方式 A：通过 transformers 库安装（推荐）

这种方式**不需要编译 CUDA 代码**，不会遇到 CUDA 版本不匹配的问题。

```bash
conda activate diffsynth

# 安装/升级 transformers（需要 >= 4.40）
pip install -U transformers

# 验证安装
python -c "
from transformers import AutoModelForZeroShotObjectDetection
print('GroundingDINO (transformers) 安装成功')
"
```

使用 transformers 后端时，模型权重会从 HuggingFace 自动下载（model ID: `IDEA-Research/grounding-dino-tiny`），**不需要** `--gdino_config` 参数和单独下载的 `.pth` 权重。

如果网络不好，可以提前下载模型到本地：
```bash
# 提前下载到本地缓存（国内用 HF 镜像）
HF_ENDPOINT=https://hf-mirror.com python -c "
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-tiny')
AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-tiny')
print('模型已缓存到本地')
"
```

**强制使用 transformers 后端**：
```bash
GDINO_BACKEND=transformers bash scripts/run_pvtt_extract_masks.sh
```

#### 方式 B：从源码安装（遇到 CUDA 版本冲突时不推荐）

> **常见报错**：`RuntimeError: The detected CUDA version (X.X) mismatches the version that was used to compile PyTorch (Y.Y)`
>
> 这是因为源码安装需要编译 CUDA 代码，要求系统 CUDA toolkit 版本与 PyTorch 编译时使用的版本一致。如果遇到此问题，**请直接使用方式 A**。

如果确实需要源码安装（如需要特定版本或性能优化）：

```bash
conda activate diffsynth

# pip 安装（需 CUDA 版本匹配）
pip install groundingdino-py

# 或从源码安装
cd /home/hxiaoap
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .

# 验证
python -c "from groundingdino.util.inference import load_model; print('GroundingDINO 安装成功')"
```

#### GroundingDINO 权重和配置文件下载

> **注意**：使用 transformers 后端（方式 A）时，以下下载步骤**可跳过**，模型会自动从 HuggingFace 下载。

以下仅在使用源码后端（方式 B）时需要：

```bash
mkdir -p /home/hxiaoap/models

# --- 下载模型权重 ---
# 方法 A：直接下载
wget -O /home/hxiaoap/models/groundingdino_swint_ogc.pth \
    https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# 方法 B：镜像下载
wget -O /home/hxiaoap/models/groundingdino_swint_ogc.pth \
    https://ghproxy.com/https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# 方法 C：从 HuggingFace 下载
HF_ENDPOINT=https://hf-mirror.com python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('ShilongLiu/GroundingDINO', 'groundingdino_swint_ogc.pth',
                local_dir='/home/hxiaoap/models')
"

# --- 下载配置文件（GroundingDINO_SwinT_OGC.py）---
# 如果是从源码安装的，配置文件在：
# /home/hxiaoap/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py
# 复制到 models 目录便于管理：
cp /home/hxiaoap/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
   /home/hxiaoap/models/

# 如果是 pip 安装的，需要单独下载配置文件：
wget -O /home/hxiaoap/models/GroundingDINO_SwinT_OGC.py \
    https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py
```

### 3. 其他依赖

```bash
pip install opencv-python-headless supervision
```

### 4. 验证完整安装

```bash
# 一次性验证所有组件
python -c "
import torch
print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')

from groundingdino.util.inference import load_model
print('GroundingDINO: OK')

from sam2.build_sam import build_sam2_video_predictor
print('SAM2: OK')

import cv2
print(f'OpenCV: {cv2.__version__}')

print('所有依赖安装成功！')
"
```

### 5. 配置路径汇总

安装完成后，确认以下文件存在：

```bash
# 检查所有必需文件
ls -la /home/hxiaoap/models/sam2.1_hiera_large.pt       # SAM2 权重 (~900MB)
ls -la /home/hxiaoap/models/groundingdino_swint_ogc.pth  # GDINO 权重 (~700MB)
ls -la /home/hxiaoap/models/GroundingDINO_SwinT_OGC.py   # GDINO 配置

# SAM2 配置文件（在源码目录中）
ls -la /home/hxiaoap/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml
```

然后在 `run_pvtt_extract_masks.sh` 中设置对应路径（或通过环境变量覆盖）：

```bash
# 方式 1：修改 sh 文件中的默认值
SAM2_CHECKPOINT="/home/hxiaoap/models/sam2.1_hiera_large.pt"
SAM2_CONFIG="/home/hxiaoap/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
GDINO_CONFIG="/home/hxiaoap/models/GroundingDINO_SwinT_OGC.py"
GDINO_CHECKPOINT="/home/hxiaoap/models/groundingdino_swint_ogc.pth"

# 方式 2：通过环境变量覆盖
SAM2_CHECKPOINT=/my/path/sam2.pt \
SAM2_CONFIG=/my/path/sam2.1_hiera_l.yaml \
    bash scripts/run_pvtt_extract_masks.sh
```

### 常见问题

**Q: `pip install sam2` 一直卡住？**
A: 因为它会从 GitHub 拉取依赖。使用上面的 **从源码安装** 方法（方法 B/C），先下载源码再 `pip install -e .`。

**Q: GroundingDINO 编译失败（CUDA 版本不匹配）？**
A: 这是最常见的问题。报错通常是：`RuntimeError: The detected CUDA version (X.X) mismatches the version that was used to compile PyTorch (Y.Y)`。

**推荐解决方案：改用 transformers 后端**（完全不需要编译 CUDA）：
```bash
pip install -U transformers
GDINO_BACKEND=transformers bash scripts/run_pvtt_extract_masks.sh
```

如果一定要用源码后端，需确保系统 CUDA 与 PyTorch CUDA 版本匹配：
```bash
# 查看 PyTorch 编译时的 CUDA 版本
python -c "import torch; print(torch.version.cuda)"
# 查看系统 CUDA 版本
nvcc --version
# 两者必须主版本一致（如都是 12.x）

# 如果不一致，加载匹配的 CUDA module 后再安装：
module load cuda12.2/toolkit/12.2.2  # 根据 PyTorch CUDA 版本选择
pip install groundingdino-py
```

**Q: SAM2 报 `FileNotFoundError: ... sam2.1_hiera_l.yaml`？**
A: 配置文件在 SAM2 **源码目录** 中，不在权重旁边。确保 `SAM2_CONFIG` 指向源码中的 `sam2/configs/sam2.1/sam2.1_hiera_l.yaml`。

**Q: GroundingDINO 检测不到目标物体？**
A: 尝试降低 `BOX_THRESHOLD`（默认 0.3 → 试 0.15）和 `TEXT_THRESHOLD`（默认 0.25 → 试 0.15）：
```bash
BOX_THRESHOLD=0.15 TEXT_THRESHOLD=0.15 bash scripts/run_pvtt_extract_masks.sh
```

**Q: SAM2 显存不足 (OOM)？**
A: 换用更小的 SAM2 配置：
```bash
SAM2_CONFIG="/home/hxiaoap/sam2/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml"
SAM2_CHECKPOINT="/home/hxiaoap/models/sam2.1_hiera_base_plus.pt"
```

## 工具脚本

| 文件 | 说明 |
|---|---|
| `extract_bbox_sequence.py` | 从 mask 序列中提取 bbox 轨迹 |
| `sync_to_server.sh` | 将项目 rsync 到 GPU 服务器 |

## 项目目录结构

```
val_vace_superpod/
  |- baseline/wan2.1-vace/
  |    |- utils.py                         # 共享工具库
  |    |- exp_precise_mask.py              # 精确 mask 实验
  |    |- exp_bbox_mask.py                 # Bbox mask 对比实验
  |    |- exp_reactive_weight.py           # Reactive 权重实验（精确 mask）
  |    |- exp_reactive_weight_bbox.py      # Reactive 权重实验（bbox mask）
  |    |- exp_neutral_fill.py              # 中性色填充 + 精确 mask 实验
  |    |- exp_neutral_fill_bbox.py         # 中性色填充 + bbox mask 实验
  |    |- exp_neutral_fill_growmask.py     # 中性色填充 + GrowMask + BlockifyMask 实验
  |    |- exp_comfyui_baseline.py          # ComfyUI 工作流 1:1 复刻实验
  |    |- exp_canny_control.py             # 双参考图（商品图+Canny）实验
  |    |- exp_canny_paste.py               # Canny 首帧粘贴（白/黑对比）实验
  |- pvtt_evaluation/                      # PVTT 数据集评估模块
  |    |- extract_masks.py                 # GroundingDINO + SAM2 掩码提取
  |    |- run_pvtt_canny_paste.py          # 批量 Canny 首帧粘贴推理（GrowMask）
  |    |- run_pvtt_canny_paste_bbox.py     # 批量 Canny 首帧粘贴推理（Bbox Mask）
  |    |- run_pvtt_neutral_fill_bbox.py    # 批量中性色填充 + Bbox Mask 推理
  |    |- run_pvtt_ffgo_i2v.py             # 批量 FFGo I2V 推理
  |- scripts/
  |    |- run_precise_mask.sh
  |    |- run_bbox_mask.sh
  |    |- run_reactive_weight.sh
  |    |- run_reactive_weight_bbox.sh
  |    |- run_neutral_fill.sh
  |    |- run_neutral_fill_bbox.sh
  |    |- run_neutral_fill_growmask.sh
  |    |- run_comfyui_baseline.sh          # ComfyUI 工作流复刻
  |    |- run_canny_control.sh             # 双参考图（商品图+Canny）
  |    |- run_canny_paste.sh               # Canny 首帧粘贴（白/黑）
  |    |- run_pvtt_extract_masks.sh        # PVTT 掩码提取
  |    |- run_pvtt_canny_paste.sh          # PVTT Canny + GrowMask
  |    |- run_pvtt_canny_paste_bbox.sh     # PVTT Canny + Bbox Mask
  |    |- run_pvtt_neutral_fill_bbox.sh    # PVTT 中性色填充 + Bbox Mask
  |    |- run_pvtt_ffgo_i2v.sh             # PVTT FFGo I2V
  |    |- extract_bbox_sequence.py
  |    |- sync_to_server.sh
  |    |- README.md                        # 本文档
  |- samples/pvtt_evaluation_datasets/     # 评估数据集
  |    |- edit_prompt/easy_new.json        # 199 个任务定义
  |    |- videos/                          # 53 个源视频 (mp4)
  |    |- product_images/                  # 35 个商品参考图
  |    |    |- output_dino_rgba/           # DINO+SAM2 预处理的 RGBA 产品图
  |    |- video_frames/                    # 提取的视频帧（由 extract_masks.py 生成）
  |    |- masks/                           # 提取的掩码（由 extract_masks.py 生成）
  |- samples/{样本名}/                     # 单样本实验数据
  |    |- video_frames/
  |    |- masks/
  |    |- reference_images/
  |- experiments/results/                  # 实验输出
  |- experiment.sbatch                     # 单样本实验 SLURM 作业
  |- pvtt_evaluation.sbatch                # PVTT 数据集评估 SLURM 作业
  |- experiment_report.md                  # 实验报告
  |- wan_based_papers_survey.md            # Wan 模型论文调研
```
