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
  |    |- exp_canny_control.py            # 双参考图（商品图+Canny）实验
  |    |- exp_canny_paste.py              # Canny 首帧粘贴（白/黑对比）实验
  |    |- test_*.py                        # （旧版脚本，已被 exp_*.py 取代）
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
  |    |- extract_bbox_sequence.py
  |    |- sync_to_server.sh
  |    |- README.md                        # 本文档
  |- samples/{样本名}/
  |    |- video_frames/                    # 模板视频帧
  |    |- masks/                           # 精确分割 mask
  |    |- reference_images/                # 商品参考图像
  |- experiments/results/                  # 实验输出
  |- experiment.sbatch                     # SLURM 批处理作业
  |- ComfyUI_VACE工作流分析.md              # ComfyUI 工作流分析文档
```
