# PVTT FFGO训练数据集构建计划

> **目标**: 构建用于 Wan2.2 TI2V 5B + FFGO LoRA 微调的训练数据集
> **数据量**: 50-60个高质量样本
> **交付日期**: [待定]
> **负责人**: [造数据的伙伴]

---

## 1. 背景简介

### 1.1 我们要做什么

我们要在 Wan2.2 TI2V 5B（5B参数的文本+图像→视频模型）上微调一个LoRA，复现FFGO论文（First Frame Is the Place to Go for Video Content Customization）的工作。

**我们的应用场景**：电商产品宣传视频。用户提供产品图+宣传模板视频，模型生成一个类似风格的新宣传视频，视频中展示的是用户提供的产品。

### 1.2 FFGO方法简介

**FFGO的核心思想**：将产品参考图和背景图拼合成一张"首帧拼贴画"（I_mix），配合特殊触发词，让模型生成一个将产品融入背景的视频。

**训练数据的作用**：模型通过训练数据学会"看到拼贴首帧 + 触发词 → 自动完成场景过渡 → 输出包含产品的连贯视频"。所以训练数据质量直接决定模型效果。

### 1.3 FFGO原版数据集是如何构建的

FFGO论文从2205个候选视频中人工精选了50个高质量样本，覆盖4类场景：

| 类别 | 占比 | 说明 |
|------|------|------|
| 人-物交互 | 60% | 人手持/使用/操作物体 |
| 元素插入 | 20% | 物体被放入场景 |
| 人-人交互 | 14% | 两人之间的交互 |
| 机器人操控 | 6% | 机械臂操控物体 |

**FFGO的数据处理流程**：
```
原始视频 → 提取首帧 → 人工标注元素名称
                          ↓
           ┌──────────────┴──────────────┐
           ↓                             ↓
    Gemini-2.5-Pro                Gemini-2.5-Pro
    提取每个前景元素               生成干净背景(移除物体)
           ↓                             ↓
        SAM 2                        干净背景图
    去白色背景→RGBA                      ↓
           ↓                             ↓
           └──────────────┬──────────────┘
                          ↓
            拼合首帧 I_mix (1280×720画布)
            左侧: 前景RGBA竖排 | 右侧: 干净背景居中
                          ↓
                   Gemini-2.5-Pro
                   生成描述性caption
                          ↓
              训练三元组: (I_mix, C_trans, V_target)
              其中 C_trans = "ad23r2 the camera view suddenly changes" + caption
```

**FFGO数据集的关键质量标准**：
- 每个视频**81帧**，统一分辨率
- 前景物体有**清晰完整的边界**，可以被干净地RGBA分割
- 背景**整洁**，移除物体后无明显瑕疵
- Caption描述**具体、视觉化**，包含产品外观、动作、场景、镜头运动
- 前景RGBA是**透明背景**（不是白底），边缘干净

---

## 2. 当前数据集的问题

数据来源：`pvtt_masked_training_datasets/`，共97个样本。

### 2.1 已有文件清单

| 文件 | 说明 |
|------|------|
| `ref_product_original_rmbg.png` | 产品抠图（⚠️ 有问题，见下） |
| `ref_product_rmbg.png` | 产品抠图另一版本（89/97有） |
| `target.mp4` | 原始完整视频 |
| `masked_input.mp4` | 产品被mask后的视频 |
| `mask_video.mp4` | 二值mask视频 |
| `reference_image.png` | 原始产品图（带背景，高清） |
| `source_frame_rmbg.png` | 视频首帧中产品的RGBA抠图 |
| `pipeline_metadata.json` | 元数据（含 `object_prompt` 产品名） |

### 2.2 存在的问题

#### 问题1: 产品RGBA抠图质量差（严重）

`ref_product_original_rmbg.png` 存在以下问题：
- **不是纯产品**：使用了bounding box mask进行提取，导致产品周围有大量杂质/背景残留
- **不是透明背景**：目前是白色底，不是RGBA透明背景
- FFGO要求前景元素是**干净的RGBA透明抠图**，当前数据不满足

**需要重新提取**（见Step 3）

#### 问题2: Mask是bounding box而非precise mask

`mask_video.mp4` 使用的是bounding box mask（矩形框），不是贴合物体轮廓的精确mask。

> 不过本次微调任务（FFGO LoRA）**不需要mask视频**——FFGO的训练数据只需要 (I_mix首帧, caption, target视频) 三元组。Mask视频是PVTT其他实验的产物，这里可以忽略。

#### 问题3: 帧数不统一

| 帧数 | 数量 | 占比 |
|------|------|------|
| 81帧 | 64 | 66.0% |
| <81帧 | 33 | 34.0% |

34%的样本不足81帧。**不足81帧的样本直接丢弃**（不做pad，原因见2.3）。

#### 问题4: 分辨率不统一且偏低

| 分辨率 | 数量 | 占比 |
|--------|------|------|
| 256×480 | 31 | 32.0% |
| 480×480 | 29 | 29.9% |
| 848×480 | 13 | 13.4% |
| 640×480 | 10 | 10.3% |
| 其他 | 14 | 14.4% |

原始视频高度均为480，但宽度从256到848不等。需要统一resize至目标分辨率。

#### 问题5: 部分reference_image不是纯产品图

例如000158（夏威夷衬衫）的reference_image包含文字"45+ Patterns"和多个花纹缩略图拼贴，不是纯产品照片。这类样本需要淘汰或特殊处理。

#### 问题6: 没有干净背景的首帧画面

#### 问题7: 训练数据集的划分和质量需进一步细化



### 2.3 关于帧数pad的说明

**不足81帧的样本直接丢弃，不做"循环最后一帧pad"。** 原因：
- pad最后一帧会导致视频末尾出现**长时间的静止画面**
- 模型在训练时会"学到"这种不自然的静止，可能导致推理时也生成停滞的结尾
- FFGO论文的数据集每个视频都是自然的81帧，没有任何pad
- 丢弃33个短视频后仍剩64个，足够筛选出50-60个高质量样本

---

## 3. 整体构建流程

```
Step 1: 人工质量筛选 (97→50-60个，仅保留≥81帧的样本)
         ↓
Step 2: 视频预处理 (帧数截取至81帧 + 分辨率统一至1280×704横屏)
         ↓
Step 3: 产品RGBA提取 (重新做，获取干净的透明背景产品图)
         ↓
Step 4: 背景提取 (从视频首帧中移除产品，获取干净背景)
         ↓
Step 5: 首帧拼合 (产品RGBA + 干净背景 → I_mix)
         ↓
Step 6: Caption生成 (VLM生成视频描述)
         ↓
Step 7: 组装最终训练数据
```

---

## 4. Step 1: 人工质量筛选

### 4.1 目标

从97个样本中精选 **50-60个** 高质量样本。

> 质量远比数量重要。FFGO论文只用了50个样本就达到了SOTA效果。宁可少10个不合格的，也不要放进去拉低整体质量。

### 4.2 前置过滤（自动化）

在人工筛选前，先自动过滤掉不合格的样本：

```bash
# 用ffprobe检查每个target.mp4的帧数，<81帧的直接标记为Fail
ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of csv=p=0 target.mp4
```

- **帧数<81**: 直接丢弃（约33个）
- 剩余约64个进入人工筛选

### 4.3 人工筛选标准

逐个浏览每个样本的 `target.mp4`（视频）和 `reference_image.png`（原始产品图），按以下标准判断pass/fail：

#### 必须Pass的条件（全部满足才通过）：

| 序号 | 检查项 | 具体标准 | 怎么检查 |
|------|--------|---------|---------|
| 1 | **视频运动流畅** | 无剧烈跳帧/闪烁、运动自然连贯 | 播放 `target.mp4` |
| 2 | **产品在视频中清晰可见** | 产品不被严重遮挡、不过小、不模糊 | 播放 `target.mp4` |
| 3 | **背景不过于杂乱** | 背景可辨识，不是纯噪声/混乱堆叠 | 播放 `target.mp4` |
| 4 | **reference_image是纯产品图** | 不是拼贴图/多图合集/带大量文字的营销图 | 查看 `reference_image.png` |

> 注：产品RGBA抠图质量**不在这里检查**——因为我们会在Step 3重新提取。

#### 优先保留的样本（加分项）：

- 人穿戴/使用产品的视频（人-产品交互）→ 多留一些
- 有明显相机运动（推拉摇移）的视频
- 产品细节丰富、辨识度高的视频
- 多产品展示的视频（`num_source_object > 1`）

#### 应该淘汰的样本（减分项）：

- 视频中产品几乎不动/变化极小（如纯静止图片转视频）
- 视频中有大量文字水印覆盖产品区域
- reference_image本身是拼贴图/多图合集（如000158的夏威夷衬衫，参考图包含文字"45+ Patterns"和多个缩略图）
- 视频画面极度模糊或压缩伪影严重

### 4.4 场景分类

筛选的同时，给每个通过的样本标注场景类别：

| 类别代号 | 类别名 | 说明 | 目标占比 |
|---------|--------|------|---------|
| **A** | 人-产品交互 | 人穿戴/佩戴/手持/使用产品 | 35-45% |
| **B** | 产品特写展示 | 产品静物/旋转/近距特写 | 30-40% |
| **C** | 产品场景融入 | 产品在生活场景中自然展示 | 15-25% |
| **D** | 多产品组合 | 多个产品同时出现 | 5-10% |

### 4.5 筛选输出格式

CSV/Excel表格：

| 序号 | 文件夹名 | 产品名(object_prompt) | 帧数 | Pass/Fail | 类别(A/B/C/D) | 备注 |
|------|---------|----------------------|------|-----------|--------------|------|
| 1 | 000004_519515046_... | Thai Cotton Harem Pants | 81 | Pass | A | 人穿戴展示 |
| 2 | 000009_561095315_... | Soft Cotton Chemo Beanie | 81 | Pass | A | 人穿戴展示 |
| 3 | 000011_616109272_... | Centipede punk patch | 81 | Fail | - | 产品太小不清晰 |
| 4 | 000058_1093127621_... | Pet Portrait Keychain | 49 | Fail(帧数) | - | <81帧，自动淘汰 |

---

## 5. Step 2: 视频预处理

### 5.1 帧数处理

| 原始帧数 | 处理方式 |
|----------|---------|
| 81帧 | 无需处理 |
| >81帧 | 截取前81帧 |
| <81帧 | **已在Step 1淘汰** |

```bash
# 截取前81帧（≥81帧的视频）
ffmpeg -i target.mp4 -vframes 81 -c:v libx264 -crf 18 -r 16 -pix_fmt yuv420p output.mp4
```

### 5.2 分辨率处理

**统一使用横屏**（与FFGO保持一致的横屏策略）。

根据社区实践调研（详见主报告7.7节），Wan2.2 TI2V-5B微调有两种常见分辨率策略：

| 策略 | 训练分辨率 | 适用场景 | 社区案例 |
|------|-----------|---------|---------|
| **策略A: 高分辨率** | **1280×704** | GPU≥40GB | FFGO论文(1344×768)，与推理分辨率一致 |
| **策略B: 低分辨率** | **832×480** | GPU 24-32GB | DiffSynth官方脚本、AMD Blog、Musubi社区均使用 |

> **社区调研发现**：DiffSynth-Studio的官方TI2V-5B训练脚本使用832×480+49帧（`--width 832 --height 480`）；AMD Blog使用832×480+81帧；Musubi-Tuner社区普遍使用480p级横屏分辨率+16fps。训练分辨率不必与推理分辨率(1280×704)完全一致，模型通过RoPE位置编码可以泛化。

**请根据实际GPU情况选择一种策略，全部数据统一到该分辨率：**

```bash
# 策略A: 高分辨率 1280×704
ffmpeg -i input.mp4 \
  -vf "scale=1280:704:force_original_aspect_ratio=decrease,pad=1280:704:(ow-iw)/2:(oh-ih)/2:black" \
  -c:v libx264 -crf 18 -r 16 -pix_fmt yuv420p output.mp4

# 策略B: 低分辨率 832×480 (横屏)
ffmpeg -i input.mp4 \
  -vf "scale=832:480:force_original_aspect_ratio=decrease,pad=832:480:(ow-iw)/2:(oh-ih)/2:black" \
  -c:v libx264 -crf 18 -r 16 -pix_fmt yuv420p output.mp4
```

> 对于竖屏/方形视频（如256×480、480×480），resize后两侧会有黑色pad条。这是可以接受的——FFGO论文也只用横屏训练，模型会学会忽略pad区域。

### 5.3 帧数与FPS规格说明

| 参数 | 值 | 依据 |
|------|------|------|
| **帧数** | **81帧** | 与FFGO一致；社区主流(49-81帧)；满足VAE约束 `(81-1)%4==0` ✓ |
| **FPS** | **16fps** | 与FFGO一致；社区标准统一为16fps，即使TI2V默认24fps |

> **关于FPS的说明**：TI2V-5B默认配置是24fps/121帧，但fps仅影响视频播放速度，不影响模型生成过程。社区所有微调案例一致使用16fps。推理时保存视频也用16fps即可。

### 5.4 视频编码要求

| 参数 | 值 |
|------|------|
| 编码器 | H.264 (libx264) |
| CRF | 18 (高质量) |
| FPS | **16** |
| 帧数 | **81** |
| 分辨率 | **1280×704** (策略A) 或 **832×480** (策略B) |
| 像素格式 | yuv420p |

---

## 6. Step 3: 产品RGBA提取

### 6.1 为什么要重新提取

当前的 `ref_product_original_rmbg.png` 存在两个问题：
1. 使用bounding box mask提取，产品周围有大量杂质
2. 是白色底而非RGBA透明背景

FFGO要求产品图是**干净的RGBA透明背景抠图**，边缘清晰无残留。

### 6.2 输入

- `reference_image.png`：原始产品图（带背景，高清）

### 6.3 方案选择

#### 方案A: Gemini 2.5 Pro（推荐，与FFGO一致）

调用Gemini 2.5 Pro的图像编辑能力提取产品：

**Prompt**：
```
Given the input image, extract the main product {PRODUCT_NAME} — return
the product alone with no resizing, compression, or background so the
output resolution exactly matches the original image.
```

然后用SAM2做精细化：去除白色背景 → 生成RGBA透明抠图。

**成本**：~$0.5（50-60次调用）

#### 方案B: GroundingDINO + SAM2（免费，效果接近）

完全开源的方案，无需API：

```
Step 1: GroundingDINO
  输入: reference_image.png + 产品名文本 (object_prompt)
  输出: 产品bounding box（文本引导的目标检测）

Step 2: SAM2 (Segment Anything Model 2)
  输入: reference_image.png + GroundingDINO输出的bbox作为prompt
  输出: 精确的像素级mask（贴合产品轮廓）

Step 3: 应用mask生成RGBA
  输入: reference_image.png + SAM2 mask
  输出: product_rgba.png（透明背景，仅保留产品区域）
```

**优势**：
- 完全免费，本地运行
- SAM2的分割精度非常高，边缘质量通常优于生成式方法
- GroundingDINO支持文本引导，可以用 `object_prompt` 直接定位产品

**硬件需求**：GroundingDINO ~2GB VRAM，SAM2 ~4GB VRAM，普通GPU即可

#### 方案C: 仅SAM2（如果产品在图中位置明显）

如果产品在reference_image中居中且显眼，可以跳过GroundingDINO，直接用SAM2的自动分割模式（automatic mask generation）提取最大的前景物体。

### 6.4 输出要求

每个样本一张 `product_rgba.png`：
- **格式**：PNG，RGBA 4通道
- **背景**：完全透明（alpha=0），不是白色
- **产品**：边缘清晰，无背景残留，产品完整无缺失
- **分辨率**：保持reference_image原始分辨率（高清），后续拼合时再缩放

### 6.5 质量检查

提取完成后，人工抽查10个样本：
- 用图片查看器打开product_rgba.png，确认背景是透明的（棋盘格）
- 放大检查边缘：是否有白色/彩色锯齿残留
- 确认产品完整：没有被切掉重要部分

---

## 7. Step 4: 背景提取

### 7.1 目标

为每个样本获取一张**干净的背景图**（产品被移除后的场景）。

### 7.2 方案选择

#### 方案A: Gemini 2.5 Pro（推荐，与FFGO一致）

FFGO论文使用Gemini 2.5 Pro做Object Removal，效果最好。

**输入**：`target.mp4` 的首帧（resize后的1280×704版本）

**Prompt**：
```
Given the input image of an e-commerce product scene, remove the product
{PRODUCT_NAME} entirely. Return the edited image only — it must preserve
the source resolution (no scaling or compression) and contain neither
the product nor any artifacts of its removal. The background should look
natural and complete.
```

**成本**：~$0.5（50-60次调用）

#### 方案B: Stable Diffusion Inpainting（免费，效果较好）

使用SD Inpainting模型在mask区域重建背景：

```
Step 1: 从target.mp4提取首帧（已resize到1280×704）

Step 2: 生成产品mask
  用Step 3中SAM2产生的mask（或GroundingDINO检测到的区域），
  投射到首帧上获取需要移除的区域

Step 3: SD Inpainting
  模型: stabilityai/stable-diffusion-2-inpainting 或 runwayml/stable-diffusion-inpainting
  输入: 首帧 + 产品mask
  Prompt: "clean background, no objects" 或根据场景描述
  输出: 产品区域被自然背景填充的完整图像
```

**优势**：免费，效果显著好于简单的ffmpeg提取或LaMa
**硬件需求**：~6-8GB VRAM

#### 方案C: ProPainter（免费，视频修复专用）

如果SD Inpainting效果仍不够好，可以用ProPainter（视频专用修复模型）：
- 利用视频中其他帧的信息来填充被mask的区域
- 通常效果优于单帧修复，因为可以"看到"产品移走后露出的背景

**注意**：ProPainter需要mask视频，而当前的mask_video.mp4是bounding box mask。如果用此方案，需要先用SAM2生成精确的首帧mask。

### 7.3 不推荐的方案

| 方案 | 为什么不推荐 |
|------|-------------|
| 直接提取masked_input.mp4首帧 | mask区域是黑色/纯色填充，不是自然背景，会导致I_mix中出现不自然的色块 |
| LaMa Inpainting | 对于大面积移除效果差，容易产生模糊/重复纹理的瑕疵 |

### 7.4 输出要求

每个样本一张 `background.png`：
- **格式**：PNG，RGB 3通道
- **分辨率**：与视频一致（策略A: 1280×704，策略B: 832×480）
- **质量**：产品区域被自然背景填充，无明显瑕疵/接缝

---

## 8. Step 5: 首帧拼合（I_mix）

### 8.1 这是什么？

首帧拼合是FFGO方法的核心。将产品RGBA抠图和干净背景拼合成一张图片，作为视频生成的"条件首帧"。模型会从这个拼贴首帧出发，生成一个将产品融入背景的连贯视频。

### 8.2 拼合规则

画布大小与视频分辨率一致（策略A: 1280×704，策略B: 832×480）。

以策略A (1280×704) 为例：

```
┌──────────────┬──────────────────────────────┐
│              │                              │
│  产品RGBA    │       干净背景               │
│  (居中)      │       (居中)                 │
│              │                              │
│  ← ~420px → │  ←      ~860px           →   │
│              │                              │
└──────────────┴──────────────────────────────┘
  约1/3宽度          约2/3宽度
```

策略B (832×480) 同理，按比例缩小。

### 8.3 详细拼合步骤

```python
# 伪代码 - 以横屏为例
# W, H = (1280, 704) 策略A，或 (832, 480) 策略B
W, H = target_resolution  # 与视频分辨率一致
canvas = new_image(W, H, color=(0,0,0))  # 黑色画布

# 左侧: 产品区域 (约1/3宽度)
prod_zone_w = int(W * 0.33)
prod_zone_h = H

# 将产品RGBA图等比缩放适配左侧区域（留20px边距）
product_img = load("product_rgba.png")  # RGBA透明背景
product_scaled = fit_in_box(product_img,
                            max_w=prod_zone_w - 20,
                            max_h=prod_zone_h - 20)
# 居中粘贴到左侧区域（用alpha通道做透明合成）
paste_centered(canvas, product_scaled,
               x_center=prod_zone_w // 2,
               y_center=704 // 2,
               use_alpha=True)

# 右侧: 背景区域 (约2/3宽度)
bg_zone_x = prod_zone_w
bg_zone_w = 1280 - prod_zone_w  # ≈ 860px

background = load("background.png")  # RGB
bg_scaled = fit_in_box(background,
                       max_w=bg_zone_w - 10,
                       max_h=704 - 10)
paste_centered(canvas, bg_scaled,
               x_center=bg_zone_x + bg_zone_w // 2,
               y_center=704 // 2)

save(canvas, "first_frame.png")  # RGB格式
```

### 8.4 多产品样本的拼合

对于 `num_source_object > 1` 的样本（约10个），左侧区域需要竖排多个产品：

```
┌──────────────┬──────────────────────────────┐
│  产品1 RGBA  │                              │
│  ─────────── │       干净背景               │
│  产品2 RGBA  │       (居中)                 │
│  ─────────── │                              │
│  产品3 RGBA  │                              │
└──────────────┴──────────────────────────────┘
```

每个产品平均分配左侧区域的高度，自动缩放适配。

### 8.5 关键注意事项

1. **产品图必须是RGBA透明背景**，粘贴时用alpha通道做mask，这样透明区域会显示黑色画布
2. **产品大小适中**：不要太小（模型看不清细节）也不要撑满整个区域（留一些边距）
3. **背景要居中**：不要拉伸变形，等比缩放后居中放置
4. **最终输出为RGB PNG**（透明区域显示为黑色）

---

## 9. Step 6: Caption生成

### 9.1 这是什么？

为每个训练样本生成一段**描述性文字**，描述视频中发生了什么。这段文字将作为训练时的文本条件。

### 9.2 方案选择

#### 方案A: Gemini 2.5 Pro（推荐，与FFGO一致，成本~$0.5-1）

对每个样本，调用Gemini 2.5 Pro API：

**输入**：
- `product_rgba.png`（产品RGBA抠图）
- `background.png`（干净背景）
- `target.mp4`（完整视频）

**Prompt**：
```
You are given an e-commerce product demonstration video and several
images showing the product. Generate a descriptive caption for the video
that prominently features the product shown in the images.

Requirements:
1. Wrap your final text in <caption>...</caption> tags.
2. The caption MUST describe:
   - The product's visual appearance (color, material, shape, details)
   - How the product is being showcased in the video (being held,
     worn, displayed on a surface, rotated, etc.)
   - The setting/background of the video
   - Any camera movement (zoom in, pan, static, etc.)
3. DO NOT include:
   - Marketing language ("perfect gift", "must-have", "best seller")
   - Prices or product specifications
   - Vague narrative filler ("The scene unfolds with elegance...")
4. Keep the caption factual, visual, and descriptive.

Examples of Good Captions:
1. Film quality, professional quality, rich details. A pair of hammered
   silver hoop earrings is displayed on a white marble surface. The
   camera slowly zooms in to show the textured surface of the earrings,
   capturing the light reflecting off the hammered finish. A hand
   gently picks up one earring and holds it at an angle.
2. A woman wearing rust-colored cotton harem pants and a white crop
   top poses on a rocky cliff edge with mountains in the background.
   She shifts her weight from one leg to the other, showing the loose,
   flowing fit of the pants. The camera captures her from a low angle.
```

**输出**：提取 `<caption>` 标签内的文字。

#### 方案B: 免费方案

**选项B1（推荐免费方案）**: 开源VLM本地推理
- 模型：Qwen2.5-VL-7B-Instruct（需16GB+ GPU）
- 输入：产品RGBA + 从target.mp4均匀抽取的6-8帧
- Prompt同上（去掉 `<caption>` 标签要求）

**选项B2**: 免费API
- 硅基流动 (SiliconFlow) 的Qwen2.5-VL系列有免费额度
- 50-60个样本在免费额度范围内

**选项B3（最快，质量最低）**: 模板化生成

```
类别A (人-产品交互):
"Film quality, professional quality, rich details. A person is
showcasing {product_name}. The person is wearing or holding the
{product_name}, demonstrating its appearance and fit. The camera
captures the product from multiple angles as the person moves naturally."

类别B (产品特写展示):
"Film quality, professional quality, rich details. {product_name} is
displayed prominently in the scene. The product is shown in detail,
highlighting its design, texture, and craftsmanship. The camera slowly
captures the product from different angles."

类别C (产品场景融入):
"Film quality, professional quality, rich details. {product_name}
is placed naturally in the scene, blending with the surrounding
environment. The camera captures the product in context, showing how
it fits into the setting."

类别D (多产品组合):
"Film quality, professional quality, rich details. Multiple
{product_name} items are displayed together, showcasing the collection.
Each piece is visible with its unique details and design elements."
```

### 9.3 Caption质量检查

生成后请人工抽查5-10个caption，确认：
- 描述是否准确反映了视频内容？
- 是否包含了产品的具体外观描述？
- 是否有幻觉（描述了视频中不存在的东西）？
- 英文语法是否通顺？

---

## 10. Step 7: 组装最终训练数据

### 10.1 文本提示拼接

对每个样本，将caption加上固定前缀：

```
C_trans = "ad23r2 the camera view suddenly changes " + caption
```

> `ad23r2` 是一个无意义触发词，LoRA训练时会将它与"首帧融合→场景过渡"的行为绑定。
> **这个前缀在所有样本中保持完全一致，一个字符都不能改。**

### 10.2 最终输出目录结构

```
ffgo_training_data/
├── sample_000/
│   ├── first_frame.png        # I_mix 拼合首帧 (与视频同分辨率, RGB)
│   ├── video.mp4              # target视频 (81帧, 16fps, 策略A:1280×704 或 策略B:832×480)
│   ├── caption.txt            # "ad23r2 the camera view suddenly changes ..."
│   └── metadata.json          # 元信息
├── sample_001/
│   ├── ...
├── ...
└── sample_049/  (或到sample_059)
    ├── ...
```

### 10.3 metadata.json 格式

```json
{
    "sample_id": "000004",
    "original_folder": "000004_519515046_pikalda-thai-cotton-harem-pants-womens_video_shot001_part01_clip001",
    "product_name": "Thai Cotton Harem Pants",
    "scene_category": "A",
    "resolution": [1280, 704],
    "num_frames": 81,
    "fps": 16,
    "num_objects": 1,
    "caption_method": "gemini-2.5-pro",
    "product_extraction_method": "grounding-dino+sam2",
    "background_extraction_method": "gemini-2.5-pro"
}
```

---

## 11. 质量检查清单（最终验收）

### 11.1 整体检查

- [ ] 总样本数在50-60之间
- [ ] 场景类别分布合理（A类≥30%，无类别为0%）
- [ ] 全部视频均为81帧、16fps、统一分辨率（策略A:1280×704 或 策略B:832×480）
- [ ] 全部caption以 `"ad23r2 the camera view suddenly changes "` 开头

### 11.2 逐样本抽查（随机抽10个）

- [ ] `first_frame.png` 左侧有产品（透明区域显示为黑色），右侧有干净背景
- [ ] `first_frame.png` 分辨率 = video.mp4 分辨率
- [ ] `first_frame.png` 中产品边缘干净，无白边/杂色残留
- [ ] `first_frame.png` 中背景自然完整，无产品残影/黑色色块
- [ ] `video.mp4` 播放流畅，产品清晰可见
- [ ] `caption.txt` 描述与视频内容一致
- [ ] `metadata.json` 字段完整

### 11.3 常见问题排查

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| I_mix中产品有白色边缘 | RGBA抠图不干净/用了白底图 | 重新用SAM2提取，确保RGBA透明 |
| I_mix中背景有黑色色块 | 背景提取时mask区域没被正确填充 | 换用Gemini或SD Inpainting重做背景 |
| I_mix中产品看不清 | 产品缩放太小 | 增大左侧区域比例或减小padding |
| 视频有绿色/紫色条纹 | 编码问题 | 确认用libx264 + yuv420p |
| 帧数不对 | 截取失败 | 用ffprobe确认：`ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of csv=p=0 video.mp4` |

---

## 附录A: 方案成本对比

| 环节 | 方案A (Gemini) | 方案B (免费) |
|------|---------------|-------------|
| 产品RGBA提取 | Gemini + SAM2 (~$0.5) | GroundingDINO + SAM2 (免费) |
| 背景提取 | Gemini (~$0.5) | SD Inpainting (免费，需GPU) |
| Caption生成 | Gemini (~$0.5) | Qwen2.5-VL-7B 或 模板化 (免费) |
| **总计** | **~$1.5** | **免费** |

---

*文档创建日期: 2026-03-19*
*文档更新日期: 2026-03-19*
