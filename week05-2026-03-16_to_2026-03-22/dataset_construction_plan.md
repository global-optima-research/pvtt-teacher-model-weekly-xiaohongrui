# PVTT FFGO训练数据集构建计划

> **目标**: 构建用于 Wan2.2 TI2V 5B + FFGO LoRA 微调的训练数据集
> **数据量**: **150个**高质量样本
> **分辨率**: **832×480**，横屏，不足宽度用黑色padding
> **交付日期**: [待定]

---

## 1. 背景简介

### 1.1 我们要做什么

我们要在 Wan2.2 TI2V 5B 上微调一个LoRA，复现FFGO论文（First Frame Is the Place to Go for Video Content Customization）的工作。

**应用场景**：电商产品宣传视频。用户提供产品图+宣传模板视频，模型生成一个类似风格的新宣传视频，视频中展示的是用户提供的产品。

### 1.2 FFGO方法简介

**核心思想**：将产品参考图和背景图拼合成一张"首帧拼贴画"（I_mix），配合特殊触发词，让模型生成一个将产品融入背景的视频。

**训练数据三元组**: `(I_mix首帧, caption文本, target视频)`

### 1.3 FFGO原版数据集构建流程

```
原始视频 → 提取首帧 → 人工标注元素名称
                          ↓
           ┌──────────────┴──────────────┐
           ↓                             ↓
    VLM (Gemini-2.5-Pro)          VLM (Gemini-2.5-Pro)
    提取每个前景元素               生成干净背景(移除物体)
           ↓                             ↓
        SAM 2                        干净背景图
    去白色背景→RGBA                      ↓
           ↓                             ↓
           └──────────────┬──────────────┘
                          ↓
            拼合首帧 I_mix (白色画布)
            左侧: 前景RGBA竖排 | 右侧: 干净背景居中
                          ↓
                   Gemini-2.5-Pro 生成描述性caption
                          ↓
              训练三元组: (I_mix, C_trans, V_target)
              其中 C_trans = "ad23r2 the camera view suddenly changes" + caption
```

---

## 2. 我们已做过的尝试与当前卡点

### 2.1 产品RGBA提取 — 已解决

**方案：GroundingDINO + SAM2**

```
GroundingDINO (文本引导检测产品bbox) → SAM2 (精确像素级分割) → RGBA透明背景抠图
```

**效果**：整体效果不错，大部分产品能被干净地提取。少量样本存在问题（如穿戴类产品提取了整个模特而非衣服），可通过人工筛查排除。

### 2.2 背景提取（Object Removal）— ⚠️ 当前最主要的卡点

需要从视频首帧中移除产品，生成"干净背景"用于首帧拼合。我们尝试了多种方案，**效果均不理想**：

| 已尝试方案 | 结果 | 问题 |
|-----------|------|------|
| **LaMa** | ❌ 效果差 | 大面积移除时产生严重模糊/重复纹理伪影 |
| **SmartEraser** (CVPR 2025) | ❌ 效果差 | 修复区域不自然，色差明显，且所有图片被resize到512×512正方形导致质量损失 |
| **ProPainter** (视频修复) | ❌ 效果差 | 利用多帧信息修复，但产品占画面比例大时仍有明显残影 |
。

---

## 3. 最终交付物规格

### 3.1 每个训练样本包含的文件

| 文件名 | 说明 | 格式 | 分辨率 |
|--------|------|------|--------|
| `first_frame.png` | FFGO拼合首帧（左侧产品+右侧背景，**白色画布**） | RGB PNG | 832×480 |
| `video.mp4` | 训练目标视频 | H.264, 16fps, yuv420p | 832×480 |
| `caption.txt` | 带触发词的描述文本 | UTF-8文本 | - |
| `product_rgba.png` | 产品干净RGBA透明背景抠图（用precise mask抠出） | RGBA PNG | 有效内容区域 |
| `product_mask.png` | 产品precise mask（像素级精确轮廓，非bounding box） | 灰度(L) PNG | 与product_rgba同尺寸 |
| `first_frame_raw.png` | 未处理过的原视频首帧图（不含padding） | RGB PNG | 有效内容区域 |
| `background.png` | 处理后的原视频首帧（做了object remove，可用作背景，不含padding） | RGB PNG | 有效内容区域 |
| `metadata.json` | 元信息（产品名、类别、方法等） | JSON | - |

> **关于 product_mask.png**：这是用 SAM2 生成的**precise mask**（像素级贴合产品轮廓），不是 bounding box 矩形框。白色(255)=产品区域，黑色(0)=背景区域。该 mask 同时用于：(1) 生成 product_rgba.png 的透明背景；(2) 指导背景提取时的 object removal 区域。

### 3.2 视频规格

| 参数 | 值 |
|------|------|
| **分辨率** | **832×480** (横屏) |
| **帧数** | **81帧** |
| **FPS** | **16fps** |
| **编码** | H.264 (libx264), CRF 18, yuv420p |

对于宽度不足832的竖屏/窄屏视频，等比缩放后用**黑色padding**补齐到832×480。

> **⚠️ 关于padding区域的重要说明**
>
> 视频（video.mp4）经过padding后两侧会有黑色边条。但以下文件**不能包含黑色padding区域**：
>
> | 文件 | 要求 |
> |------|------|
> | `first_frame_raw.png` | 只裁剪视频画面的**有效内容区域**，不带黑色padding |
> | `background.png` | 只包含**有效内容区域**的背景，不带黑色padding |
> | `product_rgba.png` | 只包含**有效内容区域**的产品抠图，不带黑色padding |
> | `first_frame.png` | 右侧放置的背景应使用**去除padding后的干净背景**，不能把黑条也贴进去 |
>
> **原因**：黑色padding是视频为了凑分辨率加的无意义区域。如果首帧拼合图和背景图中带上黑条，模型会把黑条当作"场景内容"来学习，污染训练效果。


### 3.3 首帧拼合规格（first_frame.png）

```
白色画布 832×480:
┌──────────────┬──────────────────────────────┐
│              │                              │
│  产品RGBA    │       干净背景               │
│  (居中)      │       (居中)                 │
│              │                              │
│  ← ~275px → │  ←      ~557px           →   │
│              │                              │
└──────────────┴──────────────────────────────┘
  约1/3宽度          约2/3宽度
  白色底上的产品      去除padding后的background居中
```

- **画布背景色：白色 (255, 255, 255)**
- 左侧1/3区域：产品RGBA居中放置，透明区域显示为白色
- 右侧2/3区域：**去除padding后的** background.png 等比缩放居中放置
- **右侧背景不能带有黑色padding条**

### 3.4 Caption格式

所有caption必须以固定触发词前缀开头：

```
ad23r2 the camera view suddenly changes [描述内容]
```

> `ad23r2` 是无意义触发词，LoRA训练时将其与"首帧融合→场景过渡"行为绑定。**一个字符都不能改。**

### 3.5 metadata.json 格式

```json
{
    "sample_id": "sample_000",
    "product_name": "Thai Cotton Harem Pants",
    "scene_category": "A",
    "num_objects": 1,

    "video": {
        "file": "video.mp4",
        "resolution": [832, 480],
        "num_frames": 81,
        "fps": 16,
        "original_resolution": [352, 480],
        "has_padding": true
    },

    "caption": {
        "file": "caption.txt",
        "full_text": "ad23r2 the camera view suddenly changes A woman models a pair of orange Thai Cotton Harem Pants, showcasing their loose fit and wide leg, against a scenic desert backdrop. The camera remains static, focusing on the pants as the woman shifts poses.",
        "trigger_prefix": "ad23r2 the camera view suddenly changes",
        "description_only": "A woman models a pair of orange Thai Cotton Harem Pants, showcasing their loose fit and wide leg, against a scenic desert backdrop. The camera remains static, focusing on the pants as the woman shifts poses.",
        "generation_method": "gemini-2.5-pro"
    },

    "product_rgba": {
        "file": "product_rgba.png",
        "extraction_method": "grounding-dino+sam2",
        "detection_prompt": "orange Thai Cotton Harem Pants",
        "detection_confidence": 0.72,
        "product_area_ratio": 0.18
    },

    "product_mask": {
        "file": "product_mask.png",
        "mask_type": "precise (SAM2)",
        "format": "grayscale, 255=product, 0=background"
    },

    "background": {
        "file": "background.png",
        "extraction_method": "[待定]"
    },

    "first_frame_raw": {
        "file": "first_frame_raw.png",
        "note": "原视频首帧，已裁剪padding"
    },

    "first_frame": {
        "file": "first_frame.png",
        "canvas_size": [832, 480],
        "canvas_color": [255, 255, 255],
        "product_zone_ratio": 0.33,
        "layout": "left=product_rgba, right=background"
    },

    "files": [
        "first_frame.png",
        "video.mp4",
        "caption.txt",
        "product_rgba.png",
        "product_mask.png",
        "first_frame_raw.png",
        "background.png",
        "metadata.json"
    ]
}
```

### 3.6 目录结构

```
dataset/
├── sample_000/
│   ├── first_frame.png
│   ├── video.mp4
│   ├── caption.txt
│   ├── product_rgba.png
│   ├── product_mask.png
│   ├── first_frame_raw.png
│   ├── background.png
│   └── metadata.json
├── sample_001/
│   └── ...
├── ...
└── sample_149/
    └── ...
```

---

## 4. 数据集构建流程

```
Step 1: 收集候选视频 (需要≥200个候选，最终精选150个)
         ↓
Step 2: 视频预处理 (帧数统一81帧 + 分辨率统一832×480)
         ↓
Step 3: 产品RGBA提取 (GroundingDINO + SAM2)
         ↓
Step 4: 背景提取 (⚠️ 当前卡点，方案待定)
         ↓
Step 5: 首帧拼合 (产品RGBA + 干净背景 → 白色画布I_mix)
         ↓
Step 6: Caption生成
         ↓
Step 7: 人工质检 + 组装最终数据集
```

---

## 5. Step 1: 收集与筛选候选视频

### 5.1 数量目标

需要 **≥200个** 候选视频，经筛选后精选 **150个** 进入最终数据集。

### 5.2 筛选标准

| 序号 | 检查项 | 标准 |
|------|--------|------|
| 1 | **帧数** | ≥81帧（不足的直接丢弃，不做pad） |
| 2 | **视频运动** | 运动流畅，无剧烈跳帧/闪烁 |
| 3 | **产品可见性** | 产品在视频中清晰可见，不被严重遮挡 |
| 4 | **背景整洁** | 背景可辨识，不过于杂乱 |
| 5 | **画面质量** | 不严重模糊，无大面积水印/文字遮挡 |

### 5.3 场景分类

| 类别 | 说明 | 目标占比 |
|------|------|---------|
| **A. 人-产品交互** | 人穿戴/佩戴/手持/使用产品 | 35-45% |
| **B. 产品特写展示** | 产品静物/旋转/近距特写 | 30-40% |
| **C. 产品场景融入** | 产品在生活场景中自然展示 | 15-25% |
| **D. 多产品组合** | 多个产品同时出现 | 5-10% |

---

## 6. Step 2: 视频预处理

```bash
# 截取前81帧 + 统一分辨率832×480 + 16fps
ffmpeg -i input.mp4 -vframes 81 \
  -vf "scale=832:480:force_original_aspect_ratio=decrease,pad=832:480:(ow-iw)/2:(oh-ih)/2:black" \
  -c:v libx264 -crf 18 -r 16 -pix_fmt yuv420p output.mp4
```

同时提取未处理的首帧原图：
```bash
# 从预处理后的视频提取首帧
ffmpeg -i output.mp4 -vframes 1 first_frame_raw.png
```

---

## 7. Step 3: 产品RGBA提取

**方案：GroundingDINO + SAM2**（已验证，效果不错）

```
输入: first_frame_raw.png + 产品名文本prompt
  ↓
GroundingDINO: 文本引导检测产品bounding box
  ↓
SAM2: 根据bbox做精确像素级分割 → 产品mask
  ↓
应用mask生成RGBA: 产品区域保留，背景设为透明
  ↓
输出: product_rgba.png (RGBA, 832×480, 透明背景)
```

---

## 8. Step 4: 背景提取（⚠️ 卡点）

**目标**：从 `first_frame_raw.png` 中移除产品，生成 `background.png`。

**已尝试但效果不佳的方案**：LaMa、SmartEraser、ProPainter（详见第2节）。

**输出要求**：`background.png`，RGB PNG，832×480，产品区域被自然背景填充。

**参考：FFGO原论文的Object Removal提示词（用于Gemini 2.5 Pro）**：

```
Prompt – Given the input image, remove the subset {IDENTIFIED OBJECTS}
entirely. Return the edited image only — it must preserve the source
resolution (no scaling or compression) and contain neither the specified
objects nor any artifacts of their removal.
```

电商场景适配版：
```
Given the input image of an e-commerce product scene, remove the product
{PRODUCT_NAME} entirely. Return the edited image only — it must preserve
the source resolution (no scaling or compression) and contain neither
the product nor any artifacts of its removal. The background should look
natural and complete.
```

---

## 9. Step 5: 首帧拼合

```python
# 白色画布 832×480
canvas = Image.new("RGB", (832, 480), (255, 255, 255))

# 左侧1/3: 产品RGBA
prod_zone_w = int(832 * 0.33)  # ≈ 275px
product = Image.open("product_rgba.png").convert("RGBA")
# 裁剪到非透明bounding box → 等比缩放适配左侧区域 → 居中粘贴
# 透明区域在白色画布上显示为白色

# 右侧2/3: 干净背景
background = Image.open("background.png").convert("RGB")
# 等比缩放适配右侧区域 → 居中粘贴

canvas.save("first_frame.png")
```

---

## 10. Step 6: Caption生成

每个样本需要一段描述视频内容的英文caption，包含：
- 产品外观（颜色、材质、形状）
- 展示方式（手持、穿戴、旋转、静物）
- 场景背景
- 镜头运动

最终caption格式：`"ad23r2 the camera view suddenly changes " + 描述内容`

**参考：FFGO原论文的Caption生成提示词（用于Gemini 2.5 Pro）**：

```
Task Description:
You are given a video and several images. Generate a descriptive caption
for the video that prominently features the components shown in the
images. Wrap your final text in <caption>...</caption> tags. The caption
must highlight the significance and role of these components throughout
the video, while omitting filler such as "The scene unfolds with a
whimsical and heartwarming narrative, emphasizing the simple joys of
life through the Teddy Bear's endearing actions".

Examples of Descriptive Captions:
1. Film quality, professional quality, rich details. The video begins
   to show the surface of a pond, and the camera slowly zooms in to a
   close-up. The water surface begins to bubble, and then a blonde
   woman is seen coming out of the lotus pond soaked all over, showing
   the subtle changes in her facial expression.
2. A professional male diver performs an elegant diving maneuver from
   a high platform. Full-body side view captures him wearing bright
   red swim trunks in an upside-down posture with arms fully extended
   and legs straight and pressed together. The camera pans downward
   as he dives into the water below.
```

电商场景适配版：
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
```


---

## 11. 质量检查清单

### 整体检查

- [ ] 总样本数 = **150**
- [ ] 全部视频：81帧、16fps、832×480
- [ ] 全部caption以 `"ad23r2 the camera view suddenly changes "` 开头
- [ ] 场景分类分布合理

### 逐样本抽查（随机抽15个）

- [ ] `first_frame.png`：白色画布，左侧有产品，右侧有干净背景，832×480
- [ ] `product_rgba.png`：产品干净RGBA，透明背景（棋盘格），边缘清晰
- [ ] `background.png`：产品被移除，修复区域自然无瑕疵
- [ ] `first_frame_raw.png`：未处理的原始首帧，832×480
- [ ] `video.mp4`：播放流畅，产品清晰可见
- [ ] `caption.txt`：描述准确，无幻觉
- [ ] `metadata.json`：字段完整

---

*文档创建日期: 2026-03-19*
*文档更新日期: 2026-03-25*
