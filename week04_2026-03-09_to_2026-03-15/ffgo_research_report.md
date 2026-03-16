# FFGo: First Frame Is the Place to Go for Video Content Customization

## 详细调研报告

> **论文信息**
> - **标题**: First Frame Is the Place to Go for Video Content Customization
> - **作者**: Jingxi Chen, Zongxia Li, Zhichao Liu, Guangyao Shi, Xiyang Wu, Fuxiao Liu, Cornelia Fermüller, Brandon Y. Feng, Yiannis Aloimonos
> - **机构**: University of Maryland, University of Southern California, Massachusetts Institute of Technology
> - **ArXiv**: 2511.15700v1 (2025年11月19日)
> - **主页**: http://firstframego.github.io

---

## 目录

1. [论文完整分析](#1-论文完整分析)
   - [1.1 背景与动机](#11-背景与动机)
   - [1.2 核心发现与设计理念](#12-核心发现与设计理念)
   - [1.3 方法架构](#13-方法架构)
   - [1.4 训练流程](#14-训练流程)
   - [1.5 推理流程](#15-推理流程)
   - [1.6 数学公式](#16-数学公式)
   - [1.7 实验结果](#17-实验结果)
   - [1.8 贡献与局限](#18-贡献与局限)
2. [理解验证与纠正](#2-理解验证与纠正)
3. [问题解答](#3-问题解答)
   - [3.1 Prompt 设计详解](#31-prompt-设计详解)
   - [3.2 训练数据集制作流程分析](#32-训练数据集制作流程分析)
   - [3.3 Fc=4 含义详解](#33-fc4-含义详解)
   - [3.4 VLM 发展历程与 Gemini 替代方案](#34-vlm-发展历程与-gemini-替代方案)
4. [PVTT 应用分析](#4-pvtt-应用分析)
   - [4.1 直接使用 FFGo（base model + 预训练 LoRA）](#41-直接使用-ffgobase-model--预训练-lora)
   - [4.2 方案 A：VACE I2V + FFGo 首帧参考](#42-方案-avace-i2v--ffgo-首帧参考)
   - [4.3 方案 B：VACE mask video + FFGo 首帧产品参考](#43-方案-bvace-mask-video--ffgo-首帧产品参考)
   - [4.4 总体可行性分析与建议方向](#44-总体可行性分析与建议方向)

---

## 1. 论文完整分析

### 1.1 背景与动机

**问题背景**：视频内容定制 (Video Content Customization) 是指根据用户提供的视觉参考（reference images）生成包含这些元素的视频。这在影视制作、产品展示、自动驾驶模拟、机器人操控等场景中有广泛需求。

**现有方法的两大路线及其缺陷**：

1. **架构修改路线**（如 VACE [13], SkyReels-A2 [7]）：修改预训练模型架构以接受额外的参考图像输入。缺点：破坏模型效率和兼容性，且架构限制了参考图像数量（如 VACE/SkyReels-A2 最多支持 3 个参考）。
2. **大规模微调路线**（如 DreamVideo [33], Dreamactor-h1 [31]）：在特定场景的大规模数据集上微调。缺点：导致**性能退化和泛化能力丧失**——微调数据的多样性远低于预训练数据，导致模型"遗忘"预训练知识，从通用模型退化为专用模型。

**核心问题**：*能否在不修改模型架构、不依赖大规模训练数据的前提下，将多个参考图像的内容融入预训练视频生成模型？*

### 1.2 核心发现与设计理念

**关键洞察：首帧是概念记忆缓冲区 (Conceptual Memory Buffer)**

FFGo 发现了一个此前被忽视的现象：预训练的 I2V 模型天然具备从首帧中"读取"多个视觉概念并在后续帧中融合它们的能力。也就是说，**首帧不仅是视频的时空起点，更是一个存储视觉实体供后续复用的概念记忆缓冲区**。

如 Figure 2 所示，当在 Veo 3、Sora 2、Wan 2.2 等模型上使用拼贴式首帧 + 适当的过渡短语 (transition phrase) 时，模型能够自发地进行主体融合和场景过渡。但这种"原始"能力存在三个问题：
1. 过渡短语的 prompt engineering 非常困难，耗时且依赖模型/视频
2. 场景过渡不稳定
3. 物体身份经常丢失

**FFGo 的解决方案**：通过仅 20-50 个训练样本的少样本 LoRA 微调，**可靠地激活**这种内在能力，使其变得稳定、可控。

### 1.3 方法架构

FFGo 的流水线由三个组件构成（见 Figure 3）：

#### 组件 1：数据集策划 (Dataset Curation)

利用统一视觉语言模型 (Unified VLM，如 Gemini-2.5-Pro) 从视频中提取高质量的配对训练数据。

**流程**：
```
原始视频 → 提取首帧 → 人工标注元素名称 (如 cake, Christmas hat, man)
                              ↓
              ┌───────────────┴───────────────┐
              ↓                               ↓
     Gemini-2.5-Pro                   Gemini-2.5-Pro
     (Object Extraction)              (Object Removal)
     提取每个前景元素                  生成干净背景
              ↓                               ↓
         SAM 2                          干净背景图
     去除白色背景                            ↓
     保留 RGBA 图层                          ↓
              ↓                               ↓
              └───────────────┬───────────────┘
                              ↓
                    拼合首帧 I_mix (1280×720)
                    左半部分：前景 RGBA 竖排
                    右半部分：干净背景居中
                              ↓
                    Gemini-2.5-Pro
                    (Caption Generation)
                    生成描述 caption C
                              ↓
                    训练对：(I_mix, C_trans, V_mix)
```

#### 组件 2：少样本 LoRA 适配 (Few-shot LoRA Adaptation)

- **基础模型**：Wan2.2-I2V-A14B（14B 参数的 MoE 图像到视频模型）
- **LoRA 配置**：rank = 128
- **过渡短语**：`C_trans = "<transition> + C"`，其中 `<transition>` = `"ad23r2 the camera view suddenly changes"`
- **训练数据**：仅 50 个视频样本
- **训练目标**：让模型学会在首帧和过渡场景之间进行主体融合

#### 组件 3：干净定制视频推理 (Clean Customized Video Inference)

推理时，模型生成包含 F 帧的视频 V_mix = {F_c, F_g}：
- **F_c**（前 4 帧）：时间压缩帧，包含主体融合过程，直接丢弃
- **F_g**（后 77 帧）：干净的定制视频内容

用户只需提供参考图像 + 文本描述即可获得高质量定制视频。

### 1.4 训练流程

**训练数据详情**：

| 项目 | 详情 |
|------|------|
| **数据来源** | HOIGen-1M (~2000 clips), Veo 3 演示视频 (5个), 授权短视频 → 共 2205 候选 |
| **筛选标准** | 前景清晰可分割、物体边界完整、有明确交互 |
| **最终数量** | 50 个高质量视频 |
| **每个视频** | 裁剪至 81 帧 (6-20秒) |
| **数据分布** | 人-物交互 60%, 人-人交互 14%, 元素插入 20%, 机器人操控 6% |
| **标注方式** | 人工标注每个视频中需要分割的元素名称 |

**训练配置**：

| 参数 | 值 |
|------|------|
| **基础模型** | Wan2.2-I2V-A14B (MoE, 两个去噪 Transformer) |
| **LoRA rank** | 128 |
| **学习率** | 1 × 10⁻⁴ |
| **优化器** | AdamW |
| **权重衰减** | 3 × 10⁻² |
| **epsilon** | 1 × 10⁻¹⁰ |
| **训练分辨率** | 1344 × 768 |
| **帧数** | 81 |
| **batch size** | 4 |
| **GPU** | 2 × NVIDIA H200 |
| **训练时间** | 每个 Transformer 独立训练 5 小时 (共约 10 小时) |

**关键设计**：Wan2.2-I2V-A14B 采用 MoE 架构，有两个独立的去噪 Transformer（分别处理 low-noise 和 high-noise regime），FFGo 对两者分别训练 LoRA。

### 1.5 推理流程

```
输入：
  - 参考图像集合 {O_1, O_2, ..., O_n} (RGBA 前景) + 背景图像 B
  - 文本提示 C (描述视频内容)

Step 1: 拼合首帧 I_mix
  ┌──────────────────────────────────┐
  │  O_1  │                          │
  │  O_2  │     B (背景居中)          │
  │  O_3  │                          │
  │ (竖排) │                          │
  └──────────────────────────────────┘
  画布大小：1280 × 720

Step 2: 构造过渡提示
  C_trans = "ad23r2 the camera view suddenly changes" + C

Step 3: I2V 生成
  V_mix = g_{θ+Δθ}(I_mix, C_trans)
  输出 81 帧视频

Step 4: 截断前 Fc 帧
  丢弃 V_mix[0:4]（前 4 帧是 VAE 时间压缩帧，含过渡内容）
  保留 V_mix[4:81]（77 帧干净定制视频）

输出：
  1280 × 720, 77 帧定制视频
```

### 1.6 数学公式

**标准 I2V 生成**：

$$V = g_\theta(I, C)$$

其中 $g_\theta$ 是参数为 $\theta$ 的预训练 I2V 模型，$I$ 是输入图像，$C$ 是文本提示，$V$ 是 $F$ 帧视频。

**FFGo 适配后的生成**：

$$V_{mix} = g_{\theta + \Delta\theta}(I_{mix}, C_{trans})$$

其中：
- $\Delta\theta$ 是 LoRA 学到的低秩权重更新
- $I_{mix}$ 是拼合的参考首帧
- $C_{trans} = \langle transition \rangle + C$ 是带过渡短语的提示

**LoRA 低秩分解**：

$$\theta = \theta + \Delta\theta, \quad \Delta\theta = \alpha AB$$

其中 $A \in \mathbb{R}^{d \times r}$, $B \in \mathbb{R}^{r \times k}$, $r \ll \min(d, k)$, $\alpha$ 是缩放因子。

**视频帧长度分解**：

$$F = \{F_c, F_g\}$$

其中：
- $F_c$ = temporal compression frames（时间压缩帧），在 Wan2.2 中 $F_c = 4$
- $F_g$ = generated content frames（生成内容帧），是场景过渡后的干净视频
- 在 81 帧视频中：$F_c = 4$ 帧，$F_g = 77$ 帧

### 1.7 实验结果

**用户研究** (40 名标注者, 200 组标注, Table 1):

| 模型 | Overall Quality ↑ | Object Identity ↑ | Scene Identity ↑ | Avg. Rank ↓ | % Ranked 1st ↑ |
|------|-------------------|-------------------|-------------------|-------------|----------------|
| Wan2.2-I2V-A14B | 2.09 | 3.32 | 3.01 | 3.27 | 3.4% |
| SkyReels-A2 | 2.34 | 2.89 | 3.43 | 3.02 | 4.3% |
| VACE | 3.00 | 3.50 | 3.66 | 2.50 | 11.1% |
| **FFGo (Ours)** | **4.28** | **4.53** | **4.58** | **1.21** | **81.2%** |

**关键结论**：
- FFGo 在所有指标上大幅超越所有 baseline
- 81.2% 的用户将 FFGo 排在第一名
- 仅用 50 个训练样本即超越了用数百万样本训练的 VACE 和 SkyReels-A2
- 支持最多 5 个参考（4 个前景 + 1 个背景），超越 VACE/SkyReels-A2 的 3 个上限

### 1.8 贡献与局限

**三大贡献**：
1. **发现首帧的概念记忆缓冲区角色**：首帧不仅是视频起点，更是存储和融合多参考视觉概念的缓冲区
2. **基于 VLM 的数据策划流水线**：利用 Gemini-2.5-Pro 实现高质量训练数据的自动化生成
3. **轻量级少样本 LoRA 适配**：仅 50 个样本即可将通用 I2V 模型转化为 SOTA 视频定制系统

**局限性**：
1. **参考数量限制**：虽然理论上可以放任意多参考，但实际增加参考会降低每个参考的分辨率，导致身份保持困难。经验上 4 个前景 + 1 个背景 (共 5 个) 为上限。
2. **选择性控制困难**：随着参考数增加，通过文本提示精确控制特定物体变得更难
3. **依赖 Gemini-2.5-Pro**：数据策划流程依赖闭源商业 VLM

---

## 2. 理解验证与纠正

根据你之前的描述，以下是验证结果：

### 你的理解基本正确的部分

1. **"FFGo 将首帧作为 conceptual memory buffer"** → 正确
2. **"使用过渡短语 ad23r2 the camera view suddenly changes 触发场景过渡"** → 正确
3. **"50 个训练样本的 LoRA 微调"** → 正确
4. **"基础模型是 Wan2.2-I2V-A14B"** → 正确
5. **"Gemini-2.5-Pro 用于元素提取和背景清理"** → 正确
6. **"SAM2 用于 RGBA 隔离"** → 正确
7. **"首帧布局：前景在左竖排，背景在右"** → 正确

### 需要补充说明的关键点

> **关于 Fc = 4**：详见 [3.3 Fc=4 含义详解](#33-fc4-含义详解)。简而言之：Fc=4 帧是模型输出中前 4 帧的"过渡帧"（仍带有拼贴布局痕迹），全部丢弃。Fc=4 的数值来源于 Wan2.2 VAE 的时间压缩比=4，不是人为规定的。

> **关于过渡短语 "ad23r2"**：详见 [3.1 Prompt 设计详解](#31-prompt-设计详解)。`ad23r2` 是刻意选择的无意义触发词（类似 DreamBooth 的 `sks`），通过 LoRA 训练与"首帧融合+场景过渡"行为绑定。

---

## 3. 问题解答

### 3.1 Prompt 设计详解

#### 过渡短语为什么是 "ad23r2 the camera view suddenly changes"？

**背景**：FFGo 发现，仅靠 prompt engineering 也能偶尔触发模型的"首帧融合"能力（如 Figure 2 所示），但这种方式存在致命缺陷：
- **模型依赖**：对 Veo 3 有效的过渡短语对 Sora 2 或 Wan 2.2 可能完全无效
- **视频依赖**：同一模型上，不同视频内容需要不同的过渡短语才能生效
- **耗时**：每个场景都需要反复尝试不同短语，类似于"开盲盒"
- **不稳定**：即使找到有效短语，生成结果的一致性也很差

**"ad23r2" 的来源**：这个看似随机的字符串是**刻意设计的无意义触发词**，设计逻辑与 DreamBooth 中的 `sks`、Textual Inversion 中的 `<S*>` 完全一致：

1. **为什么要无意义？** 如果使用一个有意义的词（如 "transform" 或 "transition"），该词在预训练时已经有固定的语义嵌入，LoRA 微调时会与其原有含义冲突。使用无意义的 token 可以让 LoRA 从零开始学习这个 token 的含义，不会干扰模型的预训练知识。
2. **如何确定的？** 论文没有详细说明具体的选择过程，但从 DreamBooth 等先例推断，作者大概率是随机选了一个不在常见词汇表中的字符串。`ad23r2` 很可能就是随手打出来的——就像 DreamBooth 中 `sks` 的选择也没有特殊原因。关键不在于字符串本身是什么，而在于**训练时一致使用同一个字符串作为触发信号**。
3. **后半句 "the camera view suddenly changes" 的作用**：这半句是有语义的，提供了一个"场景突变"的语义锚点。组合效果是：`ad23r2`（LoRA 学到的触发信号）+ `the camera view suddenly changes`（场景过渡的语义提示）。

**整体设计**：

```
C_trans = <transition> + C
```

其中：
- **`<transition>`** = `"ad23r2 the camera view suddenly changes"`
  - `ad23r2`：无意义触发词，LoRA 将其与"从拼贴首帧提取元素并融合"的行为绑定
  - `the camera view suddenly changes`：有语义的场景过渡提示
  - 训练时附加到所有 50 个样本的 caption 前面
  - 推理时只要加上这个前缀，LoRA 就知道需要执行"首帧融合 + 场景过渡"

- **`C`** = 描述目标视频内容的文本提示

**完整 prompt 示例**：

```
训练时：
"ad23r2 the camera view suddenly changes The video features a person in
a Santa hat and festive sweater decorating a layered cake. The cake,
positioned on a turntable, is a focal point as the person meticulously
works on its design."

推理时（用户场景）：
"ad23r2 the camera view suddenly changes Professional-quality video with
rich details. The video features a charming Teddy Bear sipping apple
juice from a bottle using its hand, while delicately holds a vibrant
red rose using its hand, admiring its beauty, perhaps as an offering
or a gesture of affection."
```

**Prompt Enhancement (推理时可选)**：FFGo 还提供了一个 Prompt Enhancement 模板 (Figure 12)，使用 Gemini-2.5-Pro 将用户的简短提示优化为更丰富的描述，以提高生成质量。

**模板如下** (Figure 12 - Video-Prompt Enhancement Output)：
```
Task Description:
You will be given a prompt and several images for video generation.
Your task is to make the prompt richer in description so the model
can understand better. Enclose your caption within <caption></caption>
tags. The caption must emphasize the significance and role of these
components (and some description of each component) throughout the
video. Your caption should exclude unnecessary information such as
"The scene unfolds with a whimsical and heartwarming narrative,
emphasizing the simple joys of life through the Teddy Bear's
endearing actions".
```

### 3.2 训练数据集制作流程分析

#### 完整工作流

```
Step 1: 视频收集与筛选
  HOIGen-1M (~2000) + Veo 3 (5) + 授权视频 → 2205 候选
  人工筛选 → 50 个高质量视频
  裁剪至 81 帧

Step 2: 人工标注
  对每个视频的首帧，人工标注需要分割的元素名称
  例如：cake, Christmas hat, man

Step 3: 元素提取 (Object Extraction, Gemini-2.5-Pro)
  输入：首帧图像 I + 元素名称列表 O
  Prompt (Figure 8):
    "Given the input image, extract the subset {IDENTIFIED OBJECT}
     (i.e., only the specified foreground objects) — return them
     alone with no resizing, compression, or background so the
     output resolution exactly matches the original image."
  输出：每个元素的独立图像（白色背景）

Step 4: 背景清理 (Object Removal, Gemini-2.5-Pro)
  输入：首帧图像 I + 需移除的元素列表
  Prompt (Figure 9):
    "Given the input image, remove the subset {IDENTIFIED OBJECTS}
     entirely. Return the edited image only — it must preserve
     the source resolution (no scaling or compression) and contain
     neither the specified objects nor any artifacts of their removal."
  输出：干净的背景图像

Step 5: SAM 2 精细化
  对 Step 3 输出的元素图像，用 SAM 2 去除白色背景，保留仅含 RGBA 的抠图
  （为什么不直接用白色背景？因为白色背景在拼合时会影响模型对元素边界的理解）

Step 6: 首帧拼合
  画布：1280 × 720
  左半部分：所有前景 RGBA 元素竖排排列，自动缩放适配
  右半部分：干净背景居中放置
  （见 Figure 10 中的 Processed Image 列）

Step 7: Caption 生成 (Gemini-2.5-Pro)
  输入：RGBA 元素图像 + 干净背景 + 完整 81 帧视频
  Prompt (Figure 11 - Training Data Prompt Generation):
    "You are given a video and several images. Generate a descriptive
     caption for the video that prominently features the components
     shown in the images. Wrap your final text in <caption>...</caption>
     tags. The caption must highlight the significance and role of
     these components throughout the video, while omitting filler..."
  输出：描述性 caption C

Step 8: 组装训练数据
  每个训练样本 = (I_mix, C_trans, V_mix)
  其中 C_trans = "ad23r2 the camera view suddenly changes" + C
  V_mix = 原始 81 帧视频
```

#### 为什么使用 RGBA 而非白色背景？

1. **视觉清晰度**：RGBA 的透明背景让模型能清晰区分前景轮廓和背景，白色背景在浅色物体（如白猫、白色产品）上会造成混淆
2. **拼合效果**：在 1280×720 画布上拼合时，RGBA 元素可以无缝叠加，不会产生白色方块
3. **SAM 2 的作用**：Gemini 生成的元素图像往往有白色背景残留，SAM 2 做精细抠图去除残留白色

#### Gemini-2.5-Pro 在此流程中的三个角色

| 任务 | 输入 | 输出 | Prompt |
|------|------|------|--------|
| Object Extraction | 首帧 + 元素名 | 元素图像（白色bg） | Figure 8 |
| Object Removal | 首帧 + 元素名 | 干净背景 | Figure 9 |
| Caption Generation | RGBA元素 + 背景 + 视频 | 描述性 caption | Figure 11 |
| Prompt Enhancement (推理) | 用户prompt + 参考图 | 丰富prompt | Figure 12 |

### 3.3 Fc=4 含义详解

#### 你的理解

> "Fc 有四帧，输入到模型中的就是一帧 I_mix（作为首帧），而后续三帧就是模型生成的过渡帧？因此这一共四帧都是要被丢弃掉的？"

**你的理解基本正确，但需要细化**。

#### Fc = 4 到底是什么？

Fc 在论文中被定义为 **temporal compression frames**（时间压缩帧），其值等于 Wan2.2 VAE 的时间压缩比。

**但 Fc = 4 并不是说"第一帧是输入的 I_mix 加三帧过渡帧"。** 实际情况是：

```
I2V 模型的工作方式（Wan2.2-I2V）：

输入：首帧图像 I_mix → VAE Encoder → 首帧 latent z_0
      噪声 latent z_noise (对应后续 80 帧)

DiT 去噪过程：
  z_0 (首帧条件) + z_noise → 逐步去噪 → 全部 81 帧的 latent

VAE Decoder → 输出 81 帧像素视频
```

关键点：**I_mix 作为条件输入，不直接出现在输出视频中**。模型输出的全部 81 帧都是"生成"的，只是前几帧受首帧条件影响最强烈。

#### 为什么恰好是 4 帧？

Fc = 4 的数值来源于 Wan2.2 VAE 的**时间压缩比 = 4**：

```
81 帧视频 → VAE 时间压缩 → latent 时间维度 = (81-1)/4 + 1 = 21 步
                                                          ↑
                                              第 1 步对应首帧附近的 4 帧
```

VAE 将每 4 帧压缩为 1 个 latent 时间步。**第一个时间步对应的 4 帧（帧 0-3）在解码时仍然保留拼贴布局的痕迹**——因为这 4 帧在 latent space 中距离首帧条件 z_0 最近，场景过渡还未完成。

所以 Fc = 4 不是"谁规定的"，而是**由 VAE 架构自然决定的**：
- VAE 时间压缩比 = 4 → 首帧对应的 latent 时间步解码出 4 帧 → 这 4 帧是过渡区域
- 如果某个模型的 VAE 时间压缩比是 8，那 Fc 就会是 8

#### 正确的理解

```
模型输出 81 帧视频：
  帧 0-3 (Fc=4)：仍带有拼贴布局痕迹的"过渡帧"  → 全部丢弃
  帧 4-80 (Fg=77)：场景过渡完成，干净定制视频     → 保留

注意：输入的 I_mix 不在输出中。I_mix 是条件，不是输出的一部分。
全部 81 帧都是模型"生成"的，但前 4 帧质量差（过渡不完全），所以丢弃。
```

**类比**：想象你给 AI 看了一张拼贴画（I_mix），让它根据这张拼贴画创作一段视频。AI 输出的视频开头几帧（4帧）还在"消化"拼贴画的内容，画面比较混乱，但从第 5 帧开始，AI 已经理解了拼贴画中的元素并将它们自然地融入了场景中。FFGo 就是把前 4 帧"消化过程"剪掉。

### 3.4 VLM 发展历程与 Gemini 替代方案

#### VLM 发展完整历程

##### 第一阶段：视觉-语言对齐（2021-2022）——只能"理解"

| 时间 | 模型 | 机构 | 关键贡献 |
|------|------|------|----------|
| 2021.01 | **CLIP** | OpenAI | 对比学习对齐图文表征，4亿图文对训练，成为标准视觉编码器 |
| 2021.01 | **DALL-E** | OpenAI | VQ-VAE + 自回归的文本到图像生成（非统一模型） |
| 2022.04 | **Flamingo** | DeepMind | 少样本多模态学习，视觉编码器 + 冻结 LLM + 交叉注意力 |

##### 第二阶段：大规模 VLM（2023）——强大的图像理解

| 时间 | 模型 | 机构 | 关键贡献 |
|------|------|------|----------|
| 2023.01 | **BLIP-2** | Salesforce | Q-Former 桥接冻结视觉编码器和冻结 LLM |
| 2023.04 | **LLaVA** | UW-Madison | 视觉指令微调，开源，催生大量变体 |
| 2023.07 | **Emu** | BAAI | 首个统一自回归目标的多模态模型（理解+生成） |
| 2023.09 | **GPT-4V** | OpenAI | 多模态 GPT-4，将 VLM 带入主流 |
| 2023.12 | **Gemini 1.0** | Google | 原生多模态（文本/图像/视频/音频/代码） |

##### 第三阶段：统一 VLM 萌芽（2024）——理解+生成开始融合

| 时间 | 模型 | 机构 | 开源？ | 关键贡献 |
|------|------|------|--------|----------|
| 2024.01 | **Qwen-VL** | 阿里 | 是 | 强大的开源 VLM，中文能力优异 |
| 2024.04 | **InternVL 1.5** | 上海 AI Lab | 是 | 开源 VLM，与 GPT-4V 竞争 |
| 2024.05 | **GPT-4o** | OpenAI | 否 | 原生多模态，多模态输入输出开始统一 |
| 2024.05 | **Chameleon** | Meta | 是 | 混合模态早期融合，离散 VQ 分词器，统一自回归 |
| 2024.08 | **Show-o** | ShowLab | 是 | 单 Transformer 同时理解+生成，混合 AR + 离散扩散 |
| 2024.09 | **Emu3** | BAAI | 是 | "Next-token prediction is all you need"——纯自回归多模态 |
| 2024.10 | **Janus** | DeepSeek | 是 | 解耦视觉编码：理解用 SigLIP，生成用 SDXL-VAE，统一 Transformer |

##### 第四阶段：全能统一 VLM（2025-2026）——处理+生成图像/视频/文本

| 时间 | 模型 | 机构 | 开源？ | 关键贡献 |
|------|------|------|--------|----------|
| 2025.01 | **Janus-Pro** | DeepSeek | 是 (1B/7B) | 改进的 Janus，理解和生成均大幅提升 |
| 2025.03 | **Gemini 2.0 Flash** | Google | 否 | 首个原生图像生成的 Gemini |
| 2025.05 | **BLIP3-o** | Salesforce | 完全开源 (4B/8B) | CLIP + Flow Matching，代码/权重/训练脚本全部开放 |
| 2025.05 | **Gemini 2.5 Pro** | Google | 否 | SOTA 推理+多模态，1M token 上下文 |
| 2025.06 | **Show-o2** | ShowLab | 是 | 改进的统一模型 |
| 2025.09 | **Qwen3-VL** | 阿里 | 是 (2B-235B) | Dense 2B/4B/8B/32B + MoE 30B-A3B/235B-A22B，1M 上下文，**与 Gemini 2.5 Pro 竞争** |
| 2025 | **InternVL3/3.5** | 上海 AI Lab | 是 (至78B) | SOTA 开源理解模型 |

**什么是"统一 VLM" (Unified VLM)**？

传统 VLM 只能"理解"图像（image → text），而统一 VLM 能同时"理解"和"生成"多种模态：
- **输入**：文本、图像、视频、音频
- **输出**：文本、图像、视频、音频

FFGo 论文中提到的三大统一 VLM：
- **Gemini-2.5-Pro** [6]（Google DeepMind）—— 闭源 API
- **Qwen2.5-Omni** [36]（阿里通义）—— 开源
- **GPT-4o** [24]（OpenAI）—— 闭源 API

**重要区分**：大多数开源"VLM"（如 Qwen2.5-VL, InternVL, LLaVA）仍然是**仅理解**模型（只能输出文本），不能生成图像。真正能同时理解+生成图像的统一模型较少（Janus-Pro, BLIP3-o, Show-o2 等）。但对于 FFGo 的数据策划流程，**不需要图像生成能力**——可以用专门的分割/修复模型替代。

#### FFGo 为什么选择 Gemini-2.5-Pro？

1. **图像生成+编辑能力强**：能精确提取/移除指定物体并保持分辨率
2. **指令跟随能力强**：能严格按照 prompt 要求输出（如"不缩放、不压缩"）
3. **视频理解能力**：能观看完整视频并生成描述性 caption
4. **端到端简便**：一个模型解决所有任务（提取、移除、描述），无需组合多个工具

#### Gemini-2.5-Pro 的替代方案

##### 商业 API

| 模型 | 提供方 | 图像理解 | 图像生成/编辑 | 视频理解 | 国内可用性 | 价格 |
|------|--------|---------|-------------|---------|-----------|------|
| Gemini-2.5-Pro | Google | 极强 | 强 | 极强 | 需VPN | 免费额度有限，按量付费 |
| GPT-4o / GPT-Image-1 | OpenAI | 强 | 中 | 强 | 需VPN | $2.5/M in, $10/M out |
| **通义千问 API** | 阿里 | 强 | 不直接支持 | 强 | **直接可用** | $0.4/M in, $1.2/M out (Plus) |
| **智谱 AI API** | 智谱 | 强 | 中 | 中 | **直接可用** | 有免费额度 |

##### 开源模型（推荐用于国内研究者）

| 模型 | 参数量 | 图像理解 | 图像生成 | 视频理解 | 推荐用途 |
|------|--------|---------|---------|---------|----------|
| **Qwen3-VL-235B-A22B** | 235B (MoE) | 极强（≈Gemini 2.5 Pro） | 不支持 | 极强 | Caption 生成（最强开源） |
| **Qwen3-VL-72B** | 72B | 极强 | 不支持 | 强 | Caption 生成（推荐） |
| **Qwen2.5-VL-72B** | 72B | 极强 | 不支持 | 强 | Caption 生成 |
| **InternVL3.5** | 至 78B | 极强 | 不支持 | 强 | Caption 生成替代 |
| **Janus-Pro-7B** | 7B | 强 | 支持 | 弱 | 理解+生成统一（轻量） |
| **BLIP3-o** | 4B/8B | 强 | 支持 | 中 | 完全开源统一模型 |

##### 针对 FFGo 流程各步骤的替代方案

**Object Extraction（元素提取）**：
- **替代方案 A（推荐）**：GroundingDINO + SAM2（我们 PVTT 流程已在使用）
  - GroundingDINO 检测物体边界框 → SAM2 精细分割 → 获得 RGBA 抠图
  - 完全开源免费，不依赖任何 API
  - 实际上比 Gemini 的图像生成式提取**更精确**（像素级分割 vs 生成式重建）
- **替代方案 B**：Qwen3-VL + SAM2
  - Qwen3-VL 识别元素位置和边界框 → SAM2 分割
  - 适合复杂场景的元素识别

**Object Removal（背景清理）**：
- **替代方案 A（推荐）**：**LaMa** (Large Mask Inpainting)
  - 完全开源，专门做图像修复，快速轻量
  - 用 SAM2 的 mask 标注需要移除的区域 → LaMa 修复
- **替代方案 B**：Stable Diffusion XL Inpainting
  - 扩散模型修复，支持文本引导
  - 效果接近 Gemini，但需要 GPU
- **替代方案 C**：**ProPainter** （视频场景推荐）
  - 视频专用修复模型，保证时序一致性

**Caption Generation（描述生成）**：
- **替代方案 A（推荐）**：**Qwen3-VL-72B**
  - 开源 SOTA，视频理解能力与 Gemini 2.5 Pro 接近
  - 原生中文支持
  - 部署需 ~160GB 显存（可用 4-bit 量化降至 ~40GB）
- **替代方案 B**：通义千问 API (qwen-vl-max / qwen-plus)
  - 阿里云 DashScope API，**国内直接可用**，有免费额度
  - 价格极低：Plus 版 $0.40/M input tokens
- **替代方案 C**：InternVL3.5
  - 开源，多模态理解能力强

##### 推荐的完全开源替代流程（适合国内研究者）

```
元素提取：GroundingDINO + SAM2              （已有，完全开源）
背景清理：LaMa                              （开源，轻量快速）
Caption：Qwen3-VL-72B（本地部署）            （开源 SOTA）
    或   通义千问 API（DashScope，国内直接可用）（有免费额度）
```

##### 国内可用的免费/低成本 API 服务

| 服务 | 可用模型 | 免费额度 | 备注 |
|------|---------|---------|------|
| **阿里云 DashScope** | Qwen3-VL, Qwen2.5-VL, Qwen-Max | 部分模型有 100 万 token 免费 | 国内直接可用 |
| **硅基流动 (SiliconFlow)** | 各类开源模型 | 有免费额度 | 国内平台 |
| **魔搭 (ModelScope)** | 所有 Qwen 模型 | 免费下载自部署 | 阿里的国内 HuggingFace 镜像 |
| **智谱 AI** | GLM-4V, CogVLM2 | 有免费额度 | 国内平台 |

#### VLM 必读论文清单

##### 基础理解类（了解 VLM 发展脉络）

| 论文 | 年份 | 重要性 | 一句话总结 |
|------|------|--------|----------|
| **CLIP**: Learning Transferable Visual Models From Natural Language Supervision | 2021 | 奠基 | 对比学习对齐视觉-语言表征，后续所有 VLM 的视觉编码器基石 |
| **LLaVA**: Visual Instruction Tuning | 2023 | 范式 | 用 GPT-4 生成的指令数据微调视觉-语言模型，开创视觉指令微调范式 |
| **BLIP-2**: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models | 2023 | 架构 | Q-Former 桥接冻结视觉编码器和 LLM，高效训练方案 |

##### 统一理解+生成类（FFGo 关注的核心类别）

| 论文 | 年份 | 重要性 | 一句话总结 |
|------|------|--------|----------|
| **Emu**: Generative Pretraining in Multimodality | 2023 | 先驱 | 首个将视觉和文本统一在自回归目标下的大规模多模态模型 |
| **Chameleon**: Mixed-Modal Early-Fusion Foundation Models | 2024 | 架构 | Meta 的混合模态早期融合，所有模态都用离散 token |
| **Janus**: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation | 2024 | 开源 | DeepSeek，解耦理解/生成视觉编码器，统一 Transformer |
| **Emu3**: Next-Token Prediction is All You Need | 2024 | 理念 | "纯自回归多模态"——token 统一理解和生成 |
| **BLIP3-o** | 2025 | 实用 | Salesforce 完全开源（代码+权重+数据），CLIP + Flow Matching |

##### 视频生成/定制类（与 FFGo 直接相关）

| 论文 | 年份 | 重要性 | 一句话总结 |
|------|------|--------|----------|
| **Wan**: Open and Advanced Large-Scale Video Generative Models | 2025 | 基础 | FFGo 的基础模型，理解 VAE 时间压缩和 MoE 架构 |
| **VACE**: All-in-One Video Creation and Editing | 2025 | 基础 | 我们使用的框架，VCU 统一视频创编辑 |
| **SkyReels-A2**: Compose Anything in Video Diffusion Transformers | 2025 | 对比 | FFGo 的 baseline 之一，架构修改路线的代表 |
| **DreamBooth**: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation | 2023 | 方法 | 理解 FFGo 的触发词 "ad23r2" 的设计灵感来源 |
| **LoRA**: Low-Rank Adaptation of Large Language Models | 2022 | 方法 | 理解 FFGo 的少样本 LoRA 适配技术 |
| **Video Models are Zero-Shot Learners and Reasoners** (Wiedemer et al.) | 2025 | 启发 | 发现预训练 I2V 模型具有感知和推理的"涌现能力"，与 FFGo 发现的首帧记忆缓冲区思想一脉相承 |

##### VLM 用于数据策划类（FFGo pipeline 相关）

| 论文 | 年份 | 重要性 | 一句话总结 |
|------|------|--------|----------|
| **Gemini 2.5**: Our Most Intelligent AI Model (Technical Report) | 2025 | 工具 | FFGo 数据策划的核心工具，理解其图像生成/编辑能力 |
| **Qwen2.5-VL** / **Qwen3-VL** Technical Report | 2025 | 替代 | 最强开源替代方案，国内可用 |
| **SAM 2**: Segment Anything in Images and Videos | 2024 | 工具 | FFGo 使用 SAM2 进行 RGBA 抠图 |
| **GroundingDINO**: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection | 2023 | 工具 | 文本引导的开放集目标检测，我们 PVTT 流程使用 |

---

## 4. PVTT 应用分析

### 4.1 直接使用 FFGo（base model + 预训练 LoRA）

#### 目标

不从头训练 LoRA，而是直接下载 FFGo 论文发布的 base model (Wan2.2-I2V-A14B) 和已训练好的 LoRA 权重，用于 PVTT 商品视频生成。

#### 资源获取（全部已公开）

| 资源 | HuggingFace 地址 | 下载命令 |
|------|-----------------|---------|
| **FFGo LoRA 权重** | `Video-Customization/FFGO-Lora-Adapter` | `huggingface-cli download Video-Customization/FFGO-Lora-Adapter --local-dir ./Models/Lora` |
| **Wan2.2-I2V-A14B** | `Wan-AI/Wan2.2-I2V-A14B` | `huggingface-cli download Wan-AI/Wan2.2-I2V-A14B --local-dir ./Wan2.2-I2V-A14B` |
| **FFGo 代码** | GitHub: `zli12321/FFGO-Video-Customization` | `git clone https://github.com/zli12321/FFGO-Video-Customization` |

- 许可证：全部 Apache 2.0
- Wan2.2-I2V-A14B：MoE 架构，~27B 总参数（14B active/step），支持 480P/720P
- 国内下载：使用 `HF_ENDPOINT=https://hf-mirror.com` 前缀加速

#### 使用流程（假设权重可获取）

```
Step 1: 下载模型
  - Wan2.2-I2V-A14B base model（两个 Transformer 的权重）
  - FFGo LoRA 权重（low-noise + high-noise 各一个 LoRA）

Step 2: 准备 PVTT 输入
  对每个 PVTT 任务：
  - 产品 RGBA 图像（已有，output_dino_rgba/）
  - 原视频首帧作为背景（或用 LaMa 移除原物体后的干净背景）
  - 拼合 FFGo 式首帧 I_mix（左侧：产品图，右侧：背景）
  - 文本 prompt = "ad23r2 the camera view suddenly changes" + target_prompt

Step 3: 推理
  加载 base model + LoRA → 输入 I_mix + prompt → 输出 81 帧
  丢弃前 4 帧 → 保留 77 帧干净视频

Step 4: 评估
  与实验十 baseline 进行对比
```

#### 关键限制

1. **无空间控制**：FFGo 只能告诉模型"用这些元素生成视频"，但不能指定产品在画面中的精确位置。对于 PVTT 这类需要产品替换到特定位置的任务，这是硬伤。
2. **场景不可控**：FFGo 生成的背景完全由模型自由发挥（根据 prompt），无法保证与原始视频的场景一致。
3. **适用场景有限**：FFGo 更适合"给定产品+背景，自由生成展示视频"的场景，而非 PVTT 的"在原视频中替换特定物体"。

#### 结论

FFGo 直接使用可以作为一个**对比 baseline**（看"纯首帧参考"能做到什么程度），但它**不能直接解决 PVTT 的核心需求**（在原视频中精确替换物体）。需要与 VACE 的空间控制能力结合。

### 4.2 方案 A：VACE I2V + FFGo 首帧参考

#### VACE 的 I2V 能力说明

**VACE 确实可以实现 I2V**。虽然 VACE 基于 Wan2.1 T2V 改造，但它的 VCU 输入机制使其天然支持首帧生成视频：

```
VACE 实现 I2V 的方式：
  Template: [首帧图像, 空帧, 空帧, ..., 空帧]
  Mask:     [全0(保留), 全1(生成), 全1(生成), ..., 全1(生成)]

  mask=0 的区域 → VACE 保留 template 原始内容
  mask=1 的区域 → VACE 根据条件自由生成

  效果 = 首帧固定 + 后续帧由模型生成 = I2V
```

这在我们之前的实验中已经验证过——VACE 的 mask 首帧全 0 + 后续帧全 1 就是 I2V。

#### VACE VCU 输入解释

先澄清 VACE 的 VCU 三元组概念：

| 组件 | 含义 | 在我们实验中的对应 |
|------|------|-------------------|
| **Template** | 输入给模型看的"参考视频" | 原始视频帧序列（在 mask 区域用中性色填充） |
| **Mask** | 二值序列，标记哪些像素需要模型生成 | 物体 mask（1=需要生成/替换，0=保留原样） |
| **Filler**（不需要单独提供） | mask=1 区域的初始填充内容。实际上就是 template 中 mask 区域的内容 | neutral fill 后的灰色区域、或嵌入 Canny 边缘的区域 |

> **Filler 不是一个独立输入**。在 VACE 的实际代码中，template 本身就包含了 filler 信息——mask=0 区域是原始内容，mask=1 区域的内容就是 filler（我们用中性色或 Canny 边缘填充过的部分）。VACE 内部会将 template 和 mask 拼接送入模型。

#### 你的设计

```
输入：
  ref_img = FFGo 式首帧（左侧：产品 RGBA，右侧：原视频首帧去除被替换物体后的画面）
  mask_seq = 首帧 mask=全0（保留首帧不动），后续帧 mask=全1（全部重新生成）
  prompt = 场景转换提示词 + 原始 target prompt

构造 VACE VCU：
  Template: [ref_img,  空帧/灰帧,  空帧/灰帧,  ...]
  Mask:     [全0(保留), 全1(生成),   全1(生成),   ...]
```

**关键点**：这是一个纯 I2V 实验——首帧固定为 FFGo 式拼贴画，后续帧 mask 全为 1，由模型完全从头生成整个画面。**不是**在原视频帧的物体 mask 区域做局部替换，而是让 VACE 从首帧的视觉信息出发，自由生成后续所有帧。

#### 分析

**这个设计的核心假设：VACE 在 I2V 模式下能从 FFGo 式拼贴首帧中提取产品和场景信息，生成包含目标产品的连贯视频。**

```
首帧 (ref_img):
  ┌──────────┬────────────────────┐
  │ 产品图   │  原视频首帧画面     │
  │ (RGBA)   │  (物体已移除)      │
  └──────────┴────────────────────┘
  首帧 mask=0，完全保留不动

后续帧 (template):
  ┌─────────────────────────────────┐
  │  灰色/空帧                      │
  │  mask=1，由模型完全生成          │
  └─────────────────────────────────┘
```

**优势**：
- 不依赖逐帧 mask 精度——后续帧全部由模型生成，不需要物体 mask 序列来做局部合成
- 首帧提供了所有必要的视觉条件：产品外观（左侧）+ 场景上下文（右侧）

**挑战**：

1. **VACE 不理解"拼贴画"**：VACE 没有被训练来理解"首帧左半部分是产品参考，右半部分是场景"这种 FFGo 特有的布局。它会把首帧当作一个正常的视频帧来看待。
2. **首帧布局异常**：首帧的左右分栏布局和正常视频帧差异很大，VACE 在 I2V 模式下可能生成不连贯的后续帧——它不知道后续画面应该是一个融合了两侧信息的正常视频。
3. **mask=0 锁定首帧**：首帧保持拼贴画原样，这意味着最终视频的第一帧是拼贴画（可能需要丢弃前几帧），但 VACE 不像 FFGo 那样被训练过"从拼贴过渡到正常视频"的能力。
4. **没有空间控制**：由于后续帧 mask=全1，模型完全自由生成，产品在画面中的位置和大小不受原视频约束。

**关于 Fc=4 在 VACE 上是否适用——详细分析**：

FFGo 论文中丢弃前 4 帧（Fc=4）是因为 Wan2.2 VAE 的时间压缩比=4：每 4 帧压缩为 1 个 latent 时间步。第一个 latent 时间步对应的 4 帧在解码时仍然保留拼贴布局的痕迹（因为它们在 latent space 中距离首帧条件最近）。

**但 Fc=4 的"4 帧后干净切换"依赖两个条件：**

1. **VAE 时间压缩比 = 4**：这是 Wan2.2 VAE 的固有属性，VACE 也使用相同的 VAE，所以这个条件成立。
2. **LoRA 训练模型学会了"拼贴首帧 → 正常视频"的过渡行为**：FFGo 用 50 个样本 + LoRA 微调了 Wan2.2-I2V，让模型学会在看到拼贴首帧 + `ad23r2` 触发词时，快速完成场景过渡。**这个条件在 VACE 上不成立**——VACE 未经过此类微调。

因此，在 VACE 上：
- VAE 层面的 4 帧时间压缩仍然存在，前 4 帧确实最接近首帧条件
- 但模型层面没有学会"在第 5 帧开始输出正常视频"这种行为
- **转场时间不可预测**：可能 4 帧就完成，也可能 20+ 帧仍在过渡，甚至可能整个视频都无法完成转场

我们的实验实现中采用了 **SSIM 自动检测**来替代固定的 Fc=4 丢弃：计算相邻帧之间的结构相似度，当连续帧 SSIM 稳定超过阈值时认为转场结束。这比固定丢弃更适合未经微调的模型。

**实验价值**：即使 VACE 未经 LoRA 微调大概率无法完美理解拼贴布局，但这个实验可以：
- 验证 VACE 的 I2V 能力上限——在拼贴首帧条件下能生成什么程度的视频
- 作为后续 LoRA 微调的 baseline 对比（微调前 vs 微调后）
- 探索 VACE 是否内在具备类似 FFGo 的"首帧概念记忆"能力

#### 解决方案（若实验效果不佳）

**方案 A-1：LoRA 微调 VACE 学习首帧过渡**

与 FFGo 思路相同：制作 ~50 个训练样本，每个样本的首帧是 FFGo 式拼贴，后续帧是正常视频。LoRA 训练 VACE 学会从拼贴首帧过渡到正常视频。

- 优点：最接近 FFGo 原始方案
- 缺点：需要训练数据制作 + LoRA 训练 + 无法利用 VACE 已有的 mask video 空间控制

**方案 A-2：不使用 FFGo 拼贴布局，改用 VACE 原生参考图机制**

不拼贴。直接将产品图作为 VACE 的 reference image（参考图），首帧使用原视频首帧（物体区域 neutral fill），mask 正常设置。

```
Template: [原视频帧0(neutral fill), 原视频帧1(neutral fill), ...]
Mask:     [物体mask0,              物体mask1,               ...]
Reference: 产品 RGBA 图像
Prompt:   "产品描述"
```

这就是我们之前实验的思路（实验 1-10 系列），但可以探索在 prompt 中更精确地描述产品外观来增强身份保持。**不需要 LoRA**，但身份保持能力有限。

### 4.3 方案 B：VACE mask video + FFGo 首帧产品参考

#### 你的设计（已纠正理解）

```
输入：
  mask_video = 原视频各帧，物体区域 GrowMask + neutral fill（和之前实验一样）
  ref_img    = 仅产品 RGBA 图像（不放背景，因为画面信息已在 mask video 中）
  mask_seq   = 物体 mask 序列
  prompt     = 目标描述

目标：VACE 在后续帧的 mask 区域内，参照首帧展示的产品进行渲染，非 mask 区域保持不变。
```

#### 分析

这个设计的核心思路是：**用一个纯产品图作为首帧，让 VACE 理解"这就是我要在后续帧的 mask 区域里渲染的东西"。**

**具体输入构造**：

```
Template video:
  帧 0: 产品 RGBA 图像（居中放置在画布上？全画面展示？）
  帧 1: 原视频帧 1（物体区域 GrowMask + neutral fill）
  帧 2: 原视频帧 2（同上）
  ...
  帧 N: 原视频帧 N

Mask video:
  帧 0: 全 0（首帧保留产品图原样）或 全 1（首帧也生成）？
  帧 1: 物体 GrowMask + BlockifyMask
  帧 2: 同上
  ...

Prompt: 产品描述
```

#### 问题分析

**问题 1：首帧产品图与后续帧之间的巨大视觉断裂**

首帧是纯产品图（白色/透明背景上的产品），后续帧是正常视频画面。VACE 被训练处理的是**时间连续的视频**，而这里首帧和后续帧之间有巨大的视觉跳变。

- 如果首帧 mask=0（保留产品图）：VACE 会尝试生成从产品图"自然过渡"到视频场景的内容，但它没有被训练过这种过渡
- 如果首帧 mask=1（也要生成）：产品图信息被当作"初始填充"而非"参考"，模型可能会覆盖掉产品外观

**问题 2：VACE 无法建立"首帧产品图 → 后续帧 mask 区域渲染"的映射**

VACE 的设计逻辑是：template 提供每帧的画面背景/上下文，mask 标记需要生成的区域，模型在 mask 区域根据上下文和 prompt 生成内容。

你的设计期望 VACE 建立一个更高层的推理：**"首帧展示的产品外观 = 我在后续帧 mask 区域应该渲染的东西"**。但 VACE 没有被训练来做这种跨帧引用推理——这恰恰是 FFGo 需要 LoRA 微调才能激活的能力。

**问题 3：你说"画面信息已在 mask video 中"是对的，但首帧仍然突兀**

后续帧确实有完整的场景信息（neutral fill 只影响 mask 区域）。但首帧作为一个孤立的产品图，在时间维度上与帧 1 完全脱节，VACE 很难理解这个"异类"首帧的意图。

#### VACE 能否看懂这个输入组合？

**不经过 LoRA 微调的情况下，大概率不能。** VACE 会把首帧的产品图当作视频的"第一帧画面"，而不是"产品参考"。它不会自动建立"首帧产品 → 后续帧 mask 区域渲染"的因果关系。

可能的结果：
- mask 区域生成的内容与首帧产品图无关（VACE 忽略了首帧的参考意图）
- 或者 mask 区域试图"续接"产品图的画面，生成混乱的过渡

#### 解决方案

**方案 B-1（推荐）：LoRA 微调 VACE 学习"首帧参考 → mask 区域渲染"**

这是最直接的解决方案。核心思路与 FFGo 类似，但适配 VACE 的 VCU 框架：

1. **制作训练数据** (~50 个样本)：
   - 收集视频，标注其中的物体
   - 训练样本：首帧 = 物体的 RGBA 参考图，后续帧 = 正常视频（物体区域标 mask）
   - 训练目标：VACE 学会"参考首帧的物体外观，在后续帧的 mask 区域渲染该物体"

2. **LoRA 训练 VACE**：
   - 可能需要自定义的过渡短语（类似 `ad23r2`）
   - rank 和训练参数参考 FFGo

3. **推理**：按你设计的输入格式使用

- 优点：最符合你的设计意图，空间控制 + 身份保持
- 缺点：需要训练数据 + GPU + 训练时间

**方案 B-2：不用 LoRA，改变首帧策略**

如果暂时不想训练 LoRA，可以修改首帧策略让 VACE 更容易理解：

```
替代方案：将产品图直接粘贴到首帧的 mask 区域中

Template video:
  帧 0: 原视频首帧，mask 区域粘贴上产品图（按位置和大小适配）
  帧 1: 原视频帧 1（neutral fill）
  帧 2: 原视频帧 2（neutral fill）
  ...

Mask video:
  帧 0: 全 0（首帧完全保留——产品已经在正确位置了）
  帧 1: 物体 mask
  ...
```

这其实就是我们实验十（Canny 首帧粘贴）的思路，只不过把 Canny 边缘换成完整的产品图。VACE 更容易理解："首帧在这个位置有这个产品，后续帧的相同位置需要生成类似内容"。

### 4.4 总体可行性分析与建议方向

#### 你设计的三个方案总结

| 方案 | 核心思想 | 不经 LoRA 是否可行 | 需要 LoRA 微调？ |
|------|---------|-------------------|----------------|
| **直接使用 FFGo** | base model + 预训练 LoRA，纯首帧参考 | 可行（如有权重） | 否（用现成的） |
| **方案 A：VACE I2V + FFGo 首帧** | FFGo 拼贴首帧 + VACE I2V（后续帧 mask=全1，全部重新生成） | 待验证（实验探索 VACE 的 I2V 上限） | 理想情况下需要 |
| **方案 B：mask video + 首帧产品参考** | 产品图首帧 + neutral fill mask video | 不可行（VACE 不懂跨帧引用） | 是 |

**核心问题**：方案 A 和 B 的共同难点在于——**VACE 没有被训练来理解"首帧是参考，后续帧根据参考渲染 mask 区域"这种模式**。FFGo 证明这种能力在 I2V 模型中是"内在"的，但在 VACE (T2V+VCU) 中是否也内在存在，需要通过 LoRA 微调来激活和验证。

#### 我的额外融合思路

##### 思路 1：Wan2.1 + FFGo 思想（绕过 VACE）

**核心**：不使用 VACE，直接在 Wan2.1 T2V 或 Wan2.1 I2V 上应用 FFGo 思想。

```
方式 A：Wan2.1-I2V + FFGo LoRA

  如果 Wan2.1 有 I2V 模型可用：
  1. 直接复用或改造 FFGo 的 LoRA 训练流程
  2. 用 PVTT 数据集的产品图 + 视频首帧制作 FFGo 式拼贴训练数据
  3. 训练 LoRA 让 Wan2.1-I2V 学会从拼贴首帧过渡
  4. 推理时输入拼贴首帧 + prompt

  优点：最贴近 FFGo 原始方案
  缺点：无空间控制（产品位置不可控）
  适合：产品展示视频（不要求精确位置替换）

方式 B：Wan2.1-T2V + 条件注入

  在 Wan2.1 T2V 上探索条件注入机制：
  1. 将产品参考图编码为 visual token
  2. 与 text prompt 一起送入 DiT
  3. 类似 IP-Adapter 的思路

  优点：不限于首帧，产品信息贯穿全程
  缺点：需要架构修改（不是纯 LoRA）
```

**与 VACE 方案的区别**：Wan2.1 原始模型没有 VCU 的 mask 机制，所以没有空间控制能力，但也没有 VACE "template 被 mask 分割"带来的复杂性。更适合 FFGo 的"纯首帧参考"模式。

##### 思路 2：VACE + FFGo 思想（利用 VACE VCU）

**核心**：在 VACE 框架内融合 FFGo 的身份保持能力。

```
方案 C：VACE Reference-Guided Inpainting（参考引导修复）

  这是我认为最有潜力的方向。不使用 FFGo 的拼贴首帧，
  而是将产品参考信息编码后注入到 VACE 的生成过程中：

  1. 保持 VACE 的 VCU 输入不变：
     Template = 原视频（mask 区域 neutral fill）
     Mask = 物体 mask 序列

  2. 新增产品参考条件通道：
     - 将产品 RGBA 图像通过 CLIP/VAE 编码为 visual embedding
     - 在 DiT 的 cross-attention 中注入产品 visual embedding
     - 或使用 IP-Adapter 风格的注入

  3. LoRA + 少量数据训练：
     训练数据 = (原视频, 物体mask, 目标产品图, 目标视频)
     目标：VACE 在 mask 区域根据产品参考生成匹配的内容

  优点：
  - 空间控制（mask 精确定位）+ 身份保持（产品参考注入）
  - 不破坏 VACE 原有能力
  - 产品参考信息贯穿所有帧（不仅限于首帧）

  缺点：
  - 需要少量架构修改（添加 cross-attention 注入层）
  - 训练数据制作复杂度中等

方案 D：VACE 首帧产品粘贴 + LoRA（最简可行方案）

  最接近你的方案 B，但做关键修改：

  1. 不把产品图作为独立首帧
  2. 而是将产品图粘贴到首帧的 mask 区域内（正确位置和大小）
  3. 首帧 mask=0（保留粘贴后的画面）
  4. 后续帧正常 mask

  Template: [首帧(产品已粘贴到正确位置), 帧1(neutral fill), 帧2, ...]
  Mask:     [全0,                        物体mask1,          mask2, ...]

  + LoRA 微调：训练 VACE 在看到首帧中有产品图时，
    在后续帧 mask 区域维持该产品的外观一致性

  这比方案 B 更可行，因为首帧和后续帧的空间布局一致
  （产品在同一位置），VACE 更容易学会时间一致性。
```

##### 思路 3：两步法——FFGo 生成 + VACE 精修

```
Step 1: FFGo 生成产品展示视频
  输入：产品图 + 原视频首帧背景 + prompt
  输出：77 帧产品展示视频（位置不受控）

Step 2: VACE 精修/替换
  输入：
    Template = 原始视频（neutral fill）
    Mask = 物体 mask 序列
    Reference = FFGo 生成的视频中提取的产品帧
  VACE 将 FFGo 生成的产品外观"移植"到原视频的正确位置

  优点：FFGo 保证产品外观逼真 + VACE 保证空间位置精确
  缺点：两步生成，累计误差可能大
```

#### Wan2.1 vs VACE 融合 FFGo 的区别

| 维度 | Wan2.1 + FFGo | VACE + FFGo |
|------|--------------|-------------|
| **模型基础** | Wan2.1 (T2V 或 I2V) | Wan2.1 + VCU（VACE） |
| **有空间控制？** | 无（除非自行添加） | 有（VCU mask 机制） |
| **FFGo 首帧可直接用？** | 是（如果用 I2V 模型） | 否（VACE 不理解拼贴布局） |
| **需要 LoRA？** | 是（FFGo 风格的 LoRA） | 是（需要教 VACE 理解参考） |
| **适合场景** | 自由生成产品展示视频 | 在原视频中精确替换物体 |
| **PVTT 适配度** | 中（无位置控制） | 高（mask 控制位置） |
| **实现复杂度** | 低（接近 FFGo 原版） | 中-高（需要设计 VCU 输入策略） |
| **学术新颖性** | 低（FFGo 复现） | **高**（VACE+FFGo 融合是新方向） |

**关键区别**：Wan2.1 是"白纸一张"（没有 mask 机制），更适合 FFGo 的"自由生成"模式；VACE 有 mask 机制提供空间控制，更适合 PVTT 的"精确替换"需求。但 VACE 的 VCU 输入格式与 FFGo 的拼贴首帧不直接兼容，需要设计新的融合方式。

#### 建议的研究路线图

```
Phase 0 (当前)：
  → 跑通实验十 PVTT baseline + 尝试直接使用 FFGo（如果权重可获取）

Phase 1 (短期实验)：
  → 方案 D：首帧产品粘贴 + LoRA
    最小化改动，测试 VACE 能否通过 LoRA 学会"首帧产品外观 → 后续帧 mask 渲染"
    成本低（~50 样本 + 几小时训练），可以快速验证思路

Phase 2 (中期研究)：
  → 方案 C：VACE Reference-Guided Inpainting
    在 VACE 中添加产品参考条件通道（IP-Adapter 风格）
    学术新颖性高，可以作为论文核心贡献

Phase 3 (论文产出)：
  → 对比分析：实验十 baseline vs FFGo 直接使用 vs 方案 D vs 方案 C
  → 论文方向："Reference-Guided Video Inpainting for Product Placement"
```

---

## 附录：FFGo 论文中的 Gemini Prompt 模板

### A. Object Extraction Task (Figure 8)

```
Prompt – Given the input image, extract the subset {IDENTIFIED OBJECT}
(i.e., only the specified foreground objects) — return them alone with
no resizing, compression, or background so the output resolution
exactly matches the original image.
```

### B. Object Removal Task (Figure 9)

```
Prompt – Given the input image, remove the subset {IDENTIFIED OBJECTS}
entirely. Return the edited image only — it must preserve the source
resolution (no scaling or compression) and contain neither the specified
objects nor any artifacts of their removal.
```

### C. Training Data Caption Generation (Figure 11)

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

### D. Video-Prompt Enhancement (Figure 12)

```
Task Description:
You will be given a prompt and several images for video generation.
Your task is to make the prompt richer in description so the model can
understand better. Enclose your caption within <caption></caption> tags.
The caption must emphasize the significance and role of these components
(and some description of each component) throughout the video. Your
caption should exclude unnecessary information such as "The scene unfolds
with a whimsical and heartwarming narrative, emphasizing the simple joys
of life through the Teddy Bear's endearing actions".

Example of a Descriptive Caption:
1. Film quality, professional quality, rich details. ...
2. A professional male diver performs an elegant diving maneuver ...

Prompt to Optimize:
{Insert your test prompt to optimize here}
```

---

*报告生成时间：2026-03-13 (更新)*
*基于论文 arXiv:2511.15700v1 完整阅读（含 14 页补充材料）*
