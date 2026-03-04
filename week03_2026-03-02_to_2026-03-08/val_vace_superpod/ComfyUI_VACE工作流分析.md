# ComfyUI VACE 商品替换工作流分析

> 参考来源：Bilibili 视频 BV1iG4yzTEwh 中使用的 ComfyUI 工作流配置
> 对比对象：本项目 4 个 PVTT 实验（precise_mask / bbox_mask / reactive_weight / reactive_weight_bbox）

---

## 一、工作流组件构成

ComfyUI 工作流使用 **WanVideoWrapper**（kijai）节点包，完整流程涉及以下核心组件：

### 1.1 模型加载与优化

| 组件 | 节点类型 | 配置 | 说明 |
|------|---------|------|------|
| VACE 模型 | `DownloadAndLoadWanVideoModel` | `wan2.1-vace-14B-fp8_e4m3fn.safetensors` | fp8 量化版本，显存占用大幅降低 |
| 文本编码器 | `DownloadAndLoadWanVideoTextEncoder` | `umt5-xxl-enc-bf16.safetensors` | UMT5-XXL，bf16 精度 |
| VAE | `DownloadAndLoadWanVideoVAE` | `wan_2.1_vae.safetensors` | 标准 VACE VAE |
| 蒸馏 LoRA | `WanVideoLoraSelect` | `Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors` | **LightX2V CFG+Step 蒸馏 LoRA**，strength = 1.0 |
| 注意力优化 | `WanVideoSageAttention` | `sageattn_varlen` | SageAttention 加速注意力计算 |

**关键发现：蒸馏 LoRA**

LightX2V 蒸馏 LoRA 是该工作流最显著的特点之一。它通过知识蒸馏将多步采样压缩为**仅 1 步推理**，同时蒸馏了 CFG 引导信息，使得单步采样也能产出高质量结果。这意味着：
- 推理速度提升数十倍（50 步 → 1 步）
- 蒸馏过程中已内化 CFG 引导，单步即可获得条件引导效果
- 可能改变了模型对条件输入（mask、参考图）的响应特性

### 1.2 输入预处理流程

这是工作流中**最关键的差异部分**，共涉及 3 条预处理路径：

#### 路径 A：参考图像预处理（ref_images）

```
原始参考图 → BiRefNetUltra（前景分割）→ 获取前景 mask
                                          ↓
                                  ImageCompositeMasked（中性色填充背景）
                                          ↓
                                  去背景参考图（用于 VACE 条件输入）
```

| 步骤 | 节点 | 参数 | 说明 |
|------|------|------|------|
| 前景分割 | `BiRefNetUltra` | 默认参数 | 高精度双目参考网络，自动提取商品前景 |
| 背景填充 | `ImageCompositeMasked` | 中性色（约灰色） | 将背景替换为纯色，消除参考图背景干扰 |

#### 路径 B：输入帧预处理（input_frames）

```
模板视频帧 + mask → INPAINT_MaskedFill（neutral 模式）
                         ↓
                  mask 区域被填充为中性灰色的视频帧
```

| 步骤 | 节点 | 参数 | 说明 |
|------|------|------|------|
| Mask 区域擦除 | `INPAINT_MaskedFill` | `fill = "neutral"` | **将 mask 区域的原始像素擦除为中性色** |

**这是最关键的预处理步骤**：输入视频帧中的 mask 区域不再保留原始物体像素，而是被中性灰色填充。这直接消除了 VACE reactive 流中模板物体的"先验"影响。

#### 路径 C：Mask 预处理

```
原始精确 mask → GrowMask（扩展 10px）→ BlockifyMask（32px 网格对齐）
                                              ↓
                                      处理后的 mask（用于 VACE 条件）
```

| 步骤 | 节点 | 参数 | 说明 |
|------|------|------|------|
| Mask 扩展 | `GrowMask` | `expand = 10` | 向外扩展 10 像素，覆盖边缘伪影 |
| 网格对齐 | `BlockifyMask` | `block_size = 32` | 对齐到 32×32 网格，匹配 VAE 下采样率 |

### 1.3 VACE 条件组装

```
处理后的 input_frames + 处理后的 mask + 处理后的 ref_images
                         ↓
              WanVaceVideoCondition
                         ↓
                   VACE 条件张量
```

节点 `WanVaceVideoCondition` 将三个输入组装为 VACE 所需的 `[Template; Filler; Mask]` 三元组条件。

### 1.4 采样配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `steps` | **1** | 仅 1 步采样（蒸馏 LoRA 支持） |
| `cfg` | 6.0 | CFG 引导强度 |
| `cfg_star` | 5.0 | Wan 专有双重 CFG 参数（后期去噪阶段使用较低引导强度） |
| `scheduler` | `dpm++_sde` | DPM++ SDE 调度器 |
| `shift` | `true` | 启用时间步偏移 |
| `denoise_strength` | 1.0 | 完全去噪 |

### 1.5 后处理

```
VACE 生成结果 + 原始视频帧 + mask → Composite（mask 合成）
                                        ↓
                                 最终输出视频
```

使用 mask 将生成内容（mask 内部）与原始视频帧（mask 外部）合成，确保非替换区域保持不变。

### 1.6 完整数据流总结

```
┌─────────────────────────────────────────────────────────────────┐
│                      ComfyUI VACE 工作流                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [参考图]──→ BiRefNetUltra ──→ 中性色填充背景 ──→ ref_images    │
│                                                                 │
│  [模板帧] + [mask] ──→ INPAINT_MaskedFill(neutral) ──→ frames  │
│                                                                 │
│  [精确mask] ──→ GrowMask(10px) ──→ BlockifyMask(32px) ──→ mask │
│                                                                 │
│  ref_images + frames + mask ──→ WanVaceVideoCondition           │
│                                        ↓                        │
│  [VACE 14B fp8] + [蒸馏LoRA] + [SageAttn]                      │
│         ↓                                                       │
│  Sampler (1 step, cfg=6, cfg_star=5, dpm++_sde)                │
│         ↓                                                       │
│  VAE Decode ──→ Composite with mask ──→ 最终视频                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、与本项目四个实验的配置区别

### 2.1 总体对比表

| 配置项 | ComfyUI 工作流 | 实验1: precise_mask | 实验2: bbox_mask | 实验3: reactive_weight | 实验4: rw_bbox |
|--------|---------------|-------------------|-----------------|---------------------|---------------|
| **模型** | 14B fp8 量化 | 1.3B / 14B（原始精度 bf16） | 同左 | 同左 | 同左 |
| **蒸馏 LoRA** | LightX2V（rank32） | 无 | 无 | 无 | 无 |
| **采样步数** | **1** | 50 | 50 | 50 | 50 |
| **CFG** | 6.0 | 7.5 | 7.5 | 7.5 | 7.5 |
| **cfg_star** | 5.0 | 无 | 无 | 无 | 无 |
| **embedded_cfg_scale** | — | — | — | 6.0 | 6.0 |
| **调度器** | dpm++_sde | DiffSynth 默认 | 同左 | 同左 | 同左 |
| **参考图去背景** | BiRefNetUltra + 中性色填充 | **无** | **无** | **无** | **无** |
| **输入帧 mask 区域处理** | 中性灰色填充（擦除原始像素） | **保留原始像素** | **保留原始像素** | Reactive 权重缩放 | Reactive 权重缩放 |
| **Mask 扩展** | GrowMask 10px | 无 | 无 | 无 | 无 |
| **Mask 网格对齐** | BlockifyMask 32px | 无 | 无 | 无 | 无 |
| **Mask 类型** | 精确 mask（处理后） | 精确 mask | Bbox mask | 精确 mask | Bbox mask |
| **注意力优化** | SageAttention | 无 | 无 | 无 | 无 |
| **后处理合成** | Composite with mask | Composite with mask | Composite with mask | Composite with mask | Composite with mask |

### 2.2 关键差异详解

#### 差异 1：参考图像预处理——去背景（影响程度：⭐⭐⭐⭐⭐）

**ComfyUI**：使用 BiRefNetUltra 自动分割参考图前景，将背景替换为中性灰色。VACE 接收到的参考图仅包含商品主体，无背景干扰。

**本项目**：直接将原始参考图（含背景）传入 pipeline。参考图中的背景元素会被模型一并"参考"，导致：
- 模型可能将背景纹理混入生成结果
- 商品主体特征被背景信息稀释
- 模型难以准确定位"需要参考的是什么"

#### 差异 2：输入帧 mask 区域处理——中性色填充 vs 保留原始像素（影响程度：⭐⭐⭐⭐⭐）

**ComfyUI**：使用 `INPAINT_MaskedFill(neutral)` 将输入帧的 mask 区域擦除为中性灰色。模板视频中原有物体的像素信息被完全移除。

**本项目（实验 1、2）**：直接将含原始物体的视频帧传入 pipeline。VACE 的 reactive 流会将这些像素作为强先验条件，导致模型倾向于**保留或重建原始物体**而非替换为参考商品。

**本项目（实验 3、4）**：虽然尝试通过 reactive weight 缩放来降低 mask 区域的像素强度，但这种做法：
- `weight=0` 虽然将像素置零（黑色），但黑色本身也是一种视觉信号，与"中性灰色"有本质区别
- 中间权重值（0.3~0.7）仍保留了部分原始物体信息
- 缺少 ComfyUI 使用的专门 inpaint 填充策略

#### 差异 3：Mask 预处理——GrowMask + BlockifyMask（影响程度：⭐⭐⭐）

**ComfyUI**：
- `GrowMask(10px)`：向外扩展 mask 边缘 10 像素，确保覆盖分割边界处的伪影和残留像素
- `BlockifyMask(32px)`：将 mask 对齐到 32×32 网格，匹配 VAE 编码器的空间下采样率（通常为 8× 或 32×），避免亚像素级的 mask 边界在潜空间中产生模糊

**本项目**：直接使用原始精确 mask 或 bbox mask，无任何后处理。可能导致：
- mask 边缘处出现"渗透"伪影（原始物体边缘残留）
- mask 与 VAE 潜空间网格不对齐，引入边界噪声

#### 差异 4：蒸馏 LoRA + 单步采样（影响程度：⭐⭐⭐⭐）

**ComfyUI**：使用 LightX2V CFG+Step 蒸馏 LoRA，仅需 1 步采样即可生成高质量结果。

**本项目**：使用标准 50 步采样，无蒸馏 LoRA。

蒸馏 LoRA 的影响不仅是速度：
- 蒸馏过程可能改变了模型对条件输入的权衡方式
- 单步采样减少了反复迭代中 reactive 先验的累积强化效应
- 蒸馏可能隐式提升了参考图像条件的相对权重

#### 差异 5：CFG 配置——cfg_star 双重引导（影响程度：⭐⭐）

**ComfyUI**：使用 `cfg=6.0` + `cfg_star=5.0` 的双重 CFG 策略。`cfg_star` 是 Wan 模型专有参数，在去噪后期阶段使用较低的 CFG 强度，避免过度引导导致的伪影。

**本项目**：使用单一 `cfg_scale=7.5`，无 `cfg_star`。较高的 CFG 值可能加剧模型对模板帧的过度忠实再现。

#### 差异 6：模型量化——fp8 vs bf16（影响程度：⭐）

**ComfyUI**：使用 fp8 (E4M3FN) 量化模型，配合 bf16 计算精度和 SageAttention。

**本项目**：使用原始 bf16 精度。

fp8 量化主要影响显存占用和速度，对生成质量的影响相对较小。但 SageAttention 可能改善注意力计算的数值稳定性。

---

## 三、四个实验无法实现目标替换的原因分析

### 3.1 根本原因：VACE Reactive 流的"模板锁定"效应

VACE 架构中，reactive 流（Template 通道）接收模板视频帧的像素信息。当 mask 区域保留原始物体像素时：

```
Reactive 流输入 = 完整的模板帧（含原始物体）
                     ↓
模型学到的 Reactive 语义 = "参考这些像素来生成 mask 区域内容"
                     ↓
生成结果 ≈ 原始物体的重建（而非参考商品的替换）
```

这就是所有实验中观察到的现象——**模型倾向于重建模板中的原始物体（茶壶），而非替换为参考商品（黄色橡皮鸭）**。

ComfyUI 工作流通过**中性色填充**打破了这一锁定：

```
Reactive 流输入 = mask 区域被灰色填充的帧
                     ↓
模型在 mask 区域没有强先验 → 更多依赖参考图和文本提示
                     ↓
生成结果 ≈ 参考商品的创意填充
```

### 3.2 各实验的具体失败原因

#### 实验 1：精确 Mask 实验

**失败原因**：
1. **Reactive 先验过强**：原始茶壶像素完整保留在输入帧中，模型直接重建茶壶
2. **参考图含背景干扰**：橡皮鸭参考图的背景信息稀释了商品特征
3. **Mask 未预处理**：边缘可能存在 VAE 网格不对齐问题

**现象**：生成结果基本是原始茶壶的重建，橡皮鸭特征几乎不可见。

#### 实验 2：Bbox Mask 实验

**失败原因**：
1. Bbox mask 虽然减弱了精确形状先验，但输入帧中**原始物体像素仍完整保留**
2. Bbox 扩大了 mask 区域，但反而给了模型更多的"原始像素参考空间"
3. 模型在更大的区域内重建原始场景，而非替换为参考商品

**现象**：与实验 1 类似，模型仍然重建原始物体，bbox 区域可能出现更多背景重建。

#### 实验 3：Reactive Weight 实验（精确 Mask）

**失败原因**：
1. `weight=0` 将 mask 区域像素置为**黑色**（全零），而非中性灰色。黑色是一种极端视觉信号，可能引导模型生成暗色内容
2. `weight=0.3~0.7` 仍保留了部分原始物体纹理信息，reactive 先验未被完全消除
3. `weight=1.0` 等同于实验 1，完全保留原始像素
4. 缺少参考图去背景步骤

**现象**：低权重可能产生暗色或失真的结果，中高权重仍重建原始物体。

#### 实验 4：Reactive Weight + Bbox Mask 实验

**失败原因**：
1. 结合了实验 2 和实验 3 的问题
2. Bbox mask + weight 缩放的组合虽然理论合理，但仍未解决核心问题——**缺少中性色填充和参考图去背景**
3. 黑色填充 + 大面积 bbox 可能产生大面积暗色区域，进一步误导模型

**现象**：可能出现大面积失真或不自然的生成结果。

### 3.3 失败原因优先级排序

| 优先级 | 原因 | 影响的实验 |
|--------|------|-----------|
| P0 | 输入帧 mask 区域未做中性色填充（reactive 先验过强） | 实验 1、2 |
| P0 | 参考图未去除背景（条件信号不纯净） | 全部 4 个 |
| P1 | 蒸馏 LoRA 缺失（模型对条件输入的响应特性不同） | 全部 4 个 |
| P1 | reactive weight 使用黑色（全零）而非中性灰色 | 实验 3、4 |
| P2 | Mask 未做 Grow + Blockify 预处理 | 全部 4 个 |
| P2 | CFG 参数差异（7.5 vs 6.0，缺少 cfg_star） | 全部 4 个 |
| P3 | 采样步数差异（50 vs 1） | 全部 4 个 |

---

## 四、新实验组设计建议

基于以上分析，建议设计以下**渐进式消融实验**，逐步对齐 ComfyUI 工作流的关键配置：

### 4.1 实验组概览

| 实验编号 | 名称 | 变量 | 目的 |
|----------|------|------|------|
| A1 | neutral_fill | 输入帧中性色填充 | 验证 neutral fill 的核心作用 |
| A2 | ref_nobg | 参考图去背景 | 验证参考图预处理的影响 |
| A3 | neutral_fill + ref_nobg | 组合 A1 + A2 | 验证两者协同效果 |
| A4 | mask_preprocess | Mask Grow + Blockify | 验证 mask 预处理的影响 |
| A5 | full_preprocess | A3 + A4 | 完整预处理对齐 |
| B1 | cfg_ablation | CFG/cfg_star 参数扫描 | 寻找最佳引导强度 |
| C1 | distill_lora | 加载蒸馏 LoRA + 单步采样 | 验证蒸馏对条件响应的影响 |
| C2 | full_pipeline | 全部对齐 | 完全复现 ComfyUI 工作流效果 |

### 4.2 各实验详细设计

#### 实验 A1：Neutral Fill（中性色填充）

**核心改动**：将输入帧的 mask 区域填充为中性灰色 `(128, 128, 128)`，替代保留原始像素。

```python
# 伪代码
for i, (frame, mask) in enumerate(zip(frames, masks)):
    mask_bool = np.array(mask) > 127
    frame_array = np.array(frame)
    frame_array[mask_bool] = 128  # 中性灰色填充
    frames[i] = Image.fromarray(frame_array)
```

**对照组**：实验 1（precise_mask，保留原始像素）
**预期**：这应该是**影响最大的单一改动**，可能直接使目标替换开始生效。

#### 实验 A2：Reference Image No-Background（参考图去背景）

**核心改动**：使用前景分割模型（如 rembg、BiRefNet）去除参考图背景，填充为中性色。

```python
# 伪代码（使用 rembg 库）
from rembg import remove
ref_image_nobg = remove(ref_image)  # RGBA
# 创建中性灰色背景
bg = Image.new('RGB', ref_image.size, (128, 128, 128))
bg.paste(ref_image_nobg, mask=ref_image_nobg.split()[3])
ref_image = bg
```

**对照组**：实验 1（precise_mask，原始参考图）
**预期**：参考图质量提升，但如果不配合 neutral fill，可能仍被 reactive 先验压制。

#### 实验 A3：Neutral Fill + Ref No-Background

**核心改动**：同时应用 A1 和 A2。

**对照组**：A1、A2
**预期**：两项关键预处理协同，应产出明显优于 A1、A2 单独使用的结果。

#### 实验 A4：Mask Preprocessing（Mask 预处理）

**核心改动**：对 mask 做 Grow（10px 膨胀）+ Blockify（32px 网格对齐）。

```python
# 伪代码
from scipy.ndimage import binary_dilation

def grow_mask(mask, pixels=10):
    struct = np.ones((2*pixels+1, 2*pixels+1))
    return binary_dilation(mask, structure=struct)

def blockify_mask(mask, block_size=32):
    h, w = mask.shape
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = mask[y:y+block_size, x:x+block_size]
            if block.any():
                mask[y:y+block_size, x:x+block_size] = True
    return mask
```

**对照组**：实验 1（原始 mask）
**预期**：改善 mask 边缘伪影，影响可能不如 A1/A2 显著。

#### 实验 A5：Full Preprocessing

**核心改动**：A1 + A2 + A4 全部预处理。

**对照组**：A3
**预期**：预处理维度上完全对齐 ComfyUI，应产出高质量替换结果。

#### 实验 B1：CFG Ablation（CFG 参数消融）

**核心改动**：在 A5 基础上，扫描 CFG 参数组合。

| 组合 | cfg_scale | cfg_star (embedded_cfg_scale) |
|------|-----------|------------------------------|
| B1-a | 6.0 | 5.0 |
| B1-b | 6.0 | — |
| B1-c | 7.5 | 5.0 |
| B1-d | 5.0 | 4.0 |

**对照组**：A5（cfg=7.5，无 cfg_star）
**预期**：较低的 cfg 可能减少过度引导伪影。

#### 实验 C1：Distill LoRA

**核心改动**：加载 LightX2V 蒸馏 LoRA，使用 1 步采样。

**注意**：需要在 DiffSynth-Studio 中实现 LoRA 加载，或切换到支持该功能的推理框架。

**对照组**：A5（50 步采样，无 LoRA）
**预期**：单步采样减少 reactive 先验的累积效应，同时大幅提速。

#### 实验 C2：Full Pipeline

**核心改动**：完全复现 ComfyUI 配置——A5 + B1 最优参数 + C1。

**预期**：如果前序实验设计正确，应当能复现 ComfyUI 的商品替换效果。

### 4.3 实验执行优先级

考虑到实验资源有限，建议按以下优先级执行：

```
第一轮（验证核心假设）：
  A1 (neutral fill) → A2 (ref nobg) → A3 (A1+A2)

第二轮（完善预处理）：
  A4 (mask preprocess) → A5 (full preprocess)

第三轮（参数调优）：
  B1 (CFG ablation)

第四轮（高级优化）：
  C1 (distill LoRA) → C2 (full pipeline)
```

**如果时间非常紧张，只做一个实验：直接做 A3**（neutral fill + ref nobg），这最有可能打破当前的"模板锁定"困局。

### 4.4 实现建议

#### 代码层面

1. **在 `utils.py` 中新增预处理函数**：
   - `neutral_fill_frames(frames, masks, fill_value=128)` —— 中性色填充输入帧
   - `remove_background(image, fill_value=128)` —— 参考图去背景
   - `grow_mask(mask, pixels=10)` —— Mask 膨胀
   - `blockify_mask(mask, block_size=32)` —— Mask 网格对齐

2. **新增实验脚本**：
   - `exp_neutral_fill.py` —— A1 实验
   - `exp_full_preprocess.py` —— A5 实验（集成所有预处理）

3. **依赖**：
   - 参考图去背景：`rembg` 或 `BiRefNet`（pip install rembg）
   - Mask 处理：`scipy`（已在大多数环境中可用）

#### 基础设施层面

1. 每个实验生成标准化输出（视频 + comparison + showcase），方便横向对比
2. 建议增加一个"预处理可视化"步骤，输出预处理后的帧、mask、参考图，便于调试
3. 考虑将分辨率降低到 384×672 用于快速迭代测试，确认方案可行后再恢复 480×848

---

## 五、总结

### 核心发现

ComfyUI 工作流的成功并非来自某个单一技巧，而是**输入预处理链的系统性设计**：

1. **参考图去背景** → 纯净的商品条件信号
2. **输入帧中性色填充** → 消除 reactive 流的模板锁定
3. **Mask Grow + Blockify** → 干净的空间边界
4. **蒸馏 LoRA + 单步采样** → 改变条件响应特性

本项目四个实验的核心问题在于**跳过了输入预处理**，直接将原始数据传入 VACE pipeline。这导致 reactive 流中的模板物体先验过强，模型始终倾向于重建原始物体而非替换为参考商品。

### 下一步行动

1. **立即可做**：实现 A1（neutral fill）实验，仅需修改 `exp_precise_mask.py` 中数行代码
2. **短期目标**：完成 A3（neutral fill + ref nobg）实验，验证预处理的核心价值
3. **中期目标**：完成完整消融实验（A1~A5 + B1），形成系统化的实验报告
4. **长期目标**：探索蒸馏 LoRA 路线（C1、C2），进一步提升质量和效率
