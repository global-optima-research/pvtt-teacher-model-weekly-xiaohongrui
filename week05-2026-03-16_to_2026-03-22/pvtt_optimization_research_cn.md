# PVTT 优化方向深度调研：从 FFGo 复现到论文级创新

**调研日期**：2026-03-18
**背景**：当前直接复用 FFGo（Wan2.2-I2V-A14B + LoRA）在 PVTT 数据集上已取得不错效果（MFS=0.992, ProdCLIP=0.730），本文档从实验中发现的问题出发，深入调研可优化的方向。

---

## 目录

1. [方向一：首帧目标去除质量问题](#方向一)
2. [方向二：从训练角度解决不完美去除（双条件 CFG 训练）](#方向二)
3. [方向三：产品身份保持增强](#方向三)
4. [方向四：VLM 驱动的提示词优化](#方向四)
5. [方向五：训练数据与 LoRA 微调策略](#方向五)
6. [方向六：生成更长时间的视频](#方向六)
7. [综合：论文框架与贡献组合](#综合)

---

## 方向一：首帧目标去除质量问题 {#方向一}

### 1.1 问题描述

当前使用 LaMa/cv2 进行首帧物体去除时存在以下问题：

1. **残留轮廓**：擦除后留下糊糊的轮廓（如项链的两条轮廓、手表的表盘轮廓），模型会把这些当作画面内容保留或还原
2. **过度去除/不足去除**：有时把不该去的也去掉，有时又去不干净
3. **影响下游生成**：模型根据首帧右半部分理解场景，残留伪影会误导生成（如把糊糊的轮廓当成需要保持的画面元素）

### 1.2 SOTA Inpainting 模型（可直接替换 LaMa）

| 模型 | 发表 | 核心优势 | 局限 |
|------|------|---------|------|
| **FLUX Fill** | 2025, Black Forest Labs | 质量最高，SD3 架构 | 模型大（~12B），推理慢 |
| **PowerPaint-V2** | ECCV 2024 | 支持多种 inpainting 模式（去除/填充/修复），质量高 | 需要 SD1.5/SDXL 基座 |
| **BrushNet** | ECCV 2024 | 即插即用双分支，可接入任意 SD 模型 | 上下文保持优秀但不如 FLUX |
| **OmniPaint** | ICCV 2025, Adobe | 统一去除和插入，CycleFlow 大规模无配对训练 | 较新，代码可能不完整 |
| **LanPaint** | 2025 | 免训练，理论保证，可接入任意 SD 模型 | 速度较慢 |
| **Token Painter** | 2025 | MAR 架构（非扩散），上下文保持强 | 非主流架构 |

**链接**：
- PowerPaint：https://powerpaint.github.io/
- BrushNet：[arXiv:2403.06976](https://arxiv.org/abs/2403.06976)
- OmniPaint：[arXiv:2503.08677](https://arxiv.org/abs/2503.08677)
- LanPaint：[arXiv:2502.03491](https://arxiv.org/abs/2502.03491)
- Token Painter：[arXiv:2509.23919](https://arxiv.org/abs/2509.23919)

### 1.3 我们的优化思路

问题链条：
1. cv2/LaMa 擦除痕迹太明显 → 残留轮廓被模型当作画面内容保留到生成视频中
2. 可以用更强的图像编辑模型（如 FLUX Fill）解决，但引入 ~12B 的大模型使工作流变得繁重，且不是端到端的
3. **真正的优化方向**：让模型**既能接受干净的去除画面，又能接受物品完全没有被去除的原始画面**，两者都能生成质量相近的 PVTT 结果。这样就彻底不依赖图像编辑模型了

这个思路的好处：
- 推理时不需要任何 inpainting 步骤，直接用原始首帧即可
- 端到端，工作流简洁
- 是真正的论文贡献（见方向二的详细方案）

### 1.4 工程层面的备选

- 直接替换 LaMa → FLUX Fill / PowerPaint-V2 可以改善训练数据质量，但**仅作为工程改进，不是论文贡献**

---

## 方向二：从训练角度解决不完美去除（双条件 CFG 训练） {#方向二}

### 2.1 你的想法

> "训练时用 FLUX 完美去除物品训一次，完全不去除物品再训一次，类似 CFG 那样学习"

### 2.2 理论基础

#### Classifier-Free Guidance（Ho & Salimans, 2022）
- **论文**：[arXiv:2207.12598](https://arxiv.org/abs/2207.12598)
- **核心机制**：训练时随机丢弃条件（10-20% 概率），让模型同时学习有条件和无条件生成。推理时外推：`output = unconditional + w × (conditional - unconditional)`
- **直接支持**你的双条件方案

#### "Unconditional Priors Matter!"（2025 年 3 月）
- **论文**：[arXiv:2503.20240](https://arxiv.org/abs/2503.20240)
- **关键发现**：LoRA 微调时联合学习无条件噪声会降低质量。**简单地用基座模型的无条件预测替换微调模型的无条件预测**，即可显著提升条件生成质量
- **与你的想法的关系**：这篇论文验证了一个更简单的方案——不需要双条件训练，只需在推理时用基座 Wan2.2 的无条件预测作为 CFG 的"负方向"

#### DynamiCrafter 的双条件 CFG
- DynamiCrafter（text+image-to-video）使用双条件 CFG，分别对文本和图像条件进行 dropout
- 用 base VideoCrafter 模型的无条件噪声替换微调模型的无条件噪声
- **验证了**在视频生成中使用不同无条件先验是可行的

### 2.3 可行性分析：你的方案完全可行

我评估了三种实现路径：

#### 方案 A：朴素双条件训练（你的原始想法）

**实现**：
- 训练数据中每个样本准备两个版本的首帧：
  - Version A：用 FLUX Fill 完美去除物品的首帧
  - Version B：完全不去除物品的原始首帧
- 训练时随机选择：15% 概率丢弃条件 → 10% 概率用完美去除首帧 → 50% 概率用原始首帧 → 25% 概率用某种中间质量的去除
- 推理时用 CFG 引导向"完美去除"方向

```python
# 训练伪代码
def training_step(batch):
    r = random.random()
    if r < 0.15:
        first_frame = null_condition        # 无条件
    elif r < 0.50:
        first_frame = batch['clean']        # FLUX 完美去除
    else:
        first_frame = batch['raw']          # 原始（不去除）

    # 标准扩散训练
    noise = torch.randn_like(batch['target_video'])
    t = random.randint(0, T)
    loss = F.mse_loss(model(add_noise(batch['target_video'], noise, t), first_frame, t), noise)
    return loss

# 推理伪代码
def inference(first_frame_imperfect, guidance_scale=7.5):
    noise_uncond = model(x_t, null_condition, t)
    noise_cond = model(x_t, first_frame_imperfect, t)
    return noise_uncond + guidance_scale * (noise_cond - noise_uncond)
```

- **可行性**：高。标准 CFG 训练
- **风险**：LoRA 可能难以区分"完美去除"和"不去除"条件的细微差异
- **新颖性**：**高**——无人在视频生成的 inpainting 鲁棒性上做过此类工作

#### 方案 B：简化版——Unconditional Priors 替换（推荐）

**实现**：
- LoRA 仅在 FLUX 完美去除的首帧上训练（单条件）
- 推理时用基座 Wan2.2 模型的无条件预测替换 LoRA 模型的无条件预测

```python
# 推理时
noise_uncond = base_wan22_model(x_t, null_condition, t)    # 用基座模型
noise_cond = lora_model(x_t, first_frame_imperfect, t)     # 用 LoRA 模型
guided = noise_uncond + w * (noise_cond - noise_uncond)
```

- **可行性**：非常高。实现简单，仅改推理代码
- **风险**：低。"Unconditional Priors Matter!" 已在多个模型上验证
- **新颖性**：中。应用已有技术到新问题域

#### 方案 C：渐进质量条件训练（最新颖）

**实现**：
- 构造一个首帧质量谱：[原始] → [LaMa 去除] → [BrushNet 去除] → [FLUX 去除] → [完美干净]
- 训练时从质量谱中随机采样作为条件
- 推理时通过 CFG 引导向"完美干净"端

- **可行性**：中。需要为每个训练样本生成多种质量级别的去除结果
- **风险**：中。训练数据流水线更复杂
- **新颖性**：**非常高**——渐进质量条件训练从未被探索过

### 2.4 疑问解答：方案 B 和 FFGo 有什么区别？

> **问**：Unconditional Priors 替换和 FFGo 有区别吗？我的理解是它们都在完美去除的首帧上训练，没什么区别啊？

**回答**：好问题。我已经阅读了 FFGo 的源码（`FFGO-Video-Customization/`），下面是分析。

#### FFGo 源码中的 CFG 实现分析

**关键发现：FFGo 的 LoRA 是永久合并到基座模型权重中的。**

在 `wan/utils/utils.py` 的 `load_and_merge_lora_weight` 函数中：
```python
delta_W = scaling_factor * torch.matmul(lora_up, lora_down)
value.data = value.data + delta_W  # 直接修改模型权重
```

在 `VideoX-Fun/examples/wan2.2/single_predict.py` 中：
```python
pipe = merge_lora(pipe, lora_low, lora_weight, device=device)
pipe = merge_lora(pipe, lora_high, lora_high_weight, ...)
```

CFG 的实际执行在 `wan/image2video.py`：
```python
noise_pred_cond = model(latent_model_input, t=timestep, **arg_c)     # 有条件预测
noise_pred_uncond = model(latent_model_input, t=timestep, **arg_null) # 无条件预测
noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)
```

**两次调用用的是同一个 model**——已经合并了 LoRA 的模型。也就是说，FFGo 的无条件预测也经过了 LoRA，**没有使用基座模型的 Unconditional Priors**。

#### 结论

FFGo 当前的实现中，条件预测和无条件预测**都经过 LoRA 合并后的模型**。这意味着：
- 方案 B（Unconditional Priors 替换）**确实有增量价值**——当前 FFGo 没有这么做
- 实现方式：在推理时，保留一份未合并 LoRA 的基座 Wan2.2 模型用于无条件预测
- 但这也意味着需要同时加载两份模型（显存翻倍），实用性受限

回到你的原始问题——区别在于 **推理时 CFG 的"负方向"用什么**：

| | FFGo 现状 | 方案 B（Unconditional Priors 替换） |
|---|---|---|
| **训练** | 在完美首帧上训 LoRA + 标准 CFG dropout | 相同（无区别） |
| **推理时无条件预测** | 用 LoRA 模型自己的无条件预测 | 用**基座 Wan2.2（未加载 LoRA）**的无条件预测 |
| **CFG 公式** | `ε_lora_uncond + w × (ε_lora_cond - ε_lora_uncond)` | `ε_base_uncond + w × (ε_lora_cond - ε_base_uncond)` |

为什么这有区别？"Unconditional Priors Matter!" 论文证明了：LoRA 微调时，由于训练数据量小（FFGo 仅 20-50 个视频），LoRA 模型学到的无条件分布质量较差——它"见过的世界太小"。而基座 Wan2.2 在海量数据上训练，它的无条件先验更加丰富和准确。

**打个比方**：CFG 的原理是"沿着条件方向远离无条件方向"。如果无条件方向本身就不靠谱（LoRA 的无条件先验太弱），那引导效果会打折扣。用基座模型的强无条件先验作为"锚点"，引导会更稳定。

**但是**，你说得对——如果当前 FFGo 的推理代码已经在用基座模型做无条件预测（有些 LoRA 推理框架默认这么做），那方案 B 确实和现状没有区别。需要检查 FFGo 的 `single_predict.py` 中 CFG 具体是怎么实现的。如果它已经用了基座无条件先验，那方案 B 没有额外收益，应该直接跳到方案 A 或 C。

**结论**：方案 B 的价值取决于 FFGo 当前的 CFG 实现。但无论如何，方案 A（双条件训练）和方案 C（渐进质量谱）都是真正的新贡献——它们让模型**在训练阶段就见过不完美的去除**，从而学会处理残留伪影，这是 FFGo 完全没有的。

### 2.5 结论

**方案 A（你的原始想法）完全可行**，且新颖性高。建议先检查 FFGo 的 CFG 实现确认方案 B 是否有增量价值，然后直接实现方案 A 或 C 作为论文贡献。

---

## 方向三：产品身份保持增强 {#方向三}

### 3.1 问题描述

当前 FFGo 的产品身份保持依赖模型的"涌现能力"——从首帧中隐式参考产品外观。对于细节丰富的产品（如海马形项链坠、复杂表盘），这种隐式参考经常不够。

### 3.2 参考架构

#### ConsisID（CVPR 2025 Highlight）
- **论文**：[arXiv:2411.17440](https://arxiv.org/abs/2411.17440)
- **核心**：将面部特征在频域中分解为低频全局特征和高频内在特征，分别注入 DiT 的不同层
- **启示**：将此思路应用于产品——低频 = 产品整体形状/轮廓，高频 = 纹理/logo/细节。**ConsisID 只做人脸，将其扩展到通用产品是新的贡献**

#### MimicBrush（NeurIPS 2024，阿里巴巴）
- **论文**：[arXiv:2406.07547](https://arxiv.org/abs/2406.07547)
- **核心**：双 U-Net 架构——参考 U-Net 提取参考图特征，通过 K/V 注入模仿 U-Net 的注意力层
- **启示**：可设计一个产品参考 DiT 分支，将产品特征通过 K/V 注入生成 DiT 的每一步

#### MagicMirror（ICCV 2025）
- **论文**：[arXiv:2501.03931](https://arxiv.org/abs/2501.03931)
- **核心**：双分支特征提取器（身份 + 结构）+ 条件自适应归一化 + 两阶段训练。专为 DiT 架构设计
- **启示**：双分支映射到产品：身份分支（产品长什么样）+ 结构分支（产品应该出现在哪里/怎样出现）

#### MAGREF（ICLR 2026，字节跳动）
- **论文**：[arXiv:2505.23742](https://arxiv.org/abs/2505.23742)
- **核心**：区域感知掩码引导 + 像素级通道拼接 + 主体解耦。**不修改骨干架构**
- **启示**：不修改 Wan2.2 骨干的方案最容易与 FFGo LoRA 结合

#### Kaleido（2025 年 10 月）
- **论文**：[arXiv:2510.18573](https://arxiv.org/abs/2510.18573)
- **核心**：基于 Wan2.1 构建，使用 Reference Rotary Positional Encoding (R-RoPE) 集成多参考图
- **启示**：**与我们的 Wan2.2 架构同源**，R-RoPE 是一种干净的参考图集成方式

### 3.3 推荐实现方案

#### 方案 A：轻量产品身份编码器（低风险，中新颖性）
- 用冻结的 DINO-v2 提取产品图特征
- 通过额外的交叉注意力层注入 Wan2.2 DiT blocks（IP-Adapter 风格）
- 仅训练交叉注意力投影层 + FFGo LoRA
- 约增加 5-10M 参数

#### 方案 B：频率分解产品注入（中风险，高新颖性）
- 受 ConsisID 启发：将产品特征分解为全局形状（低频）和纹理细节（高频）
- 低频特征注入 DiT 浅层，高频特征注入 DiT 深层
- **新颖性高**：ConsisID 只做人脸，扩展到产品是全新的

#### 方案 C：区域感知掩码引导（低风险，中新颖性）
- 借鉴 MAGREF：沿通道维度拼接产品参考特征
- 用区域掩码指定产品应出现的位置
- 不修改 Wan2.2 骨干架构，可直接与 FFGo LoRA 结合

### 3.4 疑问解答：不用 mask 的注入方法怎么知道注入到画面哪个区域？

> **问**：DINO 编码器 / 频率分解注入不需要 mask，那它们怎么知道产品特征应该注入到画面的哪个区域？

**回答**：这些方法不需要显式指定"注入到哪个像素位置"，因为它们工作在**注意力层**而非像素层。

**原理**：注入的产品特征通过 cross-attention 与生成过程中的 latent 交互。模型自己通过注意力机制决定"哪些空间位置需要参考这些产品特征"。具体来说：

```
产品参考图 → DINO 编码 → 产品特征向量 (K_prod, V_prod)
                                    ↓
DiT 生成过程中，每个空间位置的 Query 自动与 K_prod 计算注意力权重：
  - 产品区域的 Query 与 K_prod 相似度高 → 注意力权重大 → 大量吸收产品特征
  - 背景区域的 Query 与 K_prod 相似度低 → 注意力权重小 → 几乎不受影响
```

**类比 IP-Adapter 的工作方式**：IP-Adapter 注入参考图特征时也不需要 mask。你给它一张猫的参考图，生成"猫坐在沙发上"，模型自动把猫的特征应用到猫的区域，而不会把沙发也变成猫纹理——这是注意力机制的自然行为。

**为什么这对产品有效**：
1. FFGo 的首帧已经告诉模型"产品在画面的某个位置"
2. 生成过程中，产品区域的 latent 表征自然与产品参考特征更匹配
3. cross-attention 自动将产品特征路由到对应的空间位置
4. 背景区域的 latent 表征与产品特征不匹配，注意力权重接近零

**但存在边界情况**：如果产品非常小（如小耳钉），注意力可能过于分散，注入效果减弱。此时可以引入**软位置提示**（soft positional hint）：
- 在训练数据中标注产品的大致位置（bbox）
- 将位置信息编码为额外的 positional embedding 加到 Q/K 上
- 这比 VACE 的硬 mask 宽容得多——bbox 不精确也没关系，只是帮注意力更聚焦

### 3.5 疑问解答：与 FFGo 首帧参考是否冲突？

> **问**：FFGo 是参考首帧，而上面的方法都是在生成的每一步注入指导，它们会冲突吗？

**回答**：不会冲突，反而是互补关系。原因如下：

**FFGo 的首帧参考机制是"软约束"**：模型通过首帧 mask=0（保留首帧）来参考产品外观，但这种参考随着帧数增加会自然衰减——模型在生成第 50、60、80 帧时对首帧的"记忆"越来越弱。这正是为什么有些实验中后半段视频的产品外观会偏离。

**注意力注入是"持续约束"**：在每一步 denoising 中都注入产品特征，相当于在整个生成过程中持续"提醒"模型产品长什么样。

两者的关系是**递进式**的：

```
FFGo 首帧参考     →  提供初始上下文（产品+场景整体布局）
+ 注意力注入      →  持续强化产品细节（纹理/logo/形状）
= 首帧定全局 + 注入保细节
```

**类比**：就像画画时，FFGo 首帧提供了"参考照片放在旁边"，注意力注入则是"每画一笔都再看一眼参考照片上的产品细节"。两者不矛盾，前者定大方向，后者补细节。

**实际上 ConsisID、MagicMirror 等方法都是在 I2V 模型上加注入的**——它们的 I2V 模型本身也有首帧参考机制，加了身份注入后效果更好，而非更差。

**需要注意的是**：注入强度需要调节。如果注入太强，可能会让模型过度关注产品而忽视场景/运动，导致画面不自然。通常通过一个 `injection_scale` 超参数控制，需要通过实验找到最佳平衡点。

### 3.5 疑问解答：是否还会遇到 VACE 掩码实验的老问题？

> **问**：方案 C 需要 mask，那 VACE 实验中的问题（mask 过大、物体纠缠、语义冲突）还会困扰吗？

**回答**：这个问题需要区分两种完全不同的 mask 用途：

| | VACE 掩码替换 | FFGo + 身份注入 |
|---|---|---|
| **mask 的作用** | 定义"哪些像素由模型重新生成，哪些保留源视频" | 仅作为"产品区域的大致位置提示"（可选） |
| **mask 精度要求** | 极高——mask 边界直接决定合成质量 | 低——粗略 bbox 或根本不用 |
| **大 mask 问题** | 致命——mask 过大 = 丢失原视频上下文 | 不存在——FFGo 是全帧生成，不存在"保留区域" |
| **物体纠缠问题** | 严重——mask 覆盖周围物体 | 不存在——FFGo 全帧重新生成，不做局部替换 |
| **语义冲突问题** | 严重——斜挎包→手提包姿势不匹配 | 缓解——FFGo 自由生成合理姿势 |

**详细分析**：

1. **mask 过大覆盖太多区域**：在 FFGo 范式下，**根本不存在这个问题**。FFGo 是从首帧出发全帧重新生成整个视频，不存在"源视频被 mask 遮挡后无法重建"的问题。方案 A 和 B（DINO 编码器 / 频率分解注入）完全不需要 mask。方案 C（MAGREF 风格区域引导）使用的 mask 仅是"告诉模型产品大概在哪里"的提示，即使 mask 不准确，也只是注入位置偏了一点，不会导致画面崩溃。

2. **物体纠缠**：FFGo 全帧生成，模型自己决定画面构图。如果源视频中项链和模特颈部纠缠，FFGo 不会被这个困扰——它从首帧（产品图+干净背景）出发，自由生成一个合理的产品展示画面。纠缠问题是掩码替换范式的固有问题，重新生成范式天然避免。

3. **语义冲突**：FFGo 自由生成画面，模型会根据产品类型和 prompt 自动选择合理的展示方式。如果 prompt 说"手提包"，模型不会生成"背着斜挎包"的姿势。不过，FFGo 生成的画面可能与源视频的场景差异较大（这是 FFGo 的固有 trade-off）。

**总结**：VACE 掩码实验的三大问题（大 mask、纠缠、语义冲突）在 FFGo 范式下**基本不存在或大幅缓解**。即使加了方案 C 的区域引导 mask，它的作用和 VACE 的 mask 本质不同——前者是"提示"，后者是"硬约束"。

---

## 方向四：VLM 驱动的提示词优化 {#方向四}

### 4.1 问题描述

实验中发现部分产品身份保持不佳的案例（如手表）是因为**提示词错误或不精确**。

### 4.2 相关工作

#### Prompt-A-Video（2024 年 12 月）
- **论文**：[arXiv:2412.15156](https://arxiv.org/abs/2412.15156)
- **核心**：两阶段优化——(1) 奖励引导的 prompt 进化 + LLM SFT；(2) 多维度奖励的成对 DPO 对齐
- **启示**：可以训练一个产品视频专用的 prompt 优化器

#### VIVA（2025 年 12 月）
- **论文**：[arXiv:2512.16906](https://arxiv.org/html/2512.16906)
- **核心**：VLM 作为 instructor，将文本指令、源视频首帧、参考图编码为视觉锚定的指令表征
- **启示**：VLM-as-instructor 范式可用于产品视频模板生成

#### JoyCaption
- **链接**：https://github.com/fpgaminer/joycaption
- **核心**：开源 VLM，专为生成扩散模型训练数据的 caption 而设计
- **用途**：为训练数据生成高质量产品描述

### 4.3 疑问解答：Prompt-A-Video 如何集成到 PVTT？

> **问**：Prompt-A-Video 可以训练产品视频专用的 prompt 优化器，详细讲解并指导如何集成。

**Prompt-A-Video 的工作原理**：

Prompt-A-Video 解决的问题是：用户写的 prompt 往往对视频生成模型来说不够好——太短、缺少关键细节、或者表达方式不是模型偏好的格式。它通过两阶段自动优化 prompt：

**阶段 1：奖励引导的 Prompt 进化 + LLM SFT**

```
输入：用户的简单 prompt（如"A bracelet on a display stand"）
  ↓
LLM（如 Llama）生成 N 个候选 prompt 变体
  ↓
对每个变体，用视频生成模型实际生成视频
  ↓
多维度奖励函数评估生成视频质量：
  - 画面美学得分（Aesthetic Score）
  - 图文对齐度（CLIP Score）
  - 时序一致性（MFS）
  - 运动自然度
  ↓
选出高奖励的 prompt 作为正样本
  ↓
用这些正样本 SFT 微调 LLM → 得到 Stage-1 prompt optimizer
```

**阶段 2：DPO 对齐**

```
用 Stage-1 optimizer 生成更多 prompt 对
  ↓
对每对 prompt，生成视频并评估
  ↓
构造偏好对：(好 prompt, 差 prompt)
  ↓
用 DPO（Direct Preference Optimization）进一步对齐 LLM
  ↓
最终得到与视频模型偏好对齐的 prompt optimizer
```

**集成到 PVTT 的具体步骤**：

1. **定义产品视频专用的奖励函数**：
   ```python
   def pvtt_reward(generated_video, product_ref_image, target_prompt):
       clip_tgt = clip_similarity(generated_video, target_prompt)  # 图文对齐
       prod_clip = product_clip_similarity(generated_video, product_ref_image)  # 产品匹配
       mfs = mean_frame_similarity(generated_video)  # 时序一致
       aesthetic = aesthetic_score(generated_video)  # 画面美学

       # 加权组合，产品匹配权重最高
       return 0.3 * clip_tgt + 0.4 * prod_clip + 0.2 * mfs + 0.1 * aesthetic
   ```

2. **构造训练数据**：
   - 对 PVTT 数据集的每个任务，用 VLM 生成 5-10 个不同风格的 prompt 变体
   - 用 FFGo 实际生成视频
   - 用上述奖励函数评估每个变体
   - 选出最佳和最差 prompt 作为 DPO 训练对

3. **微调轻量 LLM**：
   ```
   输入：product_description + scene_description + source_video_caption
   输出：优化后的 target_prompt（适合 FFGo 视频生成模型）
   ```

4. **推理时 pipeline**：
   ```
   产品参考图 → VLM 分析 → product_description
   源视频 → VLM 分析 → scene_description + camera_movement
   ↓
   Prompt Optimizer LLM(product_description, scene_description)
   ↓
   优化后的 prompt → "ad23r2 the camera view suddenly changes. {optimized_prompt}"
   ↓
   FFGo 生成视频
   ```

**实现难度**：中等。主要工作量在于：(1) 用 FFGo 为每个 prompt 变体实际生成视频（GPU 密集）；(2) 构造足够多的偏好对（至少 1000+）。但 LLM 微调本身很轻量（7B 模型 DPO 微调仅需几小时）。

**新颖性**：中等。Prompt-A-Video 本身不新，但将其应用于产品视频模板转换（结合产品匹配奖励函数）是新的。作为支撑贡献比较合适。

---

## 方向五：训练数据与 LoRA 微调策略 {#方向五}

### 5.1 FFGo 的训练数据

FFGo 使用约 20-50 个精选视频做 LoRA 训练，训练数据构造方式：首帧拼贴（左=白底主体，右=场景）配对真实视频。使用特殊 token `ad23r2` 触发转场行为。

### 5.2 关键训练策略论文

#### DreamVideo（CVPR 2024）
- **论文**：https://dreamvideo-t2v.github.io/
- **核心**：解耦主体学习和运动学习。主体学习用 textual inversion + identity adapter，运动学习用独立的 motion adapter
- **启示**：可以设计解耦的 adapter——产品身份 adapter + 场景运动 adapter

#### Noise Consistency Regularization（CVPR 2025 Workshop）
- **论文**：[arXiv:2506.06483](https://arxiv.org/abs/2506.06483)
- **核心**：两个辅助一致性损失——(1) 先验一致性：非主体图像的噪声预测与预训练模型保持一致；(2) 主体一致性：对乘性噪声的鲁棒性。超越 DreamBooth 的 CLIP 分数、背景多样性和视觉质量
- **启示**：可直接加到 FFGo 的 LoRA 训练中，提升产品身份保持同时维持背景多样性

#### Subject-driven Disentangled（2025 年 4 月）
- **论文**：[arXiv:2504.17816](https://arxiv.org/abs/2504.17816)
- **核心**：零样本，不需要逐主体微调。随机 token dropping 防止 copy-paste 伪影
- **启示**：随机 token dropping 是有用的正则化策略

### 5.3 推荐训练策略

#### 策略一：产品专用数据增强
- 收集 100-200 个高质量产品宣传视频（vs FFGo 的 20-50）
- 用 JoyCaption/InternVL 生成详细产品描述
- 为每个视频生成多种质量的首帧去除版本（用于方向二）

#### 策略二：多阶段 LoRA 训练
- 阶段 1：在通用视频对上训练基础转场 LoRA（大数据集，通用转场行为）
- 阶段 2：在产品视频上微调产品身份 LoRA（小数据集，产品感知损失）
- 阶段 3：可选——少样本逐产品微调（3-5 个样本）

#### 策略三：产品感知损失函数
- 标准扩散损失 + DINO 产品相似度损失 + CLIP 图文对齐损失
- 产品区域损失：在产品应出现的区域施加更高的损失权重
- 时序身份一致性损失：确保产品特征在生成帧间保持一致

---

## 方向六：生成更长时间的视频 {#方向六}

### 6.1 问题描述

Wan I2V 单次生成上限为 81 帧（约 5 秒 @16fps）。产品宣传视频通常需要 15-60 秒，单次生成远远不够。

### 6.2 FFGo 当前能力下的长视频方案

#### 方案 A：滑动窗口自回归（最直接）

```
首帧(FFGo拼贴) → 生成 81 帧(第1段)
                    ↓ 取最后1帧作为新的"首帧"
                  → 生成 81 帧(第2段)
                    ↓ 取最后1帧作为新的"首帧"
                  → 生成 81 帧(第3段)
                  → ...拼接所有段
```

**优点**：实现简单，每段首帧都是自然画面（不需要转场）
**缺点**：
- 每一段的首帧参考越来越远离原始 FFGo 拼贴首帧
- **产品身份会逐段衰减**——第 3 段可能已经"忘了"产品长什么样
- 段间可能出现风格/运动不连续

#### 方案 B：每段都用 FFGo 拼贴首帧（保持产品参考）

```
FFGo拼贴首帧(产品+场景) → 第1段(81帧)
FFGo拼贴首帧(产品+场景) → 第2段(81帧，不同prompt描述后续动作)
FFGo拼贴首帧(产品+场景) → 第3段(81帧，不同prompt描述后续动作)
→ 视频平滑拼接（交叉淡入淡出）
```

**优点**：每段都有产品参考，身份不衰减
**缺点**：
- 每段都要经历转场（浪费帧）
- 段间场景/运动不连续，需要后处理
- 每段的"场景参考"是同一张去除物品的首帧，变化有限

#### 方案 C：重叠窗口 + 噪声混合（VideoPainter 风格）

```
第1段：帧 1-81
第2段：帧 61-141（与第1段重叠20帧）
第3段：帧 121-201（与第2段重叠20帧）
→ 重叠区域用噪声/latent 混合保证连续性
```

**优点**：段间连续性好
**缺点**：实现复杂，需要修改推理代码；重叠区域的产品外观可能不一致

### 6.3 长视频与产品身份注入的关系——为什么注入方法更有说服力

> **问**：如果随着时间变长首帧参考力度变小，我们的"每一步注入"就更有说服力，对吗？

**完全正确。这是一个关键洞察。**

FFGo 的首帧参考机制本质上是一种**衰减信号**：

```
首帧参考强度
  ↑
1.0│█████
   │    ████
   │        ███
   │           ██
   │             ██
   │               █
   │                █
   │                 █
   │                  █▁▁▁▁▁▁▁▁▁▁  → 长视频后半段几乎无参考
   +───────────────────────────────→ 帧序号
   0    20    40    60    80   160   240
        ← 5秒 →         ← 15秒 →
```

在短视频（5 秒 / 81 帧）中，衰减问题还不严重（实验中 ProdPersist=0.952 说明短视频内产品一致性较好）。但一旦扩展到 15 秒以上：
- **方案 A（滑动窗口）**：首帧参考在第 2 段开始就完全丢失
- **方案 B（每段重新拼贴）**：每段有参考但段间不连续

而**产品身份注入**提供的是**恒定强度的参考信号**：

```
注入参考强度
  ↑
1.0│████████████████████████████████████████
   │
   │    无论视频多长，每一步都有同等强度的产品特征注入
   │
   +───────────────────────────────────────→ 帧序号
   0    20    40    60    80   160   240
```

**这就是我们方法的核心卖点**：
- 短视频：FFGo 首帧已经够用，注入带来的增量改善可能不明显
- **长视频：FFGo 首帧参考衰减 → 注入成为唯一的产品身份保持信号 → 我们的方法显著优于 FFGo**

**论文实验设计建议**：
1. 分别生成 5 秒（81 帧）、10 秒（161 帧）、15 秒（241 帧）的视频
2. 测量 ProdCLIP 随帧数的衰减曲线
3. 展示 FFGo（无注入）在长视频中产品身份急剧衰减，而 Ours（有注入）保持稳定
4. 这将是论文中最有说服力的消融实验

### 6.4 我们的方法生成长视频的方案

有了产品身份注入，长视频方案变得更可行：

#### 推荐方案：滑动窗口 + 持续注入

```
FFGo拼贴首帧 → 第1段(81帧) + 产品DINO特征持续注入
                  ↓ 取最后1帧
                → 第2段(81帧) + 产品DINO特征持续注入  ← 注入保证产品不丢失
                  ↓ 取最后1帧
                → 第3段(81帧) + 产品DINO特征持续注入
                → 拼接
```

- 第 1 段有 FFGo 首帧 + 注入 → 最强参考
- 第 2、3 段没有 FFGo 首帧，但有注入 → 产品身份仍然得到保持
- 相比无注入的方案 A，产品不会逐段衰减

#### 进阶方案：RIFLEx 位置编码扩展

Wan2.2 支持 RIFLEx（Rotary Interpolation for Fixed Length Extrapolation），可以在不重新训练的情况下外推更长序列：

```python
# FFGo 的 single_predict.py 已经有 riflex 支持
if enable_riflex:
    pipeline.transformer.enable_riflex(k=riflex_k, L_test=latent_frames)
```

- 理论上可以直接生成 161 帧或更多
- 但质量可能随长度下降（RIFLEx 是外推，非精确）
- 配合产品注入可以缓解外推带来的身份衰减

#### 对论文的意义

长视频生成可以作为**论文的额外实验章节**，展示：
1. FFGo baseline 在长视频中产品身份衰减的定量证据
2. 我们的注入方法在长视频中保持产品身份的优势
3. 滑动窗口 + 注入方案的实用性

这将显著增强论文的说服力——不仅在短视频上优于 FFGo，在长视频场景下优势更加明显。

---

## 综合：论文框架与贡献组合 {#综合}

### FFGo 的不足（我们的切入点）

| FFGo 不足 | 对应方向 |
|-----------|---------|
| 不处理不完美输入（假设首帧完美） | 方向一 + 方向二：双条件 CFG / 渐进质量条件训练 |
| 无产品专用身份损失（仅用标准扩散损失） | 方向三 + 方向五：DINO 编码器 + 频率分解 + 产品感知损失 |
| 高度依赖 prompt 质量 | 方向四：VLM 自动 prompt 优化 |
| 首帧参考随帧数衰减，无法生成长视频 | 方向六：持续注入 + 滑动窗口 |

### 与现有工作的定位对比

| 方法 | 产品身份保持 | 不完美输入鲁棒性 | 长视频支持 | 架构改动 |
|------|:---:|:---:|:---:|:---:|
| FFGo（baseline） | 隐式（首帧参考，会衰减） | 无 | 弱（5s 后衰减） | 无（仅 LoRA） |
| VACE | 通用编辑 | N/A | N/A | Context Adapter |
| Kaleido | 多主体 S2V | N/A | N/A | R-RoPE |
| MAGREF | 多参考 | N/A | N/A | 无（掩码引导） |
| **Ours（拟提出）** | **显式（持续注入，不衰减）** | **双条件 CFG** | **滑动窗口+注入** | **轻量 adapter** |

### Ours 的完整 Pipeline

**核心理念：FFGo 首帧定全局，adapter 持续补细节，双条件训练保鲁棒。**

```
┌─────────────────────────────────────────────────────────┐
│  沿用 FFGo 的部分（不变）                                  │
│                                                          │
│  产品 RGBA 图 ──┐                                        │
│                 ├─→ FFGo 首帧拼贴 ──→ 首帧 mask=0（保留）  │
│  去除物体背景 ──┘  （或不去除，双条件训练使其可选）           │
│                                                          │
│  "ad23r2 the camera view suddenly changes. {prompt}"     │
│  → T5 文本编码 → 文本条件                                 │
│                                                          │
│  FFGo LoRA adapter → 控制转场行为和时序一致性              │
├─────────────────────────────────────────────────────────┤
│  新增的部分（我们的贡献）                                  │
│                                                          │
│  产品 RGBA 图 ──→ DINO-v2 编码器（冻结）                  │
│                   │                                      │
│                   ├─→ 低频特征（形状/轮廓）→ 注入 DiT 浅层 │
│                   └─→ 高频特征（纹理/细节）→ 注入 DiT 深层 │
│                                                          │
│  + 双条件 CFG 训练（鲁棒性，去除物品可选）                  │
│  + 产品感知损失（DINO 相似度 + CLIP 对齐）                 │
│  + VLM 优化的 prompt                                     │
└─────────────────────────────────────────────────────────┘
```

**FFGo LoRA 与产品身份 adapter 分工**：
- FFGo LoRA：转场、时序一致性、整体画面生成
- 产品身份 adapter（约 5-10M 参数）：持续注入产品外观细节，防止衰减
- 两者并行工作，各司其职

### 推荐贡献组合（按新颖性排序）

| 优先级 | 贡献 | 对应方向 | 新颖性 |
|:------:|------|---------|:------:|
| 核心 1 | 不完美去除鲁棒性训练（双条件/渐进质量 CFG） | 方向二 | **高** |
| 核心 2 | 产品身份持续注入（频率分解 DINO 特征 + 产品感知损失） | 方向三+五 | **高** |
| 支撑 3 | 长视频生成（滑动窗口+注入，展示注入在长视频中的优势） | 方向六 | **中高** |
| 支撑 4 | VLM 优化 Prompt Pipeline | 方向四 | 中 |
| 工程 5 | 训练数据质量谱构造（FLUX Fill 去除版本） | 方向一 | 中 |

### 推荐论文标题方向

1. "Beyond First-Frame Pasting: Robust Product Video Customization with Identity-Preserving Injection"
2. "RobustPVTT: Product Video Template Transformation with Dual-Condition Training and Persistent Identity Injection"

### 建议的第一步

1. **检查 FFGo CFG 实现**：确认方案 B（Unconditional Priors 替换）是否有增量价值（源码分析已在方向二给出，结论：有价值）
2. **开始构造训练数据**：为 PVTT 数据集生成 FLUX Fill 去除版本
3. **短视频 + 长视频对比实验**：测量 FFGo baseline 在 5s/10s/15s 下的 ProdCLIP 衰减曲线，验证长视频场景下注入方法的必要性
4. 用评估脚本量化每一步改进的效果

---

## 参考文献

### 物体去除 & Inpainting
- BrushNet (ECCV 2024): [arXiv:2403.06976](https://arxiv.org/abs/2403.06976)
- PowerPaint (ECCV 2024): https://powerpaint.github.io/
- OmniPaint (ICCV 2025): [arXiv:2503.08677](https://arxiv.org/abs/2503.08677)
- LanPaint (2025): [arXiv:2502.03491](https://arxiv.org/abs/2502.03491)
- Token Painter (2025): [arXiv:2509.23919](https://arxiv.org/abs/2509.23919)

### 视频定制 & 身份保持
- FFGo (2025.11): [arXiv:2511.15700](https://arxiv.org/abs/2511.15700)
- VACE (ICCV 2025): [arXiv:2503.07598](https://arxiv.org/abs/2503.07598)
- DreamVideo (CVPR 2024): https://dreamvideo-t2v.github.io/
- Subject-driven Disentangled (2025.4): [arXiv:2504.17816](https://arxiv.org/abs/2504.17816)
- CustomVideo (2024.1): [arXiv:2401.09962](https://arxiv.org/abs/2401.09962)
- Magic-Me (2024): [arXiv:2402.09368](https://arxiv.org/abs/2402.09368)
- Kaleido (2025.10): [arXiv:2510.18573](https://arxiv.org/abs/2510.18573)
- MAGREF (ICLR 2026): [arXiv:2505.23742](https://arxiv.org/abs/2505.23742)

### 身份保持架构
- MimicBrush (NeurIPS 2024): [arXiv:2406.07547](https://arxiv.org/abs/2406.07547)
- IP-Adapter: https://ip-adapter.github.io/
- I2V-Adapter (SIGGRAPH 2024): [arXiv:2312.16693](https://arxiv.org/abs/2312.16693)
- MagicMirror (ICCV 2025): [arXiv:2501.03931](https://arxiv.org/abs/2501.03931)
- ConsisID (CVPR 2025 Highlight): [arXiv:2411.17440](https://arxiv.org/abs/2411.17440)

### 训练 & 优化
- Classifier-Free Guidance: [arXiv:2207.12598](https://arxiv.org/abs/2207.12598)
- Unconditional Priors Matter! (2025.3): [arXiv:2503.20240](https://arxiv.org/abs/2503.20240)
- Noise Consistency Regularization (CVPR 2025W): [arXiv:2506.06483](https://arxiv.org/abs/2506.06483)
- Prompt-A-Video (2024.12): [arXiv:2412.15156](https://arxiv.org/abs/2412.15156)
- VIVA (2025.12): [arXiv:2512.16906](https://arxiv.org/html/2512.16906)
- JoyCaption: https://github.com/fpgaminer/joycaption

### 视频生成基座
- Wan2.2: https://github.com/Wan-Video/Wan2.2
- Wan2.1: https://github.com/Wan-Video/Wan2.1
