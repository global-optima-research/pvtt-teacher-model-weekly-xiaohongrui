# AnyV2V: A Tuning-Free Framework For Any Video-to-Video Editing Tasks 调研文档

> **论文信息**: Published in Transactions on Machine Learning Research (11/2024)
> **作者**: Max Ku, Cong Wei, Weiming Ren, Harry Yang, Wenhu Chen
> **机构**: University of Waterloo, Vector Institute, Harmony.AI
> **代码**: https://github.com/TIGER-AI-Lab/AnyV2V

---

## 一、背景 (Background)

### 1.1 研究领域现状

深度生成模型（如扩散模型）在图像生成和编辑领域取得了巨大进展（Stable Diffusion、DALL-E等），但**视频生成和编辑**领域尚未达到同等水平。主要挑战在于：

- **数据稀缺**: 大规模视频编辑数据对的获取远比图像困难
- **计算资源**: 训练视频编辑模型需要巨大的计算开销
- **时序一致性**: 视频编辑需要在保持帧间一致性的同时进行语义修改

### 1.2 已有方法的分类

现有视频编辑方法大致分为两类：

| 类别 | 代表方法 | 问题 |
|------|---------|------|
| **Zero-shot 适配** | TokenFlow, Pix2Video, FateZero | 从预训练T2I模型零样本适配，缺乏时序理解，**容易产生闪烁** |
| **微调方法** | Tune-A-Video, VideoP2P, VideoSwap | 需要fine-tune T2I或T2V模型的运动模块，**更耗时且计算量大** |

**共同局限**: 所有方法都只能处理**特定类型**的编辑（主要是基于文本提示的编辑），无法支持参考图像风格迁移、主体替换、身份操控等任务。

---

## 二、动机 (Motivation)

### 2.1 核心问题

AnyV2V 想要解决以下关键问题：

1. **编辑类型受限**: 已有方法严重依赖文本编码器，但文本描述天然存在歧义性，且很多编辑意图（如"换成某幅特定画作的风格"）无法用文本准确表达
2. **需要微调**: 大多数方法需要针对每个视频或每种任务进行参数微调
3. **缺乏灵活性**: 无法利用图像编辑领域已有的强大工具（InstructPix2Pix、InstantID、AnyDoor、Neural Style Transfer等）

### 2.2 核心洞察

> **关键观察**: 如果把视频编辑问题拆解为 (1) 图像编辑 + (2) 图像到视频生成(I2V)，就可以复用图像编辑领域丰富的工具生态，同时利用I2V模型天然的时序建模能力。

---

## 三、设计思路与方法论

### 3.1 为什么这么设计？

AnyV2V的设计基于以下推理链条：

1. **图像编辑工具已经很成熟** → 可以精确控制第一帧的编辑效果
2. **I2V模型天然具备运动生成能力** → 可以基于编辑后的第一帧生成时序一致的视频
3. **DDIM Inversion可以提取源视频的结构信息** → 通过反转噪声保留原始视频的运动和结构
4. **特征注入可以增强一致性** → 通过注入源视频的空间和时序特征，进一步保证编辑视频与源视频的一致性

### 3.2 两阶段流程设计

**Stage 1: 第一帧编辑** — 使用任意黑盒图像编辑模型修改源视频的第一帧

**Stage 2: 编辑传播** — 利用I2V模型将编辑传播到整个视频，同时通过DDIM反转噪声和特征注入保持与源视频的一致性

---

## 四、模型架构详解

### 4.1 整体架构

AnyV2V 并非一个新的神经网络，而是一个**框架/流水线 (pipeline)**，它组合了现有组件：

```
源视频 V^S = {I_1, I_2, ..., I_n}
       │
       ├──→ 第一帧 I_1 ──→ [图像编辑模型 φ_img] ──→ 编辑后第一帧 I_1*
       │
       └──→ [DDIM Inversion] ──→ 反转噪声 z_T^S + 中间特征 {f, Q, K}
                                        │
                                        ▼
                    [I2V模型去噪 + 特征注入] ──→ 编辑后视频 V*
```

### 4.2 预备知识

#### 4.2.1 I2V (Image-to-Video) 生成模型

AnyV2V基于**潜空间扩散 (Latent Diffusion)** 的I2V模型。给定输入第一帧 $I_1$、文本提示 $\mathbf{s}$ 和时间步 $t$ 处的含噪视频潜变量 $\mathbf{z}_t$，去噪模型 $\epsilon_\theta$ 恢复更干净的潜变量 $\mathbf{z}_{t-1}$：

$$\epsilon_\theta(\mathbf{z}_t, I_1, \mathbf{s}, t)$$

**符号说明**:
- $\epsilon_\theta$: 以 $\theta$ 为参数的去噪U-Net模型
- $\mathbf{z}_t$: 时间步 $t$ 处的含噪视频潜变量（所有帧的latent表示）
- $I_1$: 第一帧图像，作为I2V模型的条件输入
- $\mathbf{s}$: 文本提示
- $t$: 扩散时间步

去噪模型内部包含**空间自注意力层**和**时序自注意力层**：

**公式(1) — 自注意力的QKV计算**:

$$Q = W^Q z, \quad K = W^K z, \quad V = W^V z$$

- $z$: 自注意力层的输入隐状态
- $W^Q, W^K, W^V$: 可学习的投影矩阵，分别将 $z$ 映射为查询(Query)、键(Key)、值(Value)向量
- 对于**空间自注意力**: $z$ 是同一帧内所有空间位置的token序列
- 对于**时序自注意力**: $z$ 是所有帧中相同空间位置的token序列

**公式(2) — 注意力计算**:

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^\top}{\sqrt{d}}\right) V$$

- $QK^\top$: 查询和键的点积，衡量token之间的相似度
- $\sqrt{d}$: 缩放因子，$d$ 是键向量的维度，防止点积值过大导致softmax梯度消失
- $\text{Softmax}(\cdot)$: 将相似度归一化为注意力权重（概率分布）
- 乘以 $V$: 用注意力权重对值向量加权求和，得到输出

#### 4.2.2 DDIM Inversion

**DDIM采样** (Song et al., 2020) 是扩散模型的确定性采样方法。其逆过程 (DDIM Inversion) 允许从 $\mathbf{z}_t$ 计算 $\mathbf{z}_{t+1}$，即**将干净数据逐步加噪回高斯噪声**：

**公式 — DDIM Inversion**:

$$\mathbf{z}_{t+1} = \sqrt{\frac{\alpha_{t+1}}{\alpha_t}} \mathbf{z}_t + \left(\sqrt{\frac{1}{\alpha_{t+1}} - 1} - \sqrt{\frac{1}{\alpha_t} - 1}\right) \cdot \epsilon_\theta(\mathbf{z}_t, x_0, \mathbf{s}, t)$$

- $\mathbf{z}_t$: 当前时间步 $t$ 的潜变量
- $\mathbf{z}_{t+1}$: 下一时间步（更嘈杂）的潜变量
- $\alpha_t$: 从扩散过程的方差调度(variance schedule)导出的系数，控制信号与噪声的比例
- $\epsilon_\theta(\mathbf{z}_t, x_0, \mathbf{s}, t)$: 模型在时间步 $t$ 对噪声的预测
- 该公式的**作用**: 对源视频进行反转，得到其对应的初始噪声 $\mathbf{z}_T^S$，这个噪声编码了源视频的结构和运动信息

#### 4.2.3 Plug-and-Play (PnP) 扩散特征

PnP (Tumanyan et al., 2023a) 发现T2I去噪U-Net中间层的**卷积特征** $f$ 和**自注意力分数** $A$ 能捕捉语义区域信息（如人体的腿、躯干等）。AnyV2V将这一思想从图像扩展到视频，在I2V模型的卷积层、空间注意力层和时序注意力层中进行特征注入。

### 4.3 核心组件详解

#### 4.3.1 Stage 1 — 灵活的第一帧编辑 (Flexible First Frame Editing)

给定源视频 $V^S = \{I_1, I_2, ..., I_n\}$，提取第一帧 $I_1$，使用任意图像编辑模型 $\phi_\text{img}$ 获得编辑后的第一帧：

$$I_1^* = \phi_\text{img}(I_1, C)$$

- $I_1^*$: 编辑后的第一帧
- $\phi_\text{img}$: 任意图像编辑模型（InstructPix2Pix、AnyDoor、InstantID、Neural Style Transfer等）
- $C$: 编辑条件（文本提示、参考人脸、参考主体图像、参考风格图像等）

**支持的编辑模型类型**:
- **InstructPix2Pix**: 基于文本指令的图像编辑
- **AnyDoor**: 基于参考图像的主体替换
- **InstantID**: 基于参考人脸的身份替换
- **Neural Style Transfer (NST)**: 基于参考图像的风格迁移

#### 4.3.2 Stage 2a — DDIM反转获取结构引导 (Structural Guidance via DDIM Inversion)

**公式(3) — 源视频的DDIM反转**:

$$\mathbf{z}_t^S = \text{DDIM\_Inv}(\epsilon_\theta(\mathbf{z}_{t+1}, I_1, \varnothing, t))$$

- $\mathbf{z}_t^S$: 源视频在时间步 $t$ 的反转潜变量
- $\text{DDIM\_Inv}(\cdot)$: DDIM反转操作
- $I_1$: 原始第一帧（注意：反转时使用**原始**第一帧作为条件）
- $\varnothing$: 空文本提示（反转时不使用文本条件）

**关键点**: 反转不使用文本条件但使用第一帧条件。最终得到的噪声 $\mathbf{z}_T^S$ 编码了源视频的运动和结构信息，将作为编辑视频生成的初始噪声。

**实际处理**: 在某些I2V模型上，从完全反转的 $\mathbf{z}_T^S$ 开始采样可能产生失真，此时可以从较早的时间步 $T' < T$ 开始采样来缓解。

#### 4.3.3 Stage 2b — 空间特征注入实现外观引导 (Appearance Guidance via Spatial Feature Injection)

**问题**: 仅使用编辑后的第一帧和DDIM反转噪声，I2V模型往往无法正确保留源视频的背景和运动。

**解决方案**: 在去噪过程中，同时对源视频进行去噪，提取其空间特征并注入到编辑视频的去噪过程中。

具体来说，对源视频使用之前收集的DDIM反转潜变量在每个时间步 $t$ 进行去噪：

$$\mathbf{z}_{t-1}^S = \epsilon_\theta(\mathbf{z}_t^S, I_1, \varnothing, t)$$

在去噪过程中保存两类特征：

1. **卷积特征** $f^{l_1}$: U-Net decoder第 $l_1$ 层skip connection之前的卷积特征图
2. **空间自注意力分数** $\{A_s^{l_2}\}$: 从第 $l_2$ 层收集空间自注意力的Query和Key:  $\{Q_s^{l_2}\}$ 和 $\{K_s^{l_2}\}$

> 注意: 论文收集Query和Key而非直接收集注意力矩阵 $A$，因为注意力分数是由Q和K参数化得到的

然后将这些特征替换到编辑视频去噪分支中对应层的特征。

**阈值控制**: 使用两个阈值 $\tau_\text{conv}$ 和 $\tau_\text{sa}$ 控制注入时机，仅在前 $\tau_\text{conv}$ 和 $\tau_\text{sa}$ 步进行特征注入。这是因为：
- 早期去噪步骤决定整体布局和结构
- 后期步骤决定细节和高频信息
- 仅在早期注入可以保留结构同时允许编辑细节变化

#### 4.3.4 Stage 2c — 时序特征注入实现运动引导 (Motion Guidance via Temporal Feature Injection)

**问题**: 空间特征注入增强了背景一致性，但编辑视频的运动仍可能与源视频不一致。

**关键观察**: I2V模型通常从预训练T2I模型初始化，训练时空间层的参数被冻结或使用较低学习率，而**时序层的参数被大量更新**。因此，运动信息主要编码在时序层中。并发工作也发现时序层特征与光流有相似特征。

**解决方案**: 类似空间注入，收集源视频的时序自注意力Query和Key: $Q_t^{l_3}$ 和 $K_t^{l_3}$，注入到编辑视频去噪的时序注意力层中。

**阈值控制**: 使用阈值 $\tau_\text{ta}$，仅在前 $\tau_\text{ta}$ 步进行时序特征注入。

#### 4.3.5 整合 — 完整的特征注入公式

**公式(4) — 编辑视频去噪（带特征注入）**:

$$\mathbf{z}_{t-1}^* = \epsilon_\theta(\mathbf{z}_t^*, I^*, \mathbf{s}^*, t \;;\; \{f^{l_1}, Q_s^{l_2}, K_s^{l_2}, Q_t^{l_3}, K_t^{l_3}\})$$

**符号说明**:
- $\mathbf{z}_t^*$: 编辑视频在时间步 $t$ 的潜变量（初始噪声 $\mathbf{z}_T^* = \mathbf{z}_T^S$ 来自DDIM反转）
- $I^*$: 编辑后的第一帧
- $\mathbf{s}^*$: 目标文本提示
- $f^{l_1}$: 从源视频去噪中提取的U-Net decoder第 $l_1$ 层的卷积特征
- $Q_s^{l_2}, K_s^{l_2}$: 从源视频去噪中提取的第 $l_2$ 层空间自注意力的Query和Key
- $Q_t^{l_3}, K_t^{l_3}$: 从源视频去噪中提取的第 $l_3$ 层时序自注意力的Query和Key
- $\epsilon_\theta(\cdot \;;\; \{\cdot\})$ 中的分号表示**特征替换操作**: 将编辑分支中对应层的特征替换为源视频分支的特征

### 4.4 Python代码示例

```python
import torch
from diffusers import I2VGenXLPipeline  # 示例I2V模型

class AnyV2V:
    """AnyV2V框架的简化示意实现"""

    def __init__(self, i2v_model, image_editor):
        self.i2v_model = i2v_model          # I2V生成模型 (如I2VGen-XL)
        self.image_editor = image_editor    # 图像编辑模型 (如InstructPix2Pix)
        self.unet = i2v_model.unet          # 去噪U-Net

        # 特征注入阈值
        self.tau_conv = 0.2   # 卷积特征注入：前20%步
        self.tau_sa = 0.2     # 空间注意力注入：前20%步
        self.tau_ta = 0.5     # 时序注意力注入：前50%步

        # 特征注入层设置
        self.l1 = 4                          # 卷积特征注入层
        self.l2 = [4, 5, 6, 7, 8, 9, 10, 11]  # 空间注意力注入层
        self.l3 = [4, 5, 6, 7, 8, 9, 10, 11]  # 时序注意力注入层

    def edit_video(self, source_video, edit_condition, target_prompt):
        """
        Args:
            source_video: 源视频帧列表 [I_1, I_2, ..., I_n]
            edit_condition: 编辑条件 (文本提示/参考图像等)
            target_prompt: 目标文本提示 s*
        Returns:
            edited_video: 编辑后的视频帧列表
        """
        # ========== Stage 1: 第一帧编辑 ==========
        first_frame = source_video[0]  # I_1
        edited_first_frame = self.image_editor(first_frame, edit_condition)  # I_1*

        # ========== Stage 2: 编辑传播 ==========

        # Step 2a: DDIM Inversion — 获取源视频的反转噪声
        # 不使用文本条件，但使用原始第一帧条件
        inverted_latents = self.ddim_inversion(
            source_video,
            first_frame=first_frame,
            prompt=""  # 空文本
        )
        # inverted_latents: {z_T^S, z_{T-1}^S, ..., z_1^S}

        z_T_source = inverted_latents[T]  # 最终的反转噪声，作为初始噪声

        # Step 2b & 2c: 提取源视频特征 & 生成编辑视频
        # 先对源视频进行去噪以提取中间特征
        source_features = self.extract_source_features(
            inverted_latents, first_frame
        )

        # 使用编辑后的第一帧和源视频特征，从反转噪声开始去噪
        edited_video = self.guided_sampling(
            initial_noise=z_T_source,       # 使用源视频的反转噪声
            first_frame=edited_first_frame,  # 使用编辑后的第一帧
            prompt=target_prompt,            # 目标提示
            source_features=source_features  # 注入的源视频特征
        )

        return edited_video

    def ddim_inversion(self, video, first_frame, prompt):
        """DDIM反转：将视频编码到噪声空间"""
        # 将视频编码为潜变量
        z_0 = self.encode(video)  # z_0: 干净的视频潜变量

        inverted_latents = {0: z_0}
        z_t = z_0

        for t in range(0, T):
            # 模型预测噪声
            noise_pred = self.unet(z_t, first_frame, prompt="", t=t)

            # DDIM反转公式：z_t → z_{t+1}
            alpha_t = self.get_alpha(t)
            alpha_t1 = self.get_alpha(t + 1)

            z_t1 = (torch.sqrt(alpha_t1 / alpha_t) * z_t +
                    (torch.sqrt(1/alpha_t1 - 1) - torch.sqrt(1/alpha_t - 1)) * noise_pred)

            inverted_latents[t + 1] = z_t1
            z_t = z_t1

        return inverted_latents

    def extract_source_features(self, inverted_latents, first_frame):
        """对源视频去噪并提取中间层特征"""
        features = {}
        z_t = inverted_latents[T]

        for t in range(T, 0, -1):
            # 去噪并收集中间特征
            z_t_prev, conv_feat, spatial_attn, temporal_attn = \
                self.unet_with_feature_extraction(
                    z_t, first_frame, prompt="", t=t
                )

            features[t] = {
                'conv': conv_feat,           # f^{l_1}: 卷积特征
                'spatial_Q': spatial_attn['Q'],  # Q_s^{l_2}
                'spatial_K': spatial_attn['K'],  # K_s^{l_2}
                'temporal_Q': temporal_attn['Q'], # Q_t^{l_3}
                'temporal_K': temporal_attn['K'], # K_t^{l_3}
            }
            z_t = z_t_prev

        return features

    def guided_sampling(self, initial_noise, first_frame, prompt, source_features):
        """带特征注入的引导采样"""
        z_t = initial_noise  # z_T* = z_T^S (使用源视频的反转噪声)

        for t in range(T, 0, -1):
            # 判断当前步是否需要进行各类特征注入
            inject_conv = (t > T - self.tau_conv * T)
            inject_spatial = (t > T - self.tau_sa * T)
            inject_temporal = (t > T - self.tau_ta * T)

            # 准备注入特征
            injection = {}
            if inject_conv:
                injection['conv'] = source_features[t]['conv']
            if inject_spatial:
                injection['spatial_Q'] = source_features[t]['spatial_Q']
                injection['spatial_K'] = source_features[t]['spatial_K']
            if inject_temporal:
                injection['temporal_Q'] = source_features[t]['temporal_Q']
                injection['temporal_K'] = source_features[t]['temporal_K']

            # 去噪一步（带特征注入）
            z_t_prev = self.unet_with_injection(
                z_t, first_frame, prompt, t, injection
            )
            z_t = z_t_prev

        # 解码得到编辑后的视频
        edited_video = self.decode(z_t)
        return edited_video
```

---

## 五、推理流程 (Inference Pipeline)

### 5.1 输入

| 输入 | 描述 |
|------|------|
| 源视频 $V^S$ | 待编辑的原始视频 $\{I_1, I_2, ..., I_n\}$ |
| 编辑条件 $C$ | 取决于编辑类型：文本提示/参考人脸/参考主体/参考风格图像 |
| 目标提示 $\mathbf{s}^*$ | 描述编辑后视频的文本提示 |

### 5.2 输出

| 输出 | 描述 |
|------|------|
| 编辑后视频 $V^*$ | 编辑后的视频帧序列 |

### 5.3 数据流转详细步骤

```
Step 1: 提取第一帧
  源视频 V^S → I_1 (第一帧)

Step 2: 编辑第一帧
  I_1 + 条件C → [图像编辑模型] → I_1* (编辑后第一帧)

Step 3: DDIM反转源视频
  V^S + I_1(原始) + 空文本 → [VAE Encoder → DDIM Inversion] → {z_T^S, z_{T-1}^S, ..., z_1^S}
  (注: 每步还收集中间的去噪结果)

Step 4: 提取源视频特征（与Step 3可并行/同步进行）
  {z_t^S} + I_1 + 空文本 → [U-Net去噪 (源视频)] → 收集每步的:
    - 卷积特征 f^{l_1}
    - 空间注意力 Q_s^{l_2}, K_s^{l_2}
    - 时序注意力 Q_t^{l_3}, K_t^{l_3}

Step 5: 带特征注入的去噪采样
  初始噪声 z_T^S + 编辑后第一帧 I_1* + 目标提示 s*
  → [U-Net去噪 × T步, 每步注入源视频特征]
  → z_0* (编辑后的视频潜变量)

Step 6: VAE解码
  z_0* → [VAE Decoder] → V* (编辑后的视频像素)
```

### 5.4 推理时的计算开销

- **GPU显存**: 约15GB (Nvidia A6000 GPU)
- **推理时间**: 约100秒（16帧视频）
- **无需训练/微调**: 完全zero-shot

---

## 六、训练流程

### AnyV2V 不需要训练！

AnyV2V 是一个 **tuning-free (无需微调)** 的框架。它直接利用现成的预训练模型：

- **I2V模型** (如 I2VGen-XL, ConsistI2V, SEINE) — 直接使用预训练权重
- **图像编辑模型** (如 InstructPix2Pix, AnyDoor) — 直接使用预训练权重

唯一需要设置的是超参数：

| 超参数 | 默认值 | 含义 |
|--------|--------|------|
| $\tau_\text{conv}$ | $0.2T$ | 卷积特征注入持续的步数 |
| $\tau_\text{sa}$ | $0.2T$ | 空间注意力注入持续的步数 |
| $\tau_\text{ta}$ | $0.5T$ | 时序注意力注入持续的步数 |
| $l_1$ | 4 | 卷积特征注入的U-Net decoder层 |
| $l_2 = l_3$ | {4,5,...,11} | 空间和时序注意力注入的U-Net decoder层 |
| $T$ | 模型默认值 | 总采样步数 |

---

## 七、训练数据集

由于AnyV2V不需要训练，所以**没有专门的训练数据集**。

它依赖的预训练模型各自的训练数据：
- **I2VGen-XL**: 在大规模视频数据上训练
- **ConsistI2V**: 在视频数据上训练
- **SEINE**: 在视频数据上训练
- **InstructPix2Pix**: 在文本-图像编辑对上训练
- 等等

---

## 八、实验设计与结果

### 8.1 实验设置

#### 使用的I2V骨干模型
- **I2VGen-XL** (Zhang et al., 2023c)
- **ConsistI2V** (Ren et al., 2024)
- **SEINE** (Chen et al., 2023d)

#### 使用的图像编辑模型
- **InstructPix2Pix**: 基于提示的编辑
- **Neural Style Transfer (NST)**: 风格迁移
- **AnyDoor**: 主体替换
- **InstantID**: 身份操控

#### 基线对比方法
- **Tune-A-Video** (Wu et al., 2023b)
- **TokenFlow** (Geyer et al., 2023)
- **FLATTEN** (Cong et al., 2023)

### 8.2 定义的四类编辑任务

| 任务 | 描述 | 输入条件 |
|------|------|----------|
| Prompt-based Editing | 文本指令编辑 | 文本提示 |
| Reference-based Style Transfer | 参考风格迁移 | 参考风格图像 |
| Subject-driven Editing | 主体替换 | 参考主体图像 |
| Identity Manipulation | 身份替换 | 参考人脸图像 |

> **实验动机**: 前两类任务是已有方法也能做的（用于公平对比），后三类是AnyV2V独有的能力（展示框架的灵活性）。

### 8.3 评估指标

| 指标 | 含义 | 计算方法 |
|------|------|----------|
| **CLIP-Text** | 文本对齐度 | 编辑提示的CLIP文本嵌入与视频帧CLIP图像嵌入的平均余弦相似度 |
| **CLIP-Image** | 时序一致性 | 相邻帧CLIP图像嵌入的平均余弦相似度 |
| **Human Eval - Alignment** | 人工评估-提示对齐 | 用户投票偏好 |
| **Human Eval - Overall** | 人工评估-整体偏好 | 用户投票偏好 |

### 8.4 实验1: Prompt-based Editing 定量对比

**动机**: 验证AnyV2V在传统文本引导编辑任务上能否与专用方法竞争。

**Table 2 结果**:

| 方法 | Alignment↑ | Overall↑ | CLIP-Text↑ | CLIP-Image↑ |
|------|-----------|---------|------------|-------------|
| Tune-A-Video | 15.2% | 2.1% | 0.2902 | 0.9704 |
| TokenFlow | 31.7% | 20.7% | 0.2858 | **0.9783** |
| FLATTEN | 25.5% | 16.6% | 0.2742 | 0.9739 |
| **AnyV2V (I2VGen-XL)** | **69.7%** | **46.2%** | **0.2932** | 0.9652 |

**结论**: AnyV2V在人工评估中大幅领先（提示对齐69.7% vs 第二名31.7%，整体偏好46.2% vs 20.7%），CLIP-Text分数也最高。说明利用图像编辑模型的精确编辑能力在视频编辑中是有效的。

### 8.5 实验2: 新任务（风格迁移、主体替换、身份操控）

**动机**: 展示AnyV2V独有的多任务编辑能力——这些任务是其他基线方法无法完成的。

**结果** (定性+定量):
- **风格迁移**: AnyV2V能准确捕捉康定斯基"Composition VII"和梵高"Chateau in Auvers at Sunset"的风格
- **主体替换**: 能将猫替换为狗，将汽车替换为跑车，同时保持原始运动
- **身份操控**: 能将视频中的人脸替换为目标人脸，保持运动一致性

### 8.6 实验3: 不同I2V骨干的对比

**动机**: 研究I2V模型的选择对编辑效果的影响。

**结论**:
- **I2VGen-XL**: 最鲁棒，泛化能力最强，能处理各种运动
- **ConsistI2V**: 能生成一致运动，但有时出现水印
- **SEINE**: 泛化较弱，但在简单运动（如走路）时效果不错

### 8.7 实验4: 长视频编辑

**动机**: 验证AnyV2V能否编辑超过I2V模型训练帧数（通常16帧）的视频。

**方法**: 对更长的视频进行DDIM反转得到更长的反转latent，强制I2V模型生成更多帧。

**结果** (Figure 5): 成功编辑了121帧的视频（"将行走的女人变成机器人"），保持了时序和语义一致性。

**结论**: DDIM反转的latent包含足够的时序和语义信息，可以引导I2V模型超越训练帧数生成视频。

### 8.8 实验5: 消融研究 (Ablation Study)

**动机**: 验证AnyV2V三个核心组件各自的贡献。

**设计**: 使用AnyV2V (I2VGen-XL)，在20个样本上逐步移除各组件。

**Table 3 — 消融结果**:

| 模型变体 | CLIP-Image↑ |
|---------|-------------|
| AnyV2V (完整) | 0.9648 |
| w/o 时序注入 (T.I.) | 0.9652 |
| w/o 时序注入 & 空间注入 (T.I. & S.I.) | 0.9637 |
| w/o 时序注入 & 空间注入 & DDIM反转 | 0.9607 |

**各组件的贡献分析**:

1. **时序特征注入**: CLIP-Image分数略有提高(移除后0.9652 vs 0.9648)，但定性观察发现运动与源视频的对齐度明显下降。如Figure 6所示，移除时序注入后女性抬腿的动作未被保留。
   > CLIP-Image指标的反直觉结果说明：该指标只衡量相邻帧的一致性，不能反映与源视频运动的对齐程度。

2. **空间特征注入**: 移除后CLIP-Image显著下降(0.9637)，视频帧间出现外观不一致和运动不连贯。移除空间注入会导致主体外观错误和背景退化。

3. **DDIM反转噪声**: 替换为随机噪声后CLIP-Image进一步下降(0.9607)，I2V模型几乎丧失了驱动编辑后图像运动的能力。这证明DDIM反转噪声是编辑视频结构引导的基础。

### 8.9 实验6: 超参数分析

#### 空间注入阈值 $\tau_\text{conv}, \tau_\text{sa}$ (Figure 8)
- $\tau_{c,s} = 0$: 无空间注入，编辑视频无法保持源视频的布局和运动
- $\tau_{c,s} = 0.2T$: **最佳设置**，保持布局同时不引入多余的源视频细节
- $\tau_{c,s} > 0.2T$: 注入过多，编辑视频被源视频的高频细节"污染"

#### 时序注入阈值 $\tau_\text{ta}$ (Figure 9)
- $\tau_\text{ta} < 0.5T$: 运动引导太弱，运动与源视频仅部分对齐
- $\tau_\text{ta} = 0.5T$: **最佳设置**，平衡运动对齐、运动一致性和视频质量
- $\tau_\text{ta} > 0.5T$: 运动对齐更强，但出现画面失真

---

## 九、创新性贡献

1. **首创性的问题分解思路**: 首次将视频编辑问题分解为"图像编辑 + I2V生成"，这是一个根本性不同的解决范式。之前所有方法都试图直接在视频上进行编辑，而AnyV2V将其简化为更成熟的图像编辑问题。

2. **任务通用性**: 通过兼容任意黑盒图像编辑模型，AnyV2V首次实现了在统一框架下支持**四类编辑任务**（prompt-based、style transfer、subject-driven、identity manipulation），前三个后类任务是之前方法无法处理的。

3. **无需微调**: 完全zero-shot，不需要任何训练或微调，大大降低了使用门槛。

4. **长视频支持**: 发现通过反转超过训练帧数的视频，可以驱动I2V模型生成更长的编辑视频。

5. **特征注入机制的扩展**: 将PnP的特征注入思想从T2I扩展到I2V，在卷积层、空间注意力层和时序注意力层三个维度进行特征注入。

---

## 十、不足之处与改进方向

### 10.1 不足之处

1. **依赖I2V模型质量**: AnyV2V的编辑效果直接受限于底层I2V模型的能力。如果I2V模型本身的运动生成不够好，编辑效果也会受影响。

2. **依赖第一帧编辑质量**: 如果图像编辑模型在第一帧上的编辑效果不好（如主体形变、背景改变过多），后续的视频生成也无法恢复。

3. **仅限于基于U-Net的I2V模型**: 论文中的特征注入机制专为U-Net架构设计，无法直接应用于DiT等新型架构。

4. **时序一致性仍有限**: 在复杂编辑（大幅度外观变化）下，编辑视频的时序一致性可能不够好。

5. **推理效率**: 需要同时进行源视频去噪和编辑视频去噪（两次完整的去噪过程），推理成本是普通I2V生成的约2倍。

### 10.2 改进方向

1. **迁移到T2V模型** (论文明确提出): 目前AnyV2V仅支持I2V模型，未来可以研究如何将I2V的特性桥接到更强大的T2V模型上，以利用T2V模型更强的生成能力。

2. **适配DiT架构**: 随着Sora、Wan等基于DiT的视频生成模型的兴起，需要设计适用于DiT的特征注入机制。

3. **自适应阈值**: 当前的注入阈值是手动设置的固定值，可以研究根据编辑类型和视频内容自动调整阈值的方法。

4. **多帧编辑**: 目前只编辑第一帧然后传播，可以扩展为编辑多个关键帧以支持更复杂的编辑。

5. **提升编辑可控性**: 结合mask、深度图等额外控制信号，实现更精确的局部编辑。

---

## 十一、总结速记表

| 维度 | 内容 |
|------|------|
| **论文** | AnyV2V: A Tuning-Free Framework For Any Video-to-Video Editing Tasks |
| **发表** | TMLR 2024 (11月) |
| **核心思想** | 视频编辑 = 第一帧图像编辑 + I2V生成 + DDIM反转 + 特征注入 |
| **方法类型** | 无训练框架 (Tuning-Free Pipeline) |
| **输入** | 源视频 + 编辑条件（文本/参考图像） |
| **输出** | 编辑后的视频 |
| **骨干模型** | I2VGen-XL / ConsistI2V / SEINE (I2V模型) |
| **图像编辑模型** | InstructPix2Pix / AnyDoor / InstantID / NST |
| **关键技术** | ① DDIM Inversion (结构引导) ② 空间特征注入 (外观引导) ③ 时序特征注入 (运动引导) |
| **支持任务** | 文本编辑、风格迁移、主体替换、身份操控 (4类) |
| **核心公式** | $\mathbf{z}_{t-1}^* = \epsilon_\theta(\mathbf{z}_t^*, I^*, \mathbf{s}^*, t; \{f^{l_1}, Q_s^{l_2}, K_s^{l_2}, Q_t^{l_3}, K_t^{l_3}\})$ |
| **关键超参数** | $\tau_\text{conv}=0.2T$, $\tau_\text{sa}=0.2T$, $\tau_\text{ta}=0.5T$ |
| **训练需求** | 无需训练 |
| **推理开销** | ~15GB GPU, ~100s/16帧视频 |
| **主要优势** | 通用性强、无需微调、支持任意图像编辑工具 |
| **主要局限** | 依赖I2V模型质量、仅支持U-Net架构、推理需双重去噪 |
| **最佳骨干** | I2VGen-XL (最鲁棒、泛化最强) |
| **人工评估** | 提示对齐69.7%, 整体偏好46.2% (远超基线) |
| **未来方向** | 迁移到T2V/DiT模型、自适应阈值、多帧编辑 |
