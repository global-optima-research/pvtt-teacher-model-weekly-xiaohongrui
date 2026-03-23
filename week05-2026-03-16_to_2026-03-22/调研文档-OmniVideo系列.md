# Omni-Video 系列调研文档
# Omni-Video (v1) + Omni-Video 2

> **Omni-Video v1**: "Democratizing Unified Video Understanding and Generation" (arXiv:2507.06119, 2025)
> **Omni-Video 2**: "Scaling MLLM-Conditioned Diffusion for Unified Video Generation and Editing" (arXiv:2602.08820, 2026)
> **作者团队**: Zhiyu Tan, Hao Yang, Luozheng Qin, Jia Gong, Mengping Yang, Hao Li 等
> **机构**: Fudan University, Shanghai Academy of Artificial Intelligence for Science (SAIS)
> **代码**: https://github.com/SAIS-FUXI/Omni-Video

---

# 第一部分: Omni-Video (v1)

---

## 一、背景 (Background)

### 1.1 研究领域现状

近年来，**多模态理解**（由LLM驱动的MLLM）和**多模态生成**（由扩散模型驱动）分别取得了显著进展：

- **理解侧**: LLAVA、VILA等视觉语言模型在图像/视频理解任务上表现出色
- **生成侧**: Stable Diffusion、Sora、Wan等扩散模型在图像/视频生成上持续进步

但这两类系统通常**分别设计**，导致：
- 每个任务需要单独的模型，建模冗余且计算复杂
- 理解模型不能生成、生成模型缺乏深度理解

### 1.2 统一模型的已有探索

统一多模态模型沿两条路线发展：

| 路线 | 代表方法 | 方式 | 局限 |
|------|---------|------|------|
| **纯AR模型** | Chameleon, Emu3, Janus-pro | 将图像/视频离散化为token，用LLM做next-token prediction | 离散化损失细粒度细节 |
| **Diffusion-AR混合** | Transfusion, Show-O, MetaMorph | LLM输出连续token再由扩散模型解码 | 大多只处理**静态图像**，视频能力有限 |

**核心gap**: 现有统一模型主要针对图像，在**视频领域**（需要时序建模、运动一致性、帧间连贯性）的能力非常有限，且数据和计算需求更高。

---

## 二、动机 (Motivation)

Omni-Video 想要解决的核心问题：

1. **填补视频统一建模的空白**: 现有统一模型主要处理图像，需要一个高效且有效的框架来统一视频理解和生成
2. **资源效率**: 从头训练统一模型需要海量数据和计算资源（如Emu3），需要一种轻量级方案
3. **编辑能力**: 在统一框架中同时支持视频编辑（不仅仅是理解+生成）

### 核心洞察 (Key Insight)

> 教会现有的MLLM产生**连续视觉线索 (continuous visual clues)**，作为扩散解码器的输入条件，由扩散解码器据此生成高质量视频。这样可以复用MLLM的理解能力和扩散模型的生成能力，而无需从头训练。

---

## 三、设计思路

### 3.1 为什么这么设计？

1. **复用已有预训练模型**: MLLM（如VILA）已经具备强大的视频理解能力，T2V扩散模型（如Wan 2.1）已经具备强大的视频生成能力。设计的目标是用最小的架构修改将二者连接。

2. **双头设计**: MLLM原本只输出文本token。为了同时支持理解（输出文本）和生成（输出视觉信号），添加一个**Vision Head**来输出与视觉编码器对齐的连续视觉token。

3. **轻量级适配器**: 视觉token的空间与扩散模型的条件空间不同，需要一个Adapter来桥接，而不是修改扩散模型本身。

4. **渐进式训练**: 一次性训练所有组件太难，所以分三个阶段逐步建立连接。

### 3.2 编辑模式的特殊设计

- **生成模式**: Adapter只接收Vision Head的输出
- **编辑模式**: Adapter额外接收源视频/图像的VAE编码嵌入，提供低级视觉细节用于保留未编辑区域

---

## 四、模型架构详解

### 4.1 整体架构图

```
输入: 文本 + (可选)图像/视频
  │
  ├──→ [Tokenizer] ──→ 文本Token序列
  │
  └──→ [Visual Encoder] ──→ 视觉嵌入序列
           │
           ▼
       ┌──────────┐
       │   LLM    │  (MLLM backbone, e.g., VILA)
       └──┬───┬───┘
          │   │
     ┌────┘   └────┐
     ▼              ▼
 [Text Head]    [Vision Head]
  输出文本token   输出连续视觉token V_o
                    │
                    ▼
              [Vision Adapter]
                    │
                    ▼ Q (条件信号)
      ┌─────────────────────────┐
      │   Diffusion Decoder     │ (T2V model, e.g., Wan 2.1-1.3B)
      │   (+ 可选: VAE编码的     │
      │    源视频嵌入用于编辑)    │
      └─────────────────────────┘
                    │
                    ▼
           生成的图像/视频
```

### 4.2 各组件详解

#### 4.2.1 MLLM中的双头输出

**Text Head**: 标准的语言建模头，预测文本token。

**Vision Head**: 将MLLM的隐状态映射为连续视觉token，训练目标是与预训练视觉编码器（如SigLIP-v2）的输出对齐。

引入了4个特殊token来标记模态边界：
- `<BOI>`: Begin of Image
- `<EOI>`: End of Image
- `<BOV>`: Begin of Video
- `<EOV>`: End of Video

**公式(1) — 视觉对齐损失**:

$$\mathcal{L}_{vision} = \|V_o - E\|^2$$

- $V_o$: Vision Head输出的视觉token序列
- $E$: 预训练视觉编码器（如SigLIP-v2）提取的视觉嵌入
- $\|\cdot\|^2$: L2距离
- **含义**: 强制Vision Head的输出在表示空间上与现有视觉编码器对齐，确保语义接地(semantic grounding)和稳定的潜变量预测
- **作用**: 使MLLM生成的视觉线索具有与预训练视觉编码器相同的语义结构

**公式(2) — 文本交叉熵损失**:

$$\mathcal{L}_{text} = -\sum_{j=1}^{m} \log P(\hat{t}_j | t_j)$$

- $\hat{t}_j$: 模型预测的第 $j$ 个文本token
- $t_j$: 真实的第 $j$ 个文本token（Ground Truth）
- $P(\hat{t}_j | t_j)$: 模型在给定上下文下预测正确token的概率
- $m$: 文本序列长度
- **含义**: 标准的自回归语言建模损失，保持MLLM的文本生成能力
- **作用**: 确保理解任务的性能不被生成训练破坏

#### 4.2.2 Vision Adapter — 将视觉Token转化为扩散模型的条件

**公式(3) — Adapter映射**:

$$Q = \text{Adapter}(V_o)$$

- $V_o$: Vision Head输出的连续视觉token序列
- $Q$: Adapter输出的条件信号，送入扩散解码器
- $\text{Adapter}(\cdot)$: 轻量级适配器网络（将Vision Head的输出空间映射到扩散模型的条件空间）
- **含义**: 桥接MLLM输出空间和扩散模型条件空间的域差距
- **作用**: 使扩散解码器能理解MLLM生成的视觉线索

**训练策略**: 分两步训练Adapter：
1. 先对齐Adapter的输出与扩散模型文本编码器的textual embedding空间
2. 再用扩散模型的训练目标（flow matching）微调

**公式(4) — 扩散解码器训练目标 (Flow Matching)**:

$$\mathcal{L}_{DMs}(\theta) = E_{(X_1,Q) \sim D, t \sim U(0,1), X_0 \sim N(0,1)} \left[\|V_\theta(X_t, Q, t) - V_t\|^2\right]$$

- $\theta$: Adapter的参数（此阶段只更新Adapter）
- $X_1$: 目标图像/视频的VAE潜变量
- $X_0$: 标准高斯噪声
- $X_t$: $X_0$ 和 $X_1$ 在时间步 $t$ 的线性插值（flow matching的中间状态）
- $Q$: Adapter的输出（条件信号）
- $t \sim U(0,1)$: 均匀分布采样的时间步
- $V_\theta(X_t, Q, t)$: 模型预测的速度场（velocity field），即在 $X_t$ 处沿从噪声到数据的路径的速度
- $V_t$: 真实的目标速度（由 $X_0, X_1, t$ 确定）
- **含义**: 最小化模型预测速度和真实速度之间的差距
- **作用**: 教会扩散解码器根据Adapter输出的条件信号生成正确的图像/视频

#### 4.2.3 编辑模式的统一条件序列

在编辑模式下，Adapter的输入被扩展为多模态拼接序列：

$$\mathbf{c} = [\mathbf{v}_1, \ldots, \mathbf{v}_T \;\|\; \mathbf{h}_1, \ldots, \mathbf{h}_{T'} \;\|\; \mathbf{t}_1, \ldots, \mathbf{t}_m]$$

- $\mathbf{v}_t$: Vision Head输出的视觉token（MLLM的高级语义理解）
- $\mathbf{h}_t$: 源视频/图像经3D-Causal-VAE编码的时空嵌入（低级视觉细节）
- $\mathbf{t}_1, \ldots, \mathbf{t}_m$: 原始文本条件（文本编码器输出的嵌入）

**为什么需要 $\mathbf{h}_t$**:
- MLLM主要处理高级语义特征，不擅长保留低级视觉细节
- T2V扩散模型擅长从噪声中恢复细粒度视觉信息
- 通过VAE嵌入直接送入扩散模型，原始信号可以在所有去噪步骤中与含噪嵌入交互，确保编辑过程中未修改区域的高保真保留

### 4.3 Python代码示例

```python
import torch
import torch.nn as nn

class OmniVideo(nn.Module):
    """Omni-Video架构简化示意"""

    def __init__(self, mllm, diffusion_decoder, visual_encoder, vae):
        super().__init__()
        # 核心组件（均使用预训练权重初始化）
        self.mllm = mllm                    # MLLM骨干 (e.g., VILA)
        self.diffusion_decoder = diffusion_decoder  # T2V扩散模型 (e.g., Wan 2.1-1.3B)
        self.visual_encoder = visual_encoder  # 视觉编码器 (e.g., SigLIP-v2)
        self.vae = vae                        # 3D-Causal-VAE

        # 新增轻量级组件（需要训练）
        self.text_head = nn.Linear(mllm.hidden_size, mllm.vocab_size)
        self.vision_head = nn.Linear(mllm.hidden_size, visual_encoder.embed_dim)
        self.vision_adapter = VisionAdapter(
            input_dim=visual_encoder.embed_dim,
            output_dim=diffusion_decoder.cond_dim
        )

        # 特殊token
        self.special_tokens = ['<BOI>', '<EOI>', '<BOV>', '<EOV>']

    def forward_understanding(self, text_tokens, visual_embeddings):
        """理解模式: 输入文本+图像/视频，输出文本回答"""
        # 拼接输入序列
        input_seq = torch.cat([text_tokens, visual_embeddings], dim=1)

        # MLLM前向传播
        hidden_states = self.mllm(input_seq)

        # Text Head 输出文本token
        text_logits = self.text_head(hidden_states)
        return text_logits

    def forward_generation(self, text_tokens, visual_embeddings=None):
        """生成模式: 输入文本(+可选视觉输入)，输出图像/视频"""
        # MLLM前向传播
        if visual_embeddings is not None:
            input_seq = torch.cat([text_tokens, visual_embeddings], dim=1)
        else:
            input_seq = text_tokens

        hidden_states = self.mllm(input_seq)

        # Vision Head 输出连续视觉token
        # 在<BOI>/<BOV>到<EOI>/<EOV>之间的hidden state上应用
        vision_tokens = self.vision_head(hidden_states)  # V_o

        # Adapter: 映射到扩散模型的条件空间
        condition = self.vision_adapter(vision_tokens)    # Q

        # 扩散解码器生成图像/视频
        generated = self.diffusion_decoder.generate(condition=condition)
        return generated

    def forward_editing(self, text_tokens, visual_embeddings,
                        source_video):
        """编辑模式: 输入指令+源视频，输出编辑后的视频"""
        # MLLM处理输入
        input_seq = torch.cat([text_tokens, visual_embeddings], dim=1)
        hidden_states = self.mllm(input_seq)

        # Vision Head 输出视觉token
        vision_tokens = self.vision_head(hidden_states)  # v_1, ..., v_T

        # 从源视频提取VAE嵌入（低级细节）
        source_vae_embeddings = self.vae.encode(source_video)  # h_1, ..., h_T'

        # 获取文本嵌入
        text_embeddings = self.diffusion_decoder.text_encoder(text_tokens)  # t_1, ..., t_m

        # 拼接为统一条件序列
        unified_condition = torch.cat([
            vision_tokens,           # v_t: MLLM的高级语义
            source_vae_embeddings,   # h_t: 源视频的低级细节
            text_embeddings          # t_m: 文本条件
        ], dim=1)  # c = [v || h || t]

        # Adapter映射
        condition = self.vision_adapter(unified_condition)

        # 扩散解码器生成编辑后的视频
        edited_video = self.diffusion_decoder.generate(condition=condition)
        return edited_video

    def compute_loss(self, batch):
        """训练损失计算"""
        # 视觉对齐损失: L_vision = ||V_o - E||^2
        V_o = self.vision_head(hidden_states)
        E = self.visual_encoder(target_visual)
        loss_vision = torch.mean((V_o - E) ** 2)

        # 文本损失: L_text = -Σ log P(t_j | t_j)
        text_logits = self.text_head(hidden_states)
        loss_text = nn.CrossEntropyLoss()(text_logits, target_text)

        # 扩散损失 (Stage 2+): Flow Matching
        Q = self.vision_adapter(V_o)
        X_1 = self.vae.encode(target_visual)
        X_0 = torch.randn_like(X_1)
        t = torch.rand(1)
        X_t = (1 - t) * X_0 + t * X_1  # 线性插值
        V_pred = self.diffusion_decoder(X_t, Q, t)
        V_target = X_1 - X_0  # 目标速度
        loss_diffusion = torch.mean((V_pred - V_target) ** 2)

        return loss_vision + loss_text + loss_diffusion


class VisionAdapter(nn.Module):
    """轻量级视觉适配器"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, vision_tokens):
        return self.projector(vision_tokens)
```

---

## 五、推理流程

### 5.1 理解模式

```
输入: 视频/图像 + 用户问题（文本）
  │
  ├→ [Visual Encoder] → 视觉嵌入
  │
  ├→ [Tokenizer] → 文本Token
  │
  └→ [MLLM] → [Text Head] → 自回归生成文本回答
```

### 5.2 生成模式

```
输入: 文本提示 (e.g., "A man is surfing on...")
  │
  └→ [Tokenizer] → 文本Token
       │
       └→ [MLLM] → [Vision Head] → 视觉Token V_o
                                      │
                                      └→ [Adapter] → 条件Q
                                           │
                                           └→ [Diffusion Decoder] → 生成的视频/图像
```

### 5.3 编辑模式

```
输入: 源视频 + 编辑指令（文本）
  │
  ├→ [Visual Encoder] → 源视频的视觉嵌入
  │
  ├→ [VAE Encoder] → 源视频的VAE嵌入 (低级细节)
  │
  ├→ [Tokenizer] → 指令文本Token
  │
  └→ [MLLM] → [Vision Head] → 视觉Token V_o
                                    │
         ┌──────────────────────────┤
         │                          │
         ▼                          ▼
  [Vision Token] ∥ [VAE嵌入] ∥ [文本嵌入] = 统一条件序列 c
                          │
                          └→ [Adapter] → 条件Q
                               │
                               └→ [Diffusion Decoder] → 编辑后的视频
```

### 5.4 Think Mode (思考模式)

```
输入: 文本提示
  │
  └→ [MLLM] → 先"思考"(reasoning rewrite)
                → 重写提示
                → [Vision Head] → 根据重写提示生成视觉token
                                    │
                                    └→ [Adapter] → [Diffusion Decoder] → 更高质量的视频
```

**Think Mode的作用**: 利用MLLM的链式思维(Chain-of-Thought)推理能力，先对用户指令进行推理和改写，产生更详细、更精确的描述，然后Vision Head根据这个重写的提示调整输出。**不需要重新训练扩散模型**。

---

## 六、训练流程

### 6.1 三阶段训练策略

#### Stage 1: 教会MLLM生成视觉连续Token

```
冻结: MLLM骨干, Diffusion Decoder
训练: Text Head, Vision Head
损失: L_vision + L_text
数据: 理解 + 生成数据混合
```

**目标**: 保持MLLM的理解能力，同时快速训练Vision Head产生有意义的视觉token。MLLM学会在看到生成指令时，在`<BOI>`/`<BOV>`和`<EOI>`/`<EOV>`之间生成视觉token。

#### Stage 2: 将视觉Token投射到扩散模型的条件空间

```
冻结: MLLM骨干, Text Head, Vision Head, Diffusion Decoder (大部分)
训练: Vision Adapter
损失: L_DMs (Flow Matching目标, 公式4)
数据: T2I, T2V, 图像编辑, 视频编辑 (四种任务)
```

**目标**: 训练Adapter对齐Vision Head输出与扩散模型的条件空间。先对齐文本嵌入空间，再用flow matching微调。

**两子阶段加速**:
1. 先只训练T2I和T2V任务（建立基础生成能力）
2. 再加入编辑任务联合训练

#### Stage 3: 联合微调提升视觉生成质量

```
冻结: MLLM骨干 (大部分), Visual Encoder, Tokenizer
训练: Vision Head, Adapter, Diffusion Decoder
损失: L_vision + L_text + L_DMs
数据: 更高质量的数据, 视频帧率从8fps提升到12fps
```

**目标**: 解冻更多参数，联合微调以提升整体生成质量。

### 6.2 混合批处理策略 (Hybrid Mixed-Batching)

训练数据分为两组：
1. 视频生成 + 视频编辑
2. 图像生成 + 图像编辑

每组内按相同分辨率和帧数构建batch，在单次前向+反向传播中处理，提高计算效率和参数更新频率。

### 6.3 模型初始化

| 组件 | 初始化来源 |
|------|----------|
| MLLM | VILA 预训练权重 |
| Diffusion Decoder | Wan 2.1-1.3B 预训练权重 |
| Visual Encoder | SigLIP-v2 预训练权重 |
| Vision Head | 随机初始化 |
| Adapter | 随机初始化 |

### 6.4 内存优化

- DeepSpeed optimizer partitioning + ZeRO Stage-1
- 序列并行 (Sequence Parallelism)：分布长条件序列的内存
- 梯度检查点 (Gradient Checkpointing)

---

## 七、训练数据集

### 7.1 数据组成

**总计约4530万样本**，覆盖6类任务：

| 类别 | 数据集 | 占比 |
|------|--------|------|
| **图像理解** | JourneyDB-4M, LLaVA-v1.5, ShareGPT4V, Cambrain-7M, MMC4 | 37.90% (~1720万) |
| **视频理解** | LaVA-Video-178K, VSTaR-1M, ShareGPT4Video-40K | 4.36% (~198万) |
| **T2I生成** | LAION-5B, COYO-700M | 33.05% (~1500万) |
| **T2V生成** | HD-VILA | 3.97% (~180万) |
| **图像编辑** | AnyEdit, UltraEdit, HIVE, PromptFix | 18.95% (~860万) |
| **视频编辑** | Senorita-2M | 1.76% (~80万) |

### 7.2 数据处理流程

- **质量过滤**: 分辨率、宽高比、美学评分筛选
- **内容过滤**: 水印检测、光流运动预测、OCR文字检测
- **文本重标注**: 使用MLLM基于视觉内容重新生成描述性文本

---

## 八、实验设计与结果

### 8.1 实验设置

**生成任务**:
- 采样策略: UniPC
- CFG Scale: T2I=5.0, T2V=3.0
- 采样步数: T2I=50, T2V=40
- 使用universal negative prompt

**理解任务**:
- Top-p: 0.6, Temperature: 0.2
- 最大新token数: 1024

### 8.2 实验1: T2I性能 (GenEval Benchmark)

**动机**: 验证统一模型的图像生成能力是否能与专用模型竞争。

**Table 2结果**:

| 方法 | Single Obj | Two Obj | Count | Color | Pos | Color Attri | Overall |
|------|-----------|---------|-------|-------|-----|-------------|---------|
| SDv1.5 | 0.97 | 0.38 | 0.35 | 0.76 | 0.04 | 0.06 | 0.43 |
| DALL-E 3 | 0.96 | 0.87 | 0.47 | 0.83 | 0.43 | 0.45 | 0.67 |
| JanusFlow | 0.97 | 0.59 | 0.45 | 0.83 | **0.53** | 0.42 | 0.63 |
| **Omni-Video (Ours)** | **0.99** | **0.89** | **0.84** | **0.87** | 0.35 | **0.56** | **0.75** |

**结论**: Omni-Video在GenEval上取得了最高的Overall分数(0.75)，大幅超过专用生成模型和其他统一模型。特别是在Count(0.84)和Color Attri(0.56)上领先明显。

### 8.3 实验2: T2V性能 (VBench-Long Benchmark)

**动机**: 验证长程视频生成质量。

**Table 3结果 (部分)**:

| 方法 | Total Score | Quality | Semantic |
|------|------------|---------|----------|
| Wan2.1-T2V-1.3B | 83.31 | 85.23 | 75.65 |
| **Omni-Video (Ours)** | 83.00 | 84.27 | 77.92 |

**亮点**: Subject Consistency (98.39%), Temporal Flickering (99.87%), Object Class (93.54%), Temporal Style (25.81%)均达到SOTA水平。在10/16个子维度上超过基线Wan2.1-T2V-1.3B。

### 8.4 实验3: 图像/视频编辑

**定性结果**:
- **图像编辑** (Figure 7): 成功进行风格迁移、植物替换等编辑，保持结构完整性
- **视频编辑** (Figure 8): 成功进行背景替换、物体移除、属性添加等操作，保持跨帧一致性

### 8.5 实验4: 视频理解

**定性结果** (Figure 9): 在多样化的视频场景中展示了强大的多模态推理能力，包括物体计数、动物识别、运动推断、情感分析、季节判断等。

### 8.6 实验5: Think Mode

**动机**: 验证MLLM的链式思维推理是否能提升生成质量。

**结果** (Figure 10): Think Mode下模型能更好地理解复杂描述（如"elegant marble mausoleum, Indian architectural brilliance"），先推理出需要分层的大理石纹理和精确的穹顶几何，再生成更精确的视频。

---

## 九、创新性贡献

1. **统一视频建模范式**: 首次提出一个高效的统一框架同时支持视频理解、生成和编辑。通过教会MLLM产生连续视觉token作为扩散解码器的条件，桥接了理解和生成。

2. **轻量级架构设计**: 仅添加Vision Head和Adapter两个轻量级组件，不修改MLLM和扩散模型的核心架构，最大化复用预训练权重。

3. **高效多阶段训练**: 三阶段渐进训练策略使得系统可以用有限数据和计算资源有效训练。

4. **Think Mode**: 创新性地利用MLLM的推理能力提升生成质量，无需重训扩散模型。

5. **编辑模式的VAE嵌入注入**: 通过将源视频的VAE嵌入直接送入扩散模型（而非MLLM），实现低级视觉细节的保留。

---

## 十、不足之处与改进方向

### 10.1 不足

1. **数据可扩展性**: 高质量视频编辑数据稀缺，限制了编辑能力的上限
2. **计算约束**: 高保真长时视频合成需要大量内存，受限于当前硬件
3. **架构碎片化**: MLLM和扩散模型的解耦设计导致优化不对齐——为生成优化的潜表示可能无法无缝迁移到需要逐帧因果推理的理解任务
4. **扩散模型规模小**: 仅使用Wan 2.1-1.3B（较小版本），生成质量有进一步提升空间

### 10.2 改进方向

1. **优化diffusion-MLLM接口**: 通过共享优化来减少表示不对齐
2. **自适应训练协议**: 针对复杂物理交互开发适应性训练方案
3. **可扩展数据管线**: 构建更大规模、更高质量的视频编辑训练数据
4. **扩大模型规模**: 使用更大的扩散模型（如14B版本）

---

---

# 第二部分: Omni-Video 2

---

## 一、背景与动机

### 1.1 Omni-Video v1 的局限

Omni-Video v1 虽然成功统一了视频理解、生成和编辑，但仍存在关键问题：

1. **扩散模型规模小** (1.3B)，生成质量有限
2. **编辑指令跟随能力弱**: 自由形式的编辑提示常常模糊、不完整，直接作为T2V模型的条件会导致不稳定的编辑
3. **新条件信号干扰预训练接口**: 统一训练引入的新条件（源视频引用、编辑指令等）会干扰扩散模型预训练时学到的text-to-video条件接口，导致**分布漂移 (distribution drift)** 和**灾难性退化**

### 1.2 Omni-Video 2 的核心目标

> **最大化复用预训练T2V基础模型，同时在有限额外训练下实现统一的生成和编辑能力。**

### 1.3 核心设计洞察

1. **MLLM作为Editing Prompt Reasoner**: 利用MLLM的推理能力将模糊的编辑指令转化为明确的目标描述(target caption)，使扩散模型继续通过其预训练的caption接口接收引导
2. **新条件作为添加性优化而非替换**: 所有新增条件信号（MLLM特征、编辑指令等）作为预训练条件的**补充(additive refinements)**，而非**替换(replacements)**，避免分布漂移

---

## 二、模型架构详解

### 2.1 整体架构

Omni-Video 2由两个核心组件构成：

```
          ┌─────────────────────────────────────────┐
          │        Editing Prompt Reasoner           │
          │                                          │
          │  System Prompt: "Predict the edited      │
          │  video caption after editing"             │
          │                                          │
          │  User Instruction: "Change the cat        │
          │  to white"  + 源视频                      │
          │         │                                │
          │         ▼                                │
          │      [MLLM]                              │
          │         │                                │
          │         ├──→ Predicted Target Caption:   │
          │         │    p̂^tgt                       │
          │         │                                │
          │         └──→ Cross-modal Interaction     │
          │              Features: H^MLLM            │
          └──────────┬──────────────┬────────────────┘
                     │              │
                     ▼              ▼
          ┌──────────────────────────────────────────┐
          │     Multimodal Condition Adapter          │
          │                                          │
          │  C^mllm ← Projector(H^MLLM)             │
          │  C^tgt  ← T5_Encoder(p̂^tgt)             │
          │  C^edit ← T5_Encoder(p^edit)             │
          │  C^ref  ← VAE_Encoder(x^src)             │
          │                                          │
          │  C = [C^mllm ; C^tgt ; C^edit ; C^ref]  │
          │         │                                │
          │    [Projector] → ⊕ →                     │
          └────────────────┬─────────────────────────┘
                           │
                           ▼ Cross-Attention
          ┌────────────────────────────────────┐
          │      Diffusion Decoder (DiT)       │
          │  ┌──────────────────────────┐      │
          │  │ Layer Norm → Self-Attn   │      │
          │  │ Layer Norm → Cross-Attn  │ × N  │
          │  │ Layer Norm → FFN         │      │
          │  └──────────────────────────┘      │
          │  Noisy Latent → Denoised Output    │
          └────────────────────────────────────┘
```

### 2.2 Editing Prompt Reasoner (编辑提示推理器)

给定源视频 $x^{src}$ 和编辑指令 $p^{edit}$，MLLM进行联合多模态推理并预测目标视频描述：

$$\hat{p}^{tgt} = \text{MLLM}(x^{src}, p^{edit})$$

- $x^{src}$: 源视频
- $p^{edit}$: 用户编辑指令（如 "Change the cat to white"）
- $\hat{p}^{tgt}$: 预测的目标描述（如 "A white cat is walking on a concrete block floor, with many pink flowers in the background"）
- $\text{MLLM}$: 多模态大语言模型

**为什么需要这个组件？**

自由形式的编辑提示存在三个问题：
1. **模糊性**: "Change the cat to white" 没有说明场景的其他部分长什么样
2. **依赖上下文**: 可能隐式引用源视频中的物体而不明确命名
3. **与预训练分布不匹配**: T2V模型预训练时用的是详细的描述性caption，而非编辑指令

Editing Prompt Reasoner 将模糊的编辑指令转化为与T2V预训练分布一致的详细目标描述，**保留了扩散模型原始的caption条件接口**，最大化复用预训练先验。

### 2.3 Multimodal Condition Adapter (多模态条件适配器)

Omni-Video 2 通过四个条件源引导扩散模型：

#### 条件源1: MLLM目标描述 $C^{tgt}$

$$C^{tgt} \in \mathbb{R}^{L_{tgt} \times d_{txt}}$$

- 由T5文本编码器对预测的目标描述 $\hat{p}^{tgt}$ 编码得到
- **作用**: 作为主要语义引导，最大化复用预训练T2V的caption条件路径

#### 条件源2: 编辑指令 $C^{edit}$

$$C^{edit} \in \mathbb{R}^{L_{edit} \times d_{txt}}$$

- 由T5文本编码器对原始编辑指令 $p^{edit}$ 编码得到
- **作用**: 提供补充的指令级信息

#### 条件源3: MLLM跨模态交互特征 $C^{mllm}$

$$H^{MLLM} \in \mathbb{R}^{L \times d_{mllm}}$$

- 从MLLM最后一层的hidden states中提取
- $L$: 多模态条件token的总数（包含文本token和源视频的视觉token）
- $d_{mllm}$: MLLM的隐藏维度

通过线性投影映射到扩散模型的隐空间：

$$C^{mllm} = \text{Projector}(H^{MLLM}) \in \mathbb{R}^{L_{mllm} \times d_{dit}}$$

- **作用**: 捕获MLLM对指令和视觉输入的联合推理结果，提供融合的语义特征

#### 条件源4: 源视频VAE参考 $C^{ref}$

$$C^{ref} \in \mathbb{R}^{L_c \times d_{dit}}$$

- 从源视频的VAE潜变量计算得到
- **作用**: 提供持久的视觉锚点，保留源视频的身份、结构和时序一致性

#### 统一条件序列

所有条件拼接为单一序列：

$$C = [C^{mllm}; C^{tgt}; C^{edit}; C^{ref}] \in \mathbb{R}^{L_c \times d_{dit}}$$

通过标准cross-attention注入DiT的每个block：

$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right) V$$

其中 $K, V$ 从统一条件序列 $C$ 导出。

**关键设计原则**: 所有DiT block共享相同的条件接口和Adapter输出，不引入block-specific的注入。

### 2.4 多模态条件训练 — 随机条件Dropout

**公式(1) — 条件Dropout**:

$$\tilde{C}^{mllm} = m_{mllm} \cdot C^{mllm}, \quad \tilde{C}^{tgt} = m_{tgt} \cdot C^{tgt}, \quad \tilde{C}^{edit} = m_{edit} \cdot C^{edit}, \quad \tilde{C}^{ref} = C^{ref}$$

- $m \in \{0, 1\}$: 独立的Bernoulli掩码
- 丢弃概率: $p_{mllm}, p_{tgt}, p_{edit}$
- **注意**: $C^{ref}$ **永不丢弃**！因为源视频的VAE参考是结构性锚点

最终条件序列:

$$\tilde{C} = [\tilde{C}^{mllm}; \tilde{C}^{tgt}; \tilde{C}^{edit}; \tilde{C}^{ref}]$$

**三重作用**:
1. **推理鲁棒性**: 某些条件可能缺失时模型仍能工作
2. **保留caption路径**: 偶尔丢弃辅助条件，迫使模型依赖caption引导，强化预训练的T2V先验
3. **防止过拟合**: 防止Adapter过度拟合高维MLLM特征

### 2.5 Flow Matching训练目标

**公式(2) — Flow Matching损失**:

$$\mathcal{L}_{flow} = \mathbb{E}_{z,\epsilon,u} \left[\|v_\theta(z_u, u \mid \tilde{C}) - v^\star(z, \epsilon, u)\|_2^2\right]$$

**符号说明**:
- $z$: 目标视觉内容的VAE潜变量
- $u \sim \mathcal{U}[0, 1]$: 连续时间参数，从均匀分布采样
- $z_u$: 在预定义路径上 $z$ 和 $\epsilon$ 之间的插值潜变量（如线性插值 $z_u = (1-u) \cdot \epsilon + u \cdot z$）
- $\epsilon \sim \mathcal{N}(0, I)$: 标准高斯噪声
- $v_\theta(z_u, u \mid \tilde{C})$: 模型预测的条件速度场（在潜空间中从噪声到数据的运输方向）
- $v^\star(z, \epsilon, u)$: 由选定的路径参数化决定的目标速度（如 $v^\star = z - \epsilon$）
- $\tilde{C}$: 经过随机dropout后的条件序列
- **含义**: 学习一个将噪声运输为数据的速度场，条件于多模态条件信号
- **作用**: 这是Flow Matching框架的核心训练目标，比传统DDPM在训练效率和生成质量上都有优势

### 2.6 Python代码示例

```python
import torch
import torch.nn as nn

class OmniVideo2(nn.Module):
    """Omni-Video 2 架构简化示意"""

    def __init__(self, mllm, dit_backbone, t5_encoder, vae):
        super().__init__()
        # 冻结组件
        self.mllm = mllm              # MLLM (冻结，仅用于推理)
        self.t5_encoder = t5_encoder   # T5文本编码器 (冻结)
        self.vae = vae                 # VAE编码器/解码器

        # 可训练组件
        self.dit = dit_backbone        # 14B DiT (可训练)
        self.adapter = MultimodalConditionAdapter(
            mllm_dim=mllm.hidden_size,
            text_dim=t5_encoder.hidden_size,
            dit_dim=dit_backbone.hidden_size
        )

        # Dropout概率
        self.p_mllm = 0.1
        self.p_tgt = 0.1
        self.p_edit = 0.1

    def editing_prompt_reasoner(self, source_video, edit_instruction):
        """编辑提示推理器：将编辑指令转化为目标描述"""
        # MLLM联合推理
        system_prompt = "Predict the edited video caption after editing."
        mllm_output = self.mllm.generate(
            system_prompt=system_prompt,
            user_input=edit_instruction,
            visual_input=source_video
        )

        # 提取两个输出
        target_caption = mllm_output['text']           # p̂^tgt
        cross_modal_features = mllm_output['hidden']   # H^MLLM

        return target_caption, cross_modal_features

    def build_conditions(self, target_caption, edit_instruction,
                         cross_modal_features, source_video=None,
                         training=False):
        """构建统一条件序列"""
        # C^tgt: 目标描述的T5编码
        C_tgt = self.t5_encoder(target_caption)

        # C^edit: 编辑指令的T5编码
        C_edit = self.t5_encoder(edit_instruction)

        # C^mllm: MLLM跨模态特征的线性投影
        C_mllm = self.adapter.mllm_projector(cross_modal_features)

        # C^ref: 源视频的VAE编码（如果有的话）
        if source_video is not None:
            C_ref = self.vae.encode(source_video)
            C_ref = self.adapter.ref_projector(C_ref)
        else:
            C_ref = None

        # 训练时进行随机条件Dropout
        if training:
            # Bernoulli mask (C^ref 永不dropout)
            if torch.rand(1) < self.p_mllm:
                C_mllm = torch.zeros_like(C_mllm)
            if torch.rand(1) < self.p_tgt:
                C_tgt = torch.zeros_like(C_tgt)
            if torch.rand(1) < self.p_edit:
                C_edit = torch.zeros_like(C_edit)

        # 拼接为统一条件序列
        conditions = [C_mllm, C_tgt, C_edit]
        if C_ref is not None:
            conditions.append(C_ref)

        C = torch.cat(conditions, dim=1)  # [B, L_c, d_dit]
        return C

    def compute_loss(self, batch):
        """Flow Matching训练损失"""
        target_video = batch['target_video']
        edit_instruction = batch['edit_instruction']
        source_video = batch.get('source_video', None)
        target_caption = batch['target_caption']

        # 获取MLLM特征 (MLLM冻结，只做前向推理)
        with torch.no_grad():
            _, cross_modal_features = self.editing_prompt_reasoner(
                source_video, edit_instruction
            )

        # 构建条件
        C = self.build_conditions(
            target_caption, edit_instruction,
            cross_modal_features, source_video,
            training=True
        )

        # Flow Matching
        z = self.vae.encode(target_video)      # 目标视频的VAE编码
        epsilon = torch.randn_like(z)          # 标准高斯噪声
        u = torch.rand(z.shape[0], 1, 1, 1)   # 均匀采样时间步

        # 线性插值
        z_u = (1 - u) * epsilon + u * z        # 中间状态

        # 模型预测速度
        v_pred = self.dit(z_u, u, condition=C)

        # 目标速度
        v_target = z - epsilon

        # Flow Matching损失
        loss = torch.mean((v_pred - v_target) ** 2)
        return loss

    @torch.no_grad()
    def generate(self, prompt, source_video=None, edit_instruction=None):
        """推理：视频生成或编辑"""
        if edit_instruction is not None and source_video is not None:
            # 编辑模式
            target_caption, cross_modal_features = \
                self.editing_prompt_reasoner(source_video, edit_instruction)

            C = self.build_conditions(
                target_caption, edit_instruction,
                cross_modal_features, source_video
            )
        else:
            # 纯生成模式
            C_tgt = self.t5_encoder(prompt)
            C = self.adapter.tgt_projector(C_tgt)

        # ODE采样 (UniPC等)
        z = torch.randn(...)  # 初始噪声
        for step in sampling_steps:
            v = self.dit(z, step, condition=C)
            z = ode_step(z, v, step)

        video = self.vae.decode(z)
        return video


class MultimodalConditionAdapter(nn.Module):
    """多模态条件适配器"""
    def __init__(self, mllm_dim, text_dim, dit_dim):
        super().__init__()
        self.mllm_projector = nn.Linear(mllm_dim, dit_dim)
        self.tgt_projector = nn.Linear(text_dim, dit_dim)
        self.edit_projector = nn.Linear(text_dim, dit_dim)
        self.ref_projector = nn.Linear(dit_dim, dit_dim)  # VAE dim ≈ dit_dim

    def forward(self, C_mllm, C_tgt, C_edit, C_ref=None):
        C_mllm = self.mllm_projector(C_mllm)
        C_tgt = self.tgt_projector(C_tgt)
        C_edit = self.edit_projector(C_edit)
        parts = [C_mllm, C_tgt, C_edit]
        if C_ref is not None:
            C_ref = self.ref_projector(C_ref)
            parts.append(C_ref)
        return torch.cat(parts, dim=1)
```

---

## 三、训练流程

### 3.1 核心训练原则

> **所有新能力通过条件注入和轻量级Adapter引入，不修改DiT核心架构。**

- DiT骨干从预训练T2V模型初始化，**结构不变**
- MLLM全程冻结，仅用于推理
- 可训练参数: DiT + Adapter

### 3.2 训练规模

- **模型**: 两个14B参数的DiT（low-noise model + high-noise model）
- **硬件**: 1600块 Alibaba PPU GPU
- **优化**: DeepSpeed ZeRO-1, 混合精度训练, 梯度检查点

### 3.3 大规模训练的关键优化: 序列并行

**问题**: 在14B规模下，简单增加GPU数量和batch size并不能加速收敛。实测中1600 GPU每天仅完成约4000步。

**解决方案**: 采用Ulysses-style序列并行，将序列并行同时应用于DiT的**self-attention和cross-attention**层。

**原因**: Omni-Video 2的cross-attention条件序列很长（MLLM特征 + 目标caption + 编辑指令 + 源视频参考的拼接），cross-attention占比显著。

**效果**: 序列并行度=8时，每步训练获得4-5x加速，大幅提升收敛速度。

---

## 四、数据构建

### 4.1 数据规模与组成

总计超过**100万**训练实例，涵盖四大任务族：

| 任务 | 占比 |
|------|------|
| Text-to-Image | ~39% |
| Text-to-Video | ~20% |
| Image Editing | ~20% |
| Video Editing | ~22% |

(参见Figure 4的分布饼图)

### 4.2 数据来源

- **真实数据 (Real-world data)**: 提供多样化的视觉内容和自然语言描述，用于维持T2V的生成质量
- **合成数据 (Synthetic data)**: 通过合成构建精确的编辑对 (source-target pairs)，强化编辑任务的监督

### 4.3 视频编辑数据的6类指令

| 类别 | 描述 | 示例 |
|------|------|------|
| **Local Add** | 在局部区域添加物体 | "Add safety goggles to the scientist" |
| **Local Remove** | 移除局部物体 | "Remove the meditation cushion" |
| **Local Replace** | 替换局部物体 | "Change the fox into a badger" |
| **Global Edit** | 全局场景变化 | "Change the background to ocean" |
| **Change Attribute** | 修改属性 | "Change the dress color to red" |
| **Complex Edit** | 多约束复合编辑 | "Change jacket to trench coat AND replace fire hydrant with chrome one" |

### 4.4 统一样本格式

每个训练样本包含：
- 任务类型指示 (T2I/T2V/I2I/video-editing)
- (可选) 源视觉输入 $x^{src}$
- 编辑指令 $p^{edit}$ (对于生成任务简化为描述性提示)
- 目标描述 $p^{tgt}$ (由MLLM预测或数据集提供)
- 目标视觉输出

**关键**: 无论任务类型，都包含target caption，确保所有实例都锚定于caption风格的语义描述，强化caption引导路径。

### 4.5 四阶段数据清洗

| 阶段 | 内容 |
|------|------|
| **Stage 1: 完整性验证** | 移除损坏文件、帧数不足、分辨率/时长超范围的样本；去重 |
| **Stage 2: 视觉质量过滤** | 过滤压缩伪影、水印、字幕、退化运动、近静态视频 |
| **Stage 3: 文本-视觉一致性** | 比较描述与自动生成caption的一致性；验证编辑指令与目标caption的一致性 |
| **Stage 4: 编辑专用验证** | 对局部编辑验证差异是否空间/时间集中；对复杂编辑进行额外抽查 |

---

## 五、实验设计与结果

### 5.1 评估基准

| 基准 | 任务 | 评估维度 |
|------|------|----------|
| **FiVE-Bench** | 视频编辑 | 细粒度编辑：物体替换、颜色变化、材质修改、物体添加、物体移除 |
| **VBench** | 视频生成 | 16个维度：语义对齐、时序一致性、运动自然性、视觉质量等 |

### 5.2 实验1: 视频编辑 (FiVE-Bench)

**动机**: 验证Omni-Video 2在精细粒度、指令驱动的视频编辑上是否超越现有方法。

**Table 1 — FiVE-Bench结果**:

| 方法 | FiVE-YN | FiVE-MC | FiVE-∪ | FiVE-∩ | FiVE-Acc↑ |
|------|---------|---------|--------|--------|-----------|
| TokenFlow | 19.36 | 35.51 | 36.68 | 18.18 | 27.43 |
| AnyV2V | 30.62 | 45.42 | 48.96 | 27.09 | 38.02 |
| Wan-Edit | 41.41 | 52.53 | 55.72 | 38.22 | 46.97 |
| UniVideo | 56.50 | 68.55 | 69.95 | 55.10 | 62.53 |
| **Omni-Video 2 (Ours)** | **63.77** | **83.30** | **85.99** | **61.08** | **73.53** |

**结论**: Omni-Video 2以73.53的FiVE-Acc大幅超过第二名UniVideo的62.53（+11.0绝对值），在所有子指标上全面领先。这表明MLLM的推理能力有效提升了复杂组合编辑指令的理解和执行。

### 5.3 实验2: 视频生成 (VBench)

**动机**: 验证统一编辑训练是否会损害原始T2V生成质量。

**Table 2 — VBench结果**:

| 方法 | Total Score | Quality | Semantic |
|------|------------|---------|----------|
| EasyAnimateV5.1 | 83.42 | 85.03 | 77.01 |
| Wan2.1-T2V-1.3B | 83.31 | 85.23 | 75.65 |
| HunyuanVideo | 83.24 | 85.09 | 75.82 |
| CogVideoX1.5-5B | 82.17 | 82.78 | 79.76 |
| **Omni-Video 2 (Ours)** | **84.69** | **85.79** | **80.28** |

**结论**: Omni-Video 2不仅没有因编辑训练导致生成质量退化，反而在Total Score(84.69)和Semantic(80.28)上取得最高分。这直接验证了核心设计原则：**通过轻量级条件Adapter和caption锚定，新能力的引入不干扰预训练的T2V先验**。

### 5.4 实验3: 定性编辑结果

论文展示了6类编辑任务的丰富定性结果（Figure 5-11）：

| Figure | 编辑类型 | 示例 |
|--------|---------|------|
| Fig. 5 | 添加局部物体 | 给科学家戴安全眼镜、给鸭子戴草帽 |
| Fig. 6 | 移除局部物体 | 移除冥想垫、移除画笔 |
| Fig. 7 | 局部变化 | 将白裙改为黑长袍、将狐狸改为獾 |
| Fig. 8 | 肖像属性编辑 | 将红裙改为白色金线礼服 |
| Fig. 9 | 全局属性编辑 | 更改光照、更换背景天空 |
| Fig. 10 | 复杂运动动态 | 在高运动场景中更改服装 |
| Fig. 11 | 复杂多部分指令 | 同时更改多个物体+背景+光照 |

**关键观察**: Omni-Video 2在处理复杂、多约束的编辑指令时表现尤为突出（Figure 11），这归功于MLLM Editing Prompt Reasoner的推理能力。

---

## 六、创新性贡献

### Omni-Video v1 的贡献

1. **统一视频建模范式**: 首次提出高效统一视频理解、生成和编辑的框架
2. **轻量级双头+Adapter设计**: 最小化架构修改
3. **三阶段渐进训练**: 资源友好的训练策略
4. **Think Mode**: 利用MLLM推理提升生成质量

### Omni-Video 2 的额外贡献

1. **Editing Prompt Reasoner**: 创新性地利用MLLM将编辑指令转化为明确的目标描述，桥接编辑指令与T2V模型的caption条件接口。这是区别于所有同期工作的核心创新。

2. **Multimodal Condition Adapter**: 设计了四源条件注入机制，所有新条件作为预训练条件的补充而非替换，有效防止分布漂移。

3. **随机条件Dropout训练策略**: 通过选择性丢弃条件，保持对caption路径的依赖，强化预训练先验。

4. **规模化验证**: 首次将MLLM-Conditioned Diffusion扩展到14B参数规模，证明该范式在大规模下依然有效。

5. **全面的数据构建方法论**: 构建了覆盖6类编辑指令的统一训练数据集，配合4阶段清洗流程。

---

## 七、不足之处与改进方向

### 7.1 Omni-Video v1 的不足

1. **数据可扩展性**: 高质量视频编辑数据稀缺
2. **计算约束**: 长视频合成内存需求高
3. **架构碎片化**: MLLM和扩散模型的解耦导致优化不对齐
4. **扩散模型规模小**: 仅1.3B

### 7.2 Omni-Video 2 的不足（论文虽未显式列出Limitations节，但可从设计和实验中推断）

1. **MLLM冻结限制**: MLLM全程冻结意味着无法根据编辑任务的特殊需求微调理解能力，可能在处理非常规指令时受限

2. **推理成本高**: 推理时需要先运行MLLM生成target caption和提取MLLM特征，再运行扩散模型去噪，总推理时间较长

3. **条件序列长度**: 四个条件源拼接后序列很长，增加cross-attention的计算开销

4. **编辑精度限制**: 尽管FiVE-Acc达到73.53，仍有约26%的情况编辑不准确，特别是在超精细局部编辑（如保持特定纹理细节）上

5. **视频长度限制**: 受限于DiT的训练配置和显存

### 7.3 改进方向

1. **MLLM部分微调**: 在Adapter训练的后期，可以考虑用LoRA等方法微调MLLM的部分参数，增强其对编辑任务的适应性

2. **条件压缩**: 研究更高效的条件表示方式（如token pruning、条件蒸馏），减少cross-attention序列长度

3. **渐进式生成**: 对长视频采用滑动窗口或自回归方式生成，突破帧数限制

4. **编辑质量增强**: 引入编辑区域mask、深度信息等额外控制信号提高编辑精度

5. **统一的MLLM-Diffusion共同训练**: 探索端到端的联合优化，解决v1中提到的"架构碎片化"问题

6. **更多编辑模态**: 支持audio-guided editing、video-to-video style transfer等更多模态的编辑

---

## 八、Omni-Video v1 vs v2 对比

| 维度 | Omni-Video v1 | Omni-Video 2 |
|------|--------------|--------------|
| **扩散模型** | Wan 2.1-1.3B | 14B DiT |
| **MLLM** | VILA (可训练) | MLLM (冻结) |
| **核心创新** | Vision Head + Adapter双头设计 | Editing Prompt Reasoner + 多模态条件Adapter |
| **编辑条件** | VAE嵌入 + Vision Token | 4源条件: MLLM特征 + Target Caption + Edit指令 + VAE参考 |
| **训练策略** | 3阶段渐进训练 | 条件Dropout + 序列并行 |
| **编辑基准** | 定性展示 | FiVE-Bench定量评估 (73.53 FiVE-Acc SOTA) |
| **生成基准** | VBench-Long | VBench (Total 84.69 SOTA) |
| **训练规模** | 未详述 | 1600 GPU, 14B参数 |
| **设计哲学** | 教MLLM产生视觉token | 最大复用预训练T2V先验 |

---

## 九、总结速记表

### Omni-Video v1

| 维度 | 内容 |
|------|------|
| **论文** | Omni-Video: Democratizing Unified Video Understanding and Generation |
| **时间** | 2025 |
| **核心思想** | 教MLLM产生连续视觉token → Adapter → 扩散解码器生成视频 |
| **架构** | MLLM(VILA) + Vision Head + Adapter + T2V(Wan 2.1-1.3B) |
| **支持任务** | 视频理解、T2I、T2V、图像编辑、视频编辑 |
| **训练策略** | 三阶段: ①训练Head ②训练Adapter ③联合微调 |
| **关键公式** | $\mathcal{L}_{vision}=\|V_o-E\|^2$; $\mathcal{L}_{DMs}$ (Flow Matching) |
| **特色功能** | Think Mode (链式思维提升生成质量) |
| **数据规模** | ~4530万样本 (6类任务) |
| **生成评估** | GenEval 0.75 (SOTA), VBench-Long 83.00 |
| **主要优势** | 轻量级统一、资源友好、支持多任务 |
| **主要局限** | 扩散模型小(1.3B)、架构解耦导致优化不对齐 |

### Omni-Video 2

| 维度 | 内容 |
|------|------|
| **论文** | Omni-Video 2: Scaling MLLM-Conditioned Diffusion for Unified Video Generation and Editing |
| **时间** | 2026 |
| **核心思想** | MLLM推理生成target caption + 多模态条件Adapter注入 → 保留T2V预训练先验 |
| **架构** | MLLM(冻结) + Editing Prompt Reasoner + 4源条件Adapter + 14B DiT |
| **支持任务** | T2I、T2V、图像编辑、视频编辑 (6类编辑指令) |
| **训练策略** | 条件Dropout + 序列并行(4-5x加速) + 仅训练DiT和Adapter |
| **关键公式** | $\mathcal{L}_{flow}$ (Flow Matching); 条件Dropout公式 |
| **四源条件** | $C^{mllm}$ (MLLM特征) + $C^{tgt}$ (目标caption) + $C^{edit}$ (编辑指令) + $C^{ref}$ (VAE参考) |
| **数据规模** | >100万实例 (4大任务族, 6类编辑) |
| **训练硬件** | 1600 Alibaba PPU GPU |
| **编辑评估** | FiVE-Acc 73.53 (超第二名11.0) |
| **生成评估** | VBench Total 84.69 (SOTA) |
| **核心设计原则** | 新条件 = 预训练条件的添加性优化，而非替换 |
| **主要优势** | 编辑指令跟随强、不损害生成质量、规模化验证 |
| **主要局限** | MLLM冻结、推理成本高、条件序列长 |
| **未来方向** | MLLM微调、条件压缩、长视频、端到端联合训练 |
