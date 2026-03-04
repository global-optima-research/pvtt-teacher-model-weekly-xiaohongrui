# Wan 系列视频生成模型深度调研文档

> 本文档覆盖 **Wan 2.1 技术报告**、**VACE 论文**、**Wan-Animate 论文** 三篇论文，结合 Wan2.1/Wan2.2 源码进行深度分析。

---

## 目录

- [一、总览：Wan 系列全景图](#一总览wan-系列全景图)
- [二、Wan 2.1 基座模型](#二wan-21-基座模型)
  - [2.1 背景与动机](#21-背景与动机)
  - [2.2 Wan-VAE：时空变分自编码器](#22-wan-vae时空变分自编码器)
  - [2.3 Wan-DiT：视频扩散Transformer](#23-wan-dit视频扩散transformer)
  - [2.4 Flow Matching 数学原理](#24-flow-matching-数学原理)
  - [2.5 T2V 文本生成视频](#25-t2v-文本生成视频)
  - [2.6 I2V 图像生成视频](#26-i2v-图像生成视频)
  - [2.7 视频个性化（Video Personalization）](#27-视频个性化video-personalization)
  - [2.8 实时视频生成（Real-Time Generation）](#28-实时视频生成real-time-generation)
  - [2.9 相机运动可控性（Camera Motion）](#29-相机运动可控性camera-motion)
  - [2.10 训练过程与数据集](#210-训练过程与数据集)
  - [2.11 创新贡献与不足](#211-创新贡献与不足)
- [三、VACE：统一视频创作与编辑框架](#三vace统一视频创作与编辑框架)
  - [3.1 背景与动机](#31-背景与动机)
  - [3.2 VCU：视频控制单元](#32-vcu视频控制单元)
  - [3.3 四大基础任务分类](#33-四大基础任务分类)
  - [3.4 Context Adapter 架构详解](#34-context-adapter-架构详解)
  - [3.5 训练过程与数据](#35-训练过程与数据)
  - [3.6 组合任务能力](#36-组合任务能力)
  - [3.7 创新贡献与不足](#37-创新贡献与不足)
- [四、Wan-Animate：统一角色动画与替换](#四wan-animate统一角色动画与替换)
  - [4.1 背景与动机](#41-背景与动机)
  - [4.2 模型架构](#42-模型架构)
  - [4.3 训练过程与数据](#43-训练过程与数据)
  - [4.4 创新贡献与不足](#44-创新贡献与不足)
- [五、Wan 2.2 升级解析](#五wan-22-升级解析)
  - [5.1 双专家模型策略（"MoE"）](#51-双专家模型策略moe)
  - [5.2 Wan 2.2 VAE 升级](#52-wan-22-vae-升级)
  - [5.3 TI2V 5B 高效管线](#53-ti2v-5b-高效管线)
- [六、核心疑难问题专题答疑](#六核心疑难问题专题答疑)
  - [Q1: 3D Causal Conv 详解](#q1-3d-causal-conv-详解)
  - [Q2: VAE 中的卷积操作](#q2-vae-中的卷积操作)
  - [Q3: Feature Cache 机制详解](#q3-feature-cache-机制详解)
  - [Q4: 视频 latent 如何 token 化](#q4-视频-latent-如何-token-化)
  - [Q5: VACE 是对 Wan 的封装吗](#q5-vace-是对-wan-的封装吗)
  - [Q6: VACE 组合任务的训练与泛化](#q6-vace-组合任务的训练与泛化)
  - [Q7: Wan-Animate vs VACE](#q7-wan-animate-vs-vace)
  - [Q8: Adapter 机制详解](#q8-adapter-机制详解)
  - [Q9: MoE 是什么](#q9-moe-是什么)
  - [Q10: Attention 机制全景分析](#q10-attention-机制全景分析)
  - [Q11: VACE 的 DiT 层结构与 Context Block 注入位置](#q11-vace-的-dit-层结构与-context-block-注入位置)
  - [Q12: 如何理解 hints 与 VACE 不需要从零训练的原因](#q12-如何理解-hints-与-vace-不需要从零训练的原因)
  - [Q13: Adapter 范式全面对比与必读论文](#q13-adapter-范式全面对比与必读论文)
- [七、面试高频问题](#七面试高频问题)

---

## 一、总览：Wan 系列全景图

Wan 是阿里巴巴通义实验室开源的大规模视频生成模型系列，其核心关系如下：

```
Wan 2.1 基座模型（T2V / I2V / T2I）
    │
    ├── VACE ─── 统一视频编辑框架（在 Wan-T2V-14B 上加 Context Adapter）
    │             支持：inpainting / outpainting / 参考生成 / 可控生成 / 视频扩展 / 组合任务
    │
    ├── Wan-Animate ─── 专用角色动画系统（在 Wan-I2V-14B 上加 Face Adapter + Body Adapter）
    │                     支持：角色动画 / 角色替换 / 表情驱动 / 重打光
    │
    └── Wan 2.2 升级版
          ├── 双专家 DiT（高噪声/低噪声各一个14B模型）
          ├── Wan 2.2 VAE（压缩比 4×16×16，z_dim=48）
          └── TI2V 5B 高效管线（支持 4090 消费级显卡 720P@24fps）
```

**核心架构组件**：
- **Wan-VAE**：3D因果变分自编码器，将视频从像素空间压缩到低维latent空间
- **Wan-DiT**：基于 Flow Matching 的视频扩散 Transformer，在latent空间进行去噪生成
- **文本编码器**：umT5-XXL（多语言T5），输出512个4096维的text token
- **图像编码器**：CLIP ViT-H/14（仅I2V使用），输出257个1280维的visual token

---

## 二、Wan 2.1 基座模型

### 2.1 背景与动机

自2024年2月 OpenAI 发布 Sora 以来，视频生成领域快速发展。但开源模型存在三大核心问题：

1. **性能不足**：开源模型与闭源商业模型之间存在显著性能差距
2. **能力有限**：多数模型仅支持 T2V，但实际创作需求是多样的（I2V、编辑、个性化等）
3. **效率不够**：14B级别模型推理成本高，普通用户无法使用

Wan 的目标是同时解决这三个问题：
- **性能领先**：14B模型在 VBench 上得分 86.22%，超过 Sora（84.28%）
- **能力全面**：T2V、I2V、T2I、视频编辑、个性化、相机控制、实时生成、音频生成
- **效率亲民**：1.3B模型仅需 **8.19 GB 显存**，可在消费级GPU上运行
- **开源**：首个能生成**中英双语视觉文本**的模型

---

### 2.2 Wan-VAE：时空变分自编码器

#### 2.2.1 设计思路与功能

VAE（Variational Autoencoder）的作用是将高维的视频像素空间压缩到低维的 latent 空间，使得后续的扩散 Transformer 不需要在巨大的像素空间上操作，从而大幅降低计算量。

**类比理解**：把一个 1080P 视频（几百万像素/帧 × 几十帧）压缩成一个小得多的"特征摘要"（latent），就像把一篇长文章压缩成关键词摘要。DiT 在这个"摘要空间"里做生成工作，最后再用 VAE Decoder 将摘要还原成完整的视频。

#### 2.2.2 架构详解

**压缩比**：时间 4× / 空间 8×8 → 总共 4 × 8 × 8 = 256 倍

给定输入视频 $V \in \mathbb{R}^{(1+T) \times H \times W \times 3}$，Wan-VAE 将其压缩到：

$$z \in \mathbb{R}^{(1+T/4) \times H/8 \times W/8 \times 16}$$

其中 16 是 latent 通道数（`z_dim=16`）。

**为什么是 1+T 而不是 T**？第一帧被特殊处理——只做空间压缩、不做时间压缩（遵循 MagViT-v2 的设计）。这是因为第一帧常作为"锚点帧"（如 I2V 的输入图像），需要保留完整的时间维度信息。

**具体数值示例**：
- 输入视频：81帧，720×1280 → shape `[3, 81, 720, 1280]`
- Encoder 输出：`[16, 21, 90, 160]`
  - 时间：(81-1)/4 + 1 = 21
  - 高度：720/8 = 90
  - 宽度：1280/8 = 160

**架构总览**（参见源码 `Wan2.1/wan/modules/vae.py`）：

```
Encoder3d:
  conv1: CausalConv3d(3 → 96)           # RGB → 基础特征
  ─── Stage 0 ───
    2 × ResidualBlock(96 → 192)          # 不做时间下采样
    Resample(downsample2d)               # 空间 /2
  ─── Stage 1 ───
    2 × ResidualBlock(192 → 384)
    Resample(downsample3d)               # 空间 /2, 时间 /2
  ─── Stage 2 ───
    2 × ResidualBlock(384 → 384)
    Resample(downsample3d)               # 空间 /2, 时间 /2
  ─── Stage 3 ───
    2 × ResidualBlock(384 → 384)         # 最后一级不再下采样
  middle: Res + SpatialAttn + Res
  head: RMSNorm → SiLU → CausalConv3d(384 → 32)  # 32 = z_dim*2，输出 mu 和 log_var

Decoder3d（镜像结构）:
  conv1: CausalConv3d(16 → 384)
  middle: Res + SpatialAttn + Res
  ─── 逐级上采样 ───
    多个 ResidualBlock + Resample(upsample2d/3d)
  head: RMSNorm → SiLU → CausalConv3d → RGB
```

**关键配置**（源码 `vae.py` 第958行）：
```python
cfg = dict(
    dim=96,                                    # 基础通道数
    z_dim=16,                                  # latent 通道数（encoder 输出 32 = 16*2）
    dim_mult=[1, 2, 4, 4],                    # 通道倍率：96, 192, 384, 384
    num_res_blocks=2,                          # 每级残差块数
    attn_scales=[],                            # 不在 encoder/decoder 中使用额外 attention
    temperal_downsample=[False, True, True],   # 第0级不做时间下采样
    dropout=0.0
)
```

**模型大小**：仅 **127M 参数**，非常紧凑。

#### 2.2.3 关键组件：3D Causal Convolution

**什么是 3D 卷积？**

普通 2D 卷积是在图像的 H×W 平面上滑动一个 kH×kW 的核。3D 卷积则是在视频的 T×H×W 体积上滑动一个 kT×kH×kW 的核。

```
2D 卷积（图像）：核 [C_out, C_in, kH, kW] 在 [H, W] 上滑动
3D 卷积（视频）：核 [C_out, C_in, kT, kH, kW] 在 [T, H, W] 上滑动
```

**什么是"因果"（Causal）？**

普通 3D 卷积的时间维度 padding 是两侧对称的，这意味着卷积核可以"看到"未来帧。但在视频生成中，我们需要保证生成是从过去到未来的顺序——第 t 帧的特征只能依赖 ≤t 帧的信息，不能偷看未来。

**因果卷积的实现**（源码 `vae.py` 第17-73行）：

```python
class CausalConv3d(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 核心：把时间维的 padding 全部放到左侧（过去），右侧（未来）为 0
        self._padding = (
            self.padding[2], self.padding[2],   # W: 左右对称
            self.padding[1], self.padding[1],   # H: 左右对称
            2 * self.padding[0], 0              # T: 只在左侧 pad（因果！）
        )
        self.padding = (0, 0, 0)  # 禁用 Conv3d 内置 padding

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            # 将缓存的历史帧拼接到当前 chunk 的时间维前面
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]  # 减少零填充量
        x = F.pad(x, padding)
        return super().forward(x)
```

**直观理解**：想象一个时间核大小 kT=3 的卷积。正常 padding 是左右各1，这样每个时间位置能看到 [t-1, t, t+1]。因果版本把 padding 改成左2右0，每个时间位置只能看到 [t-2, t-1, t]——严格因果。

#### 2.2.4 Feature Cache 机制详解

**为什么需要 Cache？**

一个81帧720P视频直接塞进3D卷积网络会爆显存。解决方案是**把视频按时间切成小chunk，逐个处理**。但问题是：因果卷积需要看到前面的帧！如果简单切片，chunk 边界处的卷积就丢失了"上下文"。

**Feature Cache 的核心思想**：在处理每个 chunk 时，**缓存该 chunk 末尾的若干帧特征**，作为下一个 chunk 处理时的"过去上下文"。

**编码时的分块策略**：
```
输入视频 81 帧 → 分成 chunks：[1帧] [4帧] [4帧] ... [4帧]
                                ↓      ↓      ↓           ↓
Encoder 输出 latent：           1步     1步    1步         1步
                               ← 共 (81-1)/4+1 = 21 个 latent 时间步 →
```

**源码实现**（`vae.py` 第771-827行 `WanVAE_.encode`）：

```python
def encode(self, x, scale):
    self.clear_cache()  # 初始化所有 CausalConv3d 的 cache 槽位
    t = x.shape[2]
    iter_ = 1 + (t - 1) // 4  # latent 时间步数

    for i in range(iter_):
        self._enc_conv_idx = [0]  # 重置 conv 层计数器
        if i == 0:
            out = self.encoder(x[:, :, :1, :, :],  # 第一个 chunk：仅 1 帧
                               feat_cache=self._enc_feat_map,
                               feat_idx=self._enc_conv_idx)
        else:
            out_ = self.encoder(x[:, :, 1+4*(i-1):1+4*i, :, :],  # 后续 chunk：4 帧
                                feat_cache=self._enc_feat_map,
                                feat_idx=self._enc_conv_idx)
            out = torch.cat([out, out_], 2)  # 在时间维拼接

    mu, log_var = self.conv1(out).chunk(2, dim=1)
    mu = (mu - scale[0]) * scale[1]  # latent 标准化
    return mu
```

**每个 CausalConv3d 层的 cache 工作流程**（`ResidualBlock.forward`）：

```python
# feat_cache: List[Tensor]，长度 = 模型中 CausalConv3d 层总数
# feat_idx: [int]，当前处理到第几个 conv 层

if isinstance(layer, CausalConv3d) and feat_cache is not None:
    idx = feat_idx[0]
    cache_x = x[:, :, -CACHE_T:, :, :].clone()  # 保存当前 chunk 最后 2 帧
    x = layer(x, feat_cache[idx])                 # 用上一 chunk 的 cache 作为"过去"
    feat_cache[idx] = cache_x                      # 更新 cache
    feat_idx[0] += 1                               # 移到下一个 conv 层的 cache 槽
```

**两种设置的区别**：
- **Default setting（无时间下采样层）**：CausalConv3d 不改变帧数，缓存最后 2 帧（`CACHE_T=2`），初始 chunk 用零填充
- **2× temporal downsampling**：stride=(2,1,1) 的 CausalConv3d 将帧数减半，只缓存最后 1 帧

**解码时的分块策略**：按 latent 时间维**逐帧**解码：

```python
def decode(self, z, scale):
    self.clear_cache()
    z = z / scale[1] + scale[0]  # 反标准化
    x = self.conv2(z)
    for i in range(z.shape[2]):  # 逐时间步
        self._conv_idx = [0]
        if i == 0:
            out = self.decoder(x[:, :, i:i+1, :, :], feat_cache=self._feat_map, ...)
        else:
            out_ = self.decoder(x[:, :, i:i+1, :, :], feat_cache=self._feat_map, ...)
            out = torch.cat([out, out_], 2)
    return out
```

#### 2.2.5 VAE 中的 Attention Block

VAE 的 middle 层包含一个**逐帧空间自注意力**（不是时空联合注意力）：

```python
class AttentionBlock(nn.Module):
    def forward(self, x):  # x: [B, C, T, H, W]
        b, c, t, h, w = x.size()
        x = rearrange(x, 'b c t h w -> (b t) c h w')  # 将时间并入 batch
        q, k, v = self.to_qkv(x).reshape(b*t, 1, c*3, -1).permute(0,1,3,2).chunk(3, dim=-1)
        # q, k, v: [(B*T), 1, H*W, C]  —— 每帧独立做 H*W 个 token 的注意力
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)
        return x + identity  # 残差连接
```

**注意**：VAE 中的注意力是**每帧独立**的空间注意力（token 数 = H×W），不是时空联合注意力。时间维度的信息传递完全由 3D Causal Conv 负责。

#### 2.2.6 VAE 训练

三阶段训练：
1. **Stage 1**：先训练一个 2D 图像 VAE（同结构但无时间维）
2. **Stage 2**：将 2D VAE "膨胀"为 3D Causal VAE，在低分辨率（128×128）短视频（5帧）上训练
   - 损失函数：L1 重建损失（系数3）+ KL 散度损失（系数3e-6）+ LPIPS 感知损失（系数3）
3. **Stage 3**：在高质量视频上微调，加入 3D 判别器的 GAN 损失

#### 2.2.7 Latent 标准化

编码后的 latent 需要做**逐通道标准化**，使不同通道的数值尺度一致：

```python
# 16 个通道各有统计好的 mean 和 std
mean = [-0.7571, -0.7089, -0.9113, 0.1075, ...]  # 16 个值
std  = [2.8184, 1.4541, 2.3275, 2.6558, ...]       # 16 个值

# 编码后标准化：z_norm = (mu - mean) / std
# 解码前反标准化：z = z_norm * std + mean
```

---

### 2.3 Wan-DiT：视频扩散Transformer

#### 2.3.1 设计思路

Wan-DiT 是在 latent 空间上工作的**扩散 Transformer**。它的任务是：给定一个带噪声的 latent $x_t$、文本条件 $c_{txt}$、和时间步 $t$，预测去噪所需的**速度场** $v$（Flow Matching 框架）。

#### 2.3.2 整体架构

```
输入 latent z: [B, 16, T_lat, H_lat, W_lat]
    │
    ▼
Patchify (Conv3d, kernel=stride=(1,2,2))
    → [B, dim, T_lat, H_lat/2, W_lat/2]
    → flatten → [B, L, dim]    （L = T_lat × H_lat/2 × W_lat/2）
    │
    ▼
N × Transformer Block:
    ┌─ AdaLN(x, timestep_emb) → Self-Attention (3D RoPE) → gate → residual
    ├─ Cross-Attention (q=x, kv=text_tokens) → residual
    └─ AdaLN(x, timestep_emb) → FFN → gate → residual
    │
    ▼
Unpatchify → [B, 16, T_lat, H_lat, W_lat]   （预测的速度场 v）
```

**默认配置**（14B 模型）：

| 参数 | 值 | 说明 |
|------|-----|------|
| `patch_size` | (1, 2, 2) | 时间不 patch，空间 2×2 |
| `dim` | 5120 | 隐藏维度 |
| `ffn_dim` | 13824 | FFN 中间维度 |
| `num_heads` | 40 | 注意力头数 |
| `num_layers` | 40 | Transformer 层数 |
| `text_len` | 512 | 文本 token 最大长度 |
| `in_dim` | 16 | 输入 latent 通道数（T2V） |
| `out_dim` | 16 | 输出通道数 |
| `freq_dim` | 256 | 时间步嵌入维度 |

#### 2.3.3 Patchify：视频 latent → token 序列

**这是理解视频DiT最关键的步骤**。

视频 latent 是 4 维的 `[C, T, H, W]`。如何变成 Transformer 能处理的 1 维 token 序列？

答案：用一个 **3D 卷积**当作"打补丁"操作，然后**展平**。

```python
# 源码 model.py
self.patch_embedding = nn.Conv3d(
    in_dim, dim,
    kernel_size=patch_size,   # (1, 2, 2)
    stride=patch_size         # (1, 2, 2) — 不重叠
)

# 前向过程：
# 输入 x: [1, 16, 21, 90, 160]  （以81帧720P为例）
x = self.patch_embedding(x)
# → [1, 5120, 21, 45, 80]
# 因为 kernel=(1,2,2), stride=(1,2,2)：T不变，H/2, W/2

x = x.flatten(2).transpose(1, 2)
# flatten(2): [1, 5120, 21*45*80] = [1, 5120, 75600]
# transpose: [1, 75600, 5120]  —— 这就是 token 序列！
```

**直观理解**：
- 每个 token 对应 latent 空间中的一个 `1×2×2` 的小块（1个时间步，2×2空间区域）
- 整个视频 latent 被"拍平"成一条长序列，就像把一本立体书的每一页都撕下来排成一行
- 序列长度 L = T_lat × (H_lat/2) × (W_lat/2) = 21 × 45 × 80 = **75,600 tokens**

**为什么 `patch_size=(1,2,2)` 而不是 `(2,2,2)`？**
- 时间维不再 patch，因为 VAE 已经在时间上做了 4× 压缩，进一步压缩会丢失时间细节
- 空间维做 2×2 patch，将空间 token 数减少 4 倍，平衡了序列长度和计算量

#### 2.3.4 3D RoPE：三维旋转位置编码

普通 Transformer 使用 1D 位置编码（token 在序列中的位置）。但视频 token 来自 3D 网格 (F, H, W)，需要 **3D 位置编码**。

Wan 使用 **3D RoPE（Rotary Position Embedding）**：将每个注意力头的 head_dim 拆成三段，分别编码时间、高度、宽度位置。

```python
# 源码 model.py 第78-139行
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2  # n=heads, c=head_dim/2

    # 将 c 个频率维度拆成三段：时间 / 高度 / 宽度
    # 时间分到 c - 2*(c//3) 个维度（约 1/3，但稍多）
    # 高度和宽度各分到 c//3 个维度
    freqs = freqs.split([c - 2*(c//3), c//3, c//3], dim=1)

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].reshape(seq_len, n, -1, 2))

        # 为每个 token 构造 3D 旋转因子
        freqs_i = torch.cat([
            freqs[0][:f].view(f,1,1,-1).expand(f,h,w,-1),  # 时间位置
            freqs[1][:h].view(1,h,1,-1).expand(f,h,w,-1),  # 高度位置
            freqs[2][:w].view(1,1,w,-1).expand(f,h,w,-1),  # 宽度位置
        ], dim=-1).reshape(seq_len, 1, -1)

        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)  # 复数乘法 = 旋转
```

**直观理解**：每个 token 带有三个"坐标"——它在第几帧、第几行、第几列。3D RoPE 让注意力能感知到 token 之间的**相对时空距离**。

#### 2.3.5 AdaLN-Zero 时间步调制

每个 Transformer Block 接收时间步 embedding，产生 6 个调制参数：

```python
# 时间步 → sinusoidal embedding → MLP → 投影到 6*dim
self.time_embedding = nn.Sequential(
    nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
self.time_projection = nn.Linear(dim, 6 * dim)

# 每个 block 的可学习调制偏差
self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

# 前向：
e = (self.modulation + e0).chunk(6, dim=1)  # 6 个 [B, 1, dim]
# e[0], e[1]: self-attn 输入的 shift/scale
# e[2]: self-attn 输出的 gate
# e[3], e[4]: FFN 输入的 shift/scale
# e[5]: FFN 输出的 gate

y = self.self_attn(self.norm1(x) * (1 + e[1]) + e[0], ...)
x = x + y * e[2]  # gated residual
```

**关键设计**：`time_embedding` MLP 在所有 block 间**共享**，每个 block 只学习一组 `modulation` 偏差。这比每个 block 独立 MLP **减少约 25% 参数**，且论文实验表明性能更优。

#### 2.3.6 文本编码器：umT5-XXL

Wan 选择 **umT5**（统一多语言 T5）作为文本编码器，而非常见的 CLIP Text 或 LLM。原因：

1. **多语言能力强**：原生支持中英文，能理解视觉文字指令
2. **双向注意力**：T5 Encoder 使用双向注意力，比仅用因果注意力的 LLM 在语义理解上更优
3. **收敛更快**：相同参数量下比 Qwen2.5-7B 等 LLM 收敛更快

```python
# T5 配置（源码 t5.py）
vocab_size=256384, dim=4096, dim_attn=4096, dim_ffn=10240
num_heads=64, encoder_layers=24
# 输出：最长 512 个 token，每个 4096 维
```

---

### 2.4 Flow Matching 数学原理

**Wan 使用 Flow Matching 而非传统 DDPM 进行训练和采样。** 这是面试高频考点。

#### 2.4.1 核心区别

| 对比项 | DDPM（扩散模型） | Flow Matching |
|--------|------------------|---------------|
| 前向过程 | 逐步加高斯噪声 | 直线插值 |
| 模型预测目标 | 噪声 $\epsilon$ | 速度场 $v$（从噪声到数据的方向） |
| 采样过程 | 反向 SDE/ODE | 求解 ODE |
| 数学框架 | 随机微分方程 | 常微分方程（更简洁） |

#### 2.4.2 数学公式详解

**（1）前向插值（构造训练样本）**：

$$x_t = t \cdot x_1 + (1 - t) \cdot x_0 \tag{Eq.1}$$

- $x_1$：干净的视频 latent（训练数据经 VAE 编码得到）
- $x_0 \sim \mathcal{N}(0, I)$：标准高斯噪声
- $t \in [0, 1]$：时间步，从 logit-normal 分布采样
- $x_t$：中间状态。当 $t=0$ 时 $x_t = x_0$ 是纯噪声；当 $t=1$ 时 $x_t = x_1$ 是干净数据

**理解**：这是一条从噪声 $x_0$ 到数据 $x_1$ 的**直线路径**。$t$ 控制在路径上的位置。

**（2）真实速度场**：

$$v_t = \frac{dx_t}{dt} = x_1 - x_0 \tag{Eq.2}$$

- $v_t$：真实速度场，就是从噪声到数据的方向向量
- 注意它**不依赖于 $t$**——沿直线的速度是常数

**（3）训练损失**：

$$\mathcal{L} = \mathbb{E}_{x_0, x_1, c_{txt}, t} \left\| u(x_t, c_{txt}, t; \theta) - v_t \right\|^2 \tag{Eq.3}$$

- $u(x_t, c_{txt}, t; \theta)$：模型（DiT）预测的速度场
- $v_t = x_1 - x_0$：真实速度场
- $c_{txt}$：umT5 文本嵌入（512 tokens × 4096 维）
- $\theta$：模型参数
- 目标：让模型预测的速度尽量接近真实速度

#### 2.4.3 采样过程（推理时）

从纯噪声开始，沿模型预测的速度场"走"到数据：

```python
# 核心关系：x_0 = x_t - sigma_t * v(x_t, t)
# 其中 sigma_t 对应时间步 t

# 源码 fm_solvers.py
def convert_model_output(self, model_output, sample):
    sigma_t = self.sigmas[self.step_index]
    x0_pred = sample - sigma_t * model_output  # 数据预测
```

**Sigma 调度与 Shift**：

```python
sigma = np.linspace(1, 0, sampling_steps + 1)[:sampling_steps]
sigma = shift * sigma / (1 + (shift - 1) * sigma)
```

- `shift` 参数控制噪声调度的形状。`shift > 1` 时，高噪声阶段被压缩，更多采样步数分配给低噪声的精细化阶段

**DPM-Solver++ 多步求解器**：Wan 使用 1~3 阶的 DPM-Solver++ 加速采样（默认50步），比简单的 Euler 方法效率更高。

---

### 2.5 T2V 文本生成视频

#### 2.5.1 完整推理流程

```
用户 Prompt
    │
    ▼
Prompt Alignment（LLM 改写，丰富细节）
    │
    ▼
umT5 编码 → text_tokens: [512, 4096]
    │
    ▼
采样高斯噪声 z_noise: [16, T_lat, H_lat, W_lat]
    │
    ▼
For 每个时间步 t (从 sigma=1 到 0，约 50 步):
    │
    ├─ sinusoidal(t) → time_embedding MLP → e: [dim]
    ├─ time_projection(e) → e0: [6*dim] （AdaLN 调制参数）
    │
    ├─ patch_embedding(z): Conv3d → flatten → [L, dim]
    │
    ├─ N 个 Transformer Block:
    │     ├─ AdaLN + Self-Attn (3D RoPE) + gate residual
    │     ├─ Cross-Attn (q=x, kv=text_tokens)
    │     └─ AdaLN + FFN + gate residual
    │
    ├─ Head → unpatchify → v_pred: [16, T_lat, H_lat, W_lat]
    │
    └─ scheduler.step(v_pred, t, z) → z_updated
    │
    ▼
VAE Decode（分块）→ 视频: [3, T, H, W]
```

#### 2.5.2 Classifier-Free Guidance (CFG)

```python
# 同时做有条件和无条件预测
noise_cond = model(latent, t, context=positive_prompt)
noise_uncond = model(latent, t, context=negative_prompt)
# CFG 公式
noise_pred = noise_uncond + guide_scale * (noise_cond - noise_uncond)
```

`guide_scale` 越大，越忠实于 prompt，但可能损失多样性。默认值约 5.0。

---

### 2.6 I2V 图像生成视频

#### 2.6.1 与 T2V 的核心区别

I2V 需要额外接收一张**参考图像**作为输入。Wan 的做法是通过**通道拼接**将图像信息注入：

```python
# 构造"伪视频"：第一帧 = 输入图像，其余帧 = 0
y = self.vae.encode([torch.concat([
    img.transpose(0, 1),          # [3, 1, H, W] 输入图像
    torch.zeros(3, F-1, H, W)     # [3, F-1, H, W] 零填充
], dim=1)])
# → y: [16, T_lat, H_lat, W_lat]

# 构造 mask：第一帧 = 1（已知），其余帧 = 0（待生成）
msk = torch.ones(1, F, lat_h, lat_w)
msk[:, 1:] = 0
# 重组后 → msk: [4, T_lat, H_lat, W_lat]

# 通道拼接：mask(4ch) + image_latent(16ch) = 20ch
y = torch.concat([msk, y])  # [20, T_lat, H_lat, W_lat]

# 在模型内部，与噪声 latent(16ch) 拼接 → 总共 36ch 输入 patch_embedding
```

**I2V 模型的 `in_dim = 36`**（而 T2V 是 16）。

#### 2.6.2 Dual Cross-Attention（图像+文本双路交叉注意力）

I2V 还通过 **CLIP 图像编码器** 提取图像全局特征，注入到 Cross-Attention 中：

```python
# CLIP 编码输入图像 → 257 tokens × 1280 维
clip_features = self.clip.visual(img)  # [257, 1280]
clip_features = self.mlp_proj(clip_features)  # [257, dim]

# 与 text tokens 拼接
context = concat([clip_features, text_tokens])  # [257+512, dim]

# Dual Cross-Attention:
class WanI2VCrossAttention(WanSelfAttention):
    def forward(self, x, context, context_lens):
        context_img = context[:, :257]   # CLIP 图像 token
        context_txt = context[:, 257:]   # 文本 token

        q = self.q(x)
        # 文本路径
        k_txt, v_txt = self.k(context_txt), self.v(context_txt)
        out_txt = flash_attention(q, k_txt, v_txt, k_lens=context_lens)
        # 图像路径（独立的 K/V 投影）
        k_img, v_img = self.k_img(context_img), self.v_img(context_img)
        out_img = flash_attention(q, k_img, v_img, k_lens=None)

        return self.o(out_txt + out_img)  # 简单相加
```

---

### 2.7 视频个性化（Video Personalization）

**目标**：给定用户提供的参考人脸，生成保持该人身份一致性的视频。

**核心设计决策**：
- **不使用任何外部特征提取器**（如 ArcFace、InsightFace）——避免信息损失
- **直接在 VAE latent 空间中以人脸图像为条件**
- 使用**自注意力范式**（而非交叉注意力）

**实现方式**：
1. 将人脸图像检测、分割、对齐到标准画布上
2. 在视频前扩展 K 帧，用分割好的人脸图像填充
3. 通道维度拼接：人脸图像 + 全 1 mask（前 K 帧）| 空白图像 + 全 0 mask（后续帧）
4. 以 **inpainting 模式** 做扩散

**训练数据**：
- 从 O(100)M 视频中筛选出 O(10)M 包含单一人脸的视频
- 每个视频平均分割出约 5 张人脸
- 额外用 Instant-ID 合成 O(1)M 带合成人脸的视频（增加多样性）

---

### 2.8 实时视频生成（Real-Time Generation）

**核心创新**：将固定长度的生成改为**流式滑动窗口去噪**。

**Streamer 架构**：
- 维护一个**去噪队列**，token 在时间维度上以滑动窗口方式处理
- 每完成一轮去噪后：最左侧的 token（噪声最低）出队并缓存；右侧追加新的高斯噪声 token
- 无预定义长度限制——支持**无限长视频**生成

**一致性模型蒸馏**：
- 将原始扩散过程蒸馏为 **4 步一致性模型**（LCM/VideoLCM）
- 推理加速 10-20 倍
- 达到 8-16 FPS 渲染速率

**量化优化**：
- Int8 + TensorRT 量化
- 单张 RTX 4090 可达 **实时 20 FPS**

---

### 2.9 相机运动可控性（Camera Motion）

**两个核心组件**：

1. **Camera Pose Encoder**：
   - 输入：每帧的相机外参 $[R, t] \in \mathbb{R}^{3 \times 4}$ 和内参 $K_f \in \mathbb{R}^{3 \times 3}$
   - 通过 **Plücker 坐标**将像素坐标转换为精细位置 $P \in \mathbb{R}^{6 \times F \times H \times W}$
   - PixelUnshuffle 降空间分辨率 + 多级卷积提取特征

2. **Camera Pose Adapter**：
   - 将相机运动特征转换为缩放因子 $\gamma_i$ 和偏移参数 $\beta_i$
   - 注入方式：$f_i = (\gamma_i + 1) \cdot f_{i-1} + \beta_i$

**训练数据**：使用 VGGSfM 算法从带明显相机运动的训练视频中提取相机轨迹，约 O(1)K 视频片段。

---

### 2.10 训练过程与数据集

#### 2.10.1 数据清洗流程（四步）

| 步骤 | 内容 | 淘汰率 |
|------|------|--------|
| Step 1: 基础维度 | OCR检测、美学评分、NSFW过滤、水印检测、黑边裁剪、过曝/模糊/合成图检测 | ~50% |
| Step 2: 视觉质量 | 100 clusters + 1-5分专家评分 | 进一步筛选 |
| Step 3: 运动质量 | 6级运动质量分类（最优→低质量→抖动，后两类排除） | 进一步筛选 |
| Step 4: 视觉文本 | 合成数百M含文字图像 + 真实含文字图像（OCR + Qwen2-VL 标注） | 补充数据 |

#### 2.10.2 训练阶段

```
Stage 1: 图像预训练（256px T2I）
    → 建立跨模态语义对齐
    │
Stage 2: 图像-视频联合训练
    ├─ 阶段 1: 256px 图像 + 192px/5s 视频
    ├─ 阶段 2: 480px 图像 + 480px/5s 视频
    └─ 阶段 3: 720px 图像 + 720px/5s 视频
    │
Stage 3: 后训练（Post-training）
    → 高质量数据微调，480px + 720px 联合
```

**优化器**：AdamW，bf16 混合精度，初始学习率 1e-4，遇 FID/CLIP Score 平台期动态降低。

#### 2.10.3 并行训练策略

- **VAE + 文本编码器**：数据并行（DP）
- **DiT**：FSDP + 2D Context Parallelism（Ring Attention 外层 + Ulysses 内层）
- 128 GPU 示例：CP=16（Ulysses=8, Ring=2），FSDP=32，DP=4，全局 batch=8

---

### 2.11 创新贡献与不足

#### 贡献

1. **新型时空 VAE（Wan-VAE）**：3D因果VAE，4×8×8压缩比，Feature Cache支持无限长视频，127M参数，比SOTA快2.5倍
2. **高效DiT架构**：共享adaLN MLP减少25%参数且性能更优
3. **首个中英双语视觉文本生成**
4. **全面的能力栈**：T2V/I2V/T2I/编辑/个性化/相机控制/实时/音频
5. **消费级效率**：1.3B模型仅需8.19GB显存
6. **完善的数据管线与评估体系**（Wan-Bench）

#### 不足

1. **大运动场景下细节保持困难**
2. **14B模型推理慢**：单GPU约30分钟
3. **缺乏领域专业性**：教育、医疗等垂直领域表现不足
4. **音频不支持人声**（训练时排除了语音数据）

---

## 三、VACE：统一视频创作与编辑框架

### 3.1 背景与动机

**问题**：视频下游任务（重绘、编辑、可控生成、参考生成等）日益多样化，但每个任务都需要**独立的专用模型**，导致：
- 部署开销大（多模型并行）
- 任务之间无法组合（如"参考人物 + 姿态控制"需要两个独立模型）
- 视频的时空一致性更难维护

**VACE 的核心思想**：设计一个**统一的输入表示**（VCU），使得所有视频生成/编辑任务都能用同一个模型、同一个接口来完成。

**论文定位**：VACE 是首个基于视频 DiT 架构的 all-in-one 视频创编模型。

---

### 3.2 VCU：视频控制单元

**VCU（Video Condition Unit）** 是 VACE 最核心的概念创新。

$$V = [T; F; M] \tag{Eq.4}$$

- $T$：文本 prompt
- $F = \{u_1, u_2, \ldots, u_n\}$：上下文视频帧序列
- $M = \{m_1, m_2, \ldots, m_n\}$：二值 mask 序列

**约定**：
- $u$ 在 RGB 空间，归一化到 $[-1, 1]$
- $m$ 为二值：$1$ = 该像素需要生成/编辑，$0$ = 该像素保持不变
- $F$ 和 $M$ 在空间 $h \times w$ 和时间 $n$ 上对齐

**万物皆 VCU**：所有任务都只是 $(T, F, M)$ 的不同配置！

---

### 3.3 四大基础任务分类

| 任务 | F（帧） | M（mask） | 含义 |
|------|---------|-----------|------|
| **T2V** | 全 0 帧 × n | 全 1 mask × n | 纯文本生成，所有像素都要生成 |
| **R2V** | 参考图前置 + 全 0 帧 | 参考帧对应 mask=0 + 生成帧 mask=1 | 参考图像引导生成 |
| **V2V** | 源视频帧 / 控制信号帧 | 全 1 mask × n | 视频到视频转换（深度/姿态/灰度等控制） |
| **MV2V** | 源视频帧 | 逐帧空间 mask（部分区域 1，部分 0） | 局部编辑（inpainting/outpainting/时间扩展） |

**具体示例**：
- **Inpainting**（擦除重绘）：F = 源视频帧，M = 被擦除区域为 1，保留区域为 0
- **Outpainting**（画布扩展）：F = 源视频帧（居中），M = 新扩展区域为 1
- **深度控制生成**：F = 深度图序列，M = 全 1（整个视频重新生成，但受深度图引导）
- **人脸参考生成**：F = [参考人脸, 0, 0, ..., 0]，M = [0, 1, 1, ..., 1]

---

### 3.4 Context Adapter 架构详解

#### 3.4.1 Concept Decoupling（概念解耦）

在编码 VCU 之前，先将帧分成两类：

$$F_c = F \odot M \quad \text{（reactive 帧——将被修改的像素）}$$
$$F_k = F \odot (1 - M) \quad \text{（inactive 帧——将被保留的像素）}$$

**为什么要解耦？** 因为 $F$ 中同时包含两种语义完全不同的内容：
- 控制信号（如深度图、参考人物）
- 需要保留的原始内容（如背景）

不解耦会导致模型混淆，训练难以收敛。论文消融实验证实解耦后 loss 下降更快（Figure 5d）。

#### 3.4.2 Context Latent Encoding

将 $F_c$、$F_k$、$M$ 都编码到与 $X$（噪声latent）相同的空间：
- $F_c$、$F_k$ 通过 **Wan-VAE** 编码，得到同维 latent
- **参考图像特别处理**：单独编码后在时间维拼接（避免图像和视频混合编码产生伪影）
- $M$ 直接 resize/interpolate 到 latent shape

最终 $F_c$、$F_k$、$M$ 都与 $X$ 在 $n' \times h' \times w'$ 上对齐。

#### 3.4.3 Context Embedder

将 $F_c$、$F_k$、$M$ 在**通道维度拼接**，然后通过一个扩展的 patchify 层 token 化：

```python
# 源码 Wan2.1/wan/modules/vace_model.py
# 输入通道 = F_c(16ch) + F_k(16ch) + M(对应通道数) = 总共更多通道
# 权重初始化：F_c 和 F_k 对应权重从原始 patch_embedding 复制
#             M 对应权重初始化为 0（初始不影响）
```

#### 3.4.4 两种训练策略

**策略(a) Full Fine-tuning**：
- Context tokens 直接加到噪声 tokens 上
- 全参数训练（DiT + Context Embedder）

**策略(b) Context Adapter Tuning（推荐）**：

```
原始 DiT（冻结）                     Context Branch（可训练）
    │                                     │
Block_0: x → self-attn → cross-attn → FFN    Context Block_0: c → transformer → c_skip
    │                    + c_skip ←──────────┘
    ▼                                     │
Block_5: x → self-attn → cross-attn → FFN    Context Block_1: c → transformer → c_skip
    │                    + c_skip ←──────────┘
    ▼                                     │
   ...                                   ...
```

**实现细节**（源码 `vace_model.py`）：

```python
class VaceWanAttentionBlock(WanAttentionBlock):
    """条件分支的 Transformer Block"""
    def forward(self, c, x, **kwargs):
        if self.block_id == 0:
            c = self.before_proj(c) + x  # 第一层：对齐条件 token 到主干空间
        c = super().forward(c, **kwargs)  # 标准 transformer
        c_skip = self.after_proj(c)        # 零初始化投影 → hint
        return c, c_skip

class BaseWanAttentionBlock(WanAttentionBlock):
    """主干的 Transformer Block"""
    def forward(self, x, hints, context_scale=1.0, **kwargs):
        x = super().forward(x, **kwargs)
        if self.block_id is not None:
            x = x + hints[self.block_id] * context_scale  # 注入 hint
        return x
```

**关键设计**：
- `before_proj` 和 `after_proj` 都是**零初始化**的，保证训练初期不影响原始模型
- Context Blocks 选择**分布式放置**（如 [0, 5, 10, 15, 20, 25, 30, 35]），而非连续放置——消融实验表明分布式效果更好
- Wan2.1-T2V-14B：40 层 DiT，8 个 Context Blocks

---

### 3.5 训练过程与数据

#### 3.5.1 VACE 不需要从头训练！

这是关键点。VACE 基于**预训练好的 T2V 基座模型**（如 Wan2.1-T2V-14B），只训练新增的部分：
- **策略(a)**：所有参数微调
- **策略(b)**：仅训练 Context Embedder + Context Blocks（原始 DiT 冻结）

#### 3.5.2 数据构建流程

1. **视频切片** → 初步过滤（分辨率、美学、运动幅度）
2. **物体检测**（Grounding DINO）→ **视频分割**（SAM2）→ 实例级 mask
3. **针对各任务构建训练对**：
   - V2V：提取深度/姿态/灰度/光流/涂鸦
   - MV2V：随机 instance mask（inpainting/outpainting）
   - R2V：提取参考人脸/物体
   - 时间扩展：首帧/末帧/双端帧

#### 3.5.3 三阶段训练

| 阶段 | 内容 |
|------|------|
| Phase 1 | 基础任务（inpainting + 时间扩展），建立 mask 理解和空间/时间上下文生成 |
| Phase 2 | 任务扩展：单参考→多参考，单任务→组合任务 |
| Phase 3 | 质量微调：更高质量数据、更长序列、任意分辨率 |

**训练中，组合任务也会被随机采样加入训练数据**——这是组合泛化能力的来源。

#### 3.5.4 训练配置

Wan-T2V-14B 版 VACE：
- 128 × A100 GPU
- 200K 训练步
- 学习率 5e-5
- 序列长度 75,600 tokens
- 采样步数 25，CFG scale 4.0
- ~260s/视频（8 GPU推理）

---

### 3.6 组合任务能力

VCU 天然支持组合——只需将不同任务的 F 和 M 合理拼接：

| 组合任务 | F 配置 | M 配置 |
|----------|--------|--------|
| **Swap Anything**（参考+重绘） | [参考图, 源视频帧...] | [0×l, m_1, m_2, ...] |
| **Animate Anything**（参考+姿态） | [参考图, 姿态帧...] | [0×l, 1×n] |
| **Move Anything**（参考+轨迹） | [参考图, 布局帧...] | [0, 1, 1, ..., 1] |
| **Expand Anything**（扩展+延伸） | 画布扩展 + 时间延伸 | 混合 mask |

---

### 3.7 创新贡献与不足

#### 贡献
1. **首个基于视频 DiT 的 all-in-one 模型**
2. **VCU 统一输入范式**
3. **Concept Decoupling**：显式分离 reactive/inactive 帧
4. **Context Adapter（Res-Tuning）**：可插拔、收敛快、不影响基座
5. **组合泛化能力**
6. **VACE-Benchmark**：12 任务 × 480 样本

#### 不足
1. **质量受限于基座模型**：小模型快但质量低，大模型质量高但慢
2. **训练规模不够**：身份保持、复杂组合任务的控制力仍有提升空间
3. **操作复杂度**：视频的时间维 + 多模态输入提高了使用门槛

---

## 四、Wan-Animate：统一角色动画与替换

### 4.1 背景与动机

**问题**：角色动画（Character Animation）领域存在三大缺口：

1. **开源方法多基于 UNet（如 SD/SVD）**：生成质量远落后于 DiT 架构
2. **已有 DiT 开源方法不完整**：身体驱动的模型缺表情、表情驱动的模型不含身体
3. **缺乏统一的身体+面部控制**：没有模型能同时精细控制身体动作和面部表情
4. **缺乏环境整合（角色替换）**：VACE 虽能近似实现，但存在身份一致性和可用性问题

**Wan-Animate 的解决方案**：
- 基于 **Wan-I2V-14B** 构建（DiT 架构，高质量基础）
- **统一两种模式**：Animation Mode（动画模式）和 Replacement Mode（替换模式）
- **解耦身体+面部控制**：Body Adapter + Face Adapter
- **Relighting LoRA**：角色替换时自动适配目标场景光照

### 4.2 模型架构

#### 4.2.1 总体结构

```
                      ┌─── CLIP ViT ─── MLPProj ─── 图像特征（257 tokens）
                      │                                    ↓
参考人物图像 ─────────┤                              Cross-Attn
                      │
                      └─── Wan-VAE ─── 参考 latent（拼入条件 latent 时间维）
                                            ↓
驱动姿态视频 ─── Wan-VAE ─── 姿态 latent ─ Proj ── ⊕ ── 噪声 latent
                                                         ↓
驱动面部视频 ─── Face Encoder ─── Motion Vec ─── Face Adapter
                   (512→20→512→5120)                     ↓
                                                  Wan-DiT (40层)
                                                  每5层注入一次 Face Adapter
                                                         ↓
                                                    VAE Decode → 生成视频
```

#### 4.2.2 Body Adapter（身体适配器）

- **信号选择**：2D 骨骼（VitPose 提取），而非 SMPL
  - 原因：2D 骨骼泛化性更好（非人形角色也能用），不含体型信息（避免影响身份一致性学习）
- **注入方式**：骨骼帧 → Wan-VAE 编码 → Projection → **直接加到噪声 latent 上**

```python
# 源码 model_animate.py
def after_patch_embedding(self, x, pose_latents, face_pixel_values):
    # 骨骼 latent 经 patchify 后直接加到 token 上
    pose_latents = [self.pose_patch_embedding(u.unsqueeze(0)) for u in pose_latents]
    for x_, pose_ in zip(x, pose_latents):
        x_[:, :, 1:] += pose_  # 注意：跳过第0帧（参考帧不加姿态）
```

#### 4.2.3 Face Adapter（面部适配器）

这是 Wan-Animate 最核心的创新组件。

**为什么不用面部关键点？**
- 关键点在提取时丢失精细表情信息
- 跨身份时，面部形状差异导致关键点需要极高精度的重定向

**设计方案**：直接用**原始面部图像**作为驱动输入。

**Face Encoder 架构**（源码 `motion_encoder.py` + `face_blocks.py`）：

```
面部图像 [B, 3, 512, 512]
    │
    ▼
Motion Encoder（StyleGAN 风格）
    ├─ Appearance Encoder: 逐级下采样 512→4→512维特征
    ├─ Motion FC: 512→20 维 motion_dim
    └─ Direction: QR 分解正交方向矩阵 × motion → 512 维
    │
    ▼
每帧一个 512 维 motion vector
    │
    ▼
FaceEncoder:
    ├─ CausalConv1d(512 → 4096, k=3)     # 展开为多头
    ├─ CausalConv1d(1024 → 1024, k=3, stride=2)  # 时间 /2
    ├─ CausalConv1d(1024 → 1024, k=3, stride=2)  # 时间 /2（总共 /4）
    ├─ Linear(1024 → 5120)                 # 投影到 DiT 维度
    └─ 拼接 learnable padding token
    │
    ▼
face_embeddings: [B, T//4, num_heads+1, 5120]
    │
    ▼
FaceBlock（每 5 层注入一次，共 8 个）:
    Q = video_features    K, V = face_embeddings
    → 时间对齐的 Cross-Attention（每帧的 token 只看对应时间步的 face embedding）
    → 残差加回主干
```

**身份-表情解耦策略**（关键设计）：
1. **空间压缩到 1D 向量**：将 512×512 的面部图像压缩为单个向量，去除身份相关的低级纹理信息
2. **训练时数据增强**：对面部图像施加缩放、颜色抖动、随机噪声，制造与目标面部的差异，防止模型过拟合身份特征

#### 4.2.4 Relighting LoRA

**问题**：在替换模式下，如果严格保持角色原始外观，光照/色调会与新场景不匹配。

**解决方案**：训练一个 LoRA 学习"将角色外观适配到新场景光照"。

**训练数据构造**：
1. 从视频片段中提取参考图像
2. 分割出角色
3. 用 **IC-Light** 将角色合成到随机背景上（光照改变）
4. 合成图像作为参考输入，原始视频作为 ground truth
5. LoRA 学习从"错误光照"还原到"正确光照"

```python
# LoRA 配置（源码 animate_utils.py）
def get_loraconfig(transformer, rank=128, alpha=128):
    target_modules = []
    for name, module in transformer.named_modules():
        if "blocks" in name and "face" not in name and "modulation" not in name \
           and isinstance(module, torch.nn.Linear):
            target_modules.append(name)
    # LoRA 应用在 DiT blocks 的所有 Linear 层（排除 face adapter 和调制层）
```

### 4.3 训练过程与数据

#### 4.3.1 五阶段渐进训练

| 阶段 | 内容 | 数据 |
|------|------|------|
| Stage 1 | 仅身体控制（动画模式） | 全身动作视频 |
| Stage 2 | 加入面部控制（Face Adapter） | 肖像数据（面部运动主导） |
| Stage 3 | 身体+面部联合训练 | 全数据集 |
| Stage 4 | 加入替换模式 | 双模式数据 |
| Stage 5 | Relighting LoRA | IC-Light 增强数据对 |

**为什么要分阶段？** 消融实验表明，直接联合训练所有组件会导致不收敛。渐进式训练让每个组件先独立学好，再联合微调。

#### 4.3.2 训练数据

- 大规模人体视频（说话、表情、身体运动）
- 每个视频确保**单一角色**
- 骨骼提取（VitPose）+ 角色分割（SAM2）
- 文本描述由 QwenVL2.5-72B 生成

### 4.4 创新贡献与不足

#### 贡献
1. **首个开源统一动画+替换框架**
2. **解耦身体+面部控制**（Body Adapter + Face Adapter）
3. **隐式面部特征替代关键点**（保留精细表情）
4. **Relighting LoRA**（场景光照适配）
5. **五阶段渐进训练**
6. SOTA 性能（人类评估 win rate vs Runway Act-two: 67.2%）

#### 不足
1. **2D 骨骼的姿态重定向局限**：身体比例差异大时不准确
2. **替换模式不支持姿态重定向**（可能破坏环境交互）
3. **文本控制是辅助性的**（运动信号主导）
4. **仅支持单角色**

---

## 五、Wan 2.2 升级解析

### 5.1 双专家模型策略（"MoE"）

官方宣传 Wan 2.2 引入了 "Mixture-of-Experts (MoE)"。但从源码分析，**实际实现并非传统意义上的 MoE（稀疏门控专家路由）**，而是一种**双模型时间步切分策略**。

#### 5.1.1 源码实现

```python
# 源码 Wan2.2/wan/text2video.py
class WanT2V:
    def __init__(self, config, ...):
        self.low_noise_model = self.load_model(config.low_noise_checkpoint)
        self.high_noise_model = self.load_model(config.high_noise_checkpoint)

    def _prepare_model_for_timestep(self, t, boundary, ...):
        if t.item() >= boundary:
            required_model_name = 'high_noise_model'
        else:
            required_model_name = 'low_noise_model'
```

```python
# 配置 wan_t2v_A14B.py
boundary = 0.875  # 分界点：sigma >= 0.875 用高噪声专家
sample_guide_scale = (3.0, 4.0)  # 低噪声/高噪声各自的 CFG 权重
```

#### 5.1.2 工作原理

```
t=1.0 ────────────────── t=0.875 ─────────────────── t=0.0
   高噪声专家模型                 低噪声专家模型
  （处理粗略结构）              （处理精细细节）
   CFG scale=4.0                 CFG scale=3.0
```

**同一时刻只有一个模型在 GPU 上**，另一个卸载到 CPU（节省显存）。

#### 5.1.3 与传统 MoE 的对比

| 对比项 | 传统 MoE | Wan 2.2 "MoE" |
|--------|----------|----------------|
| 专家数量 | 多个小专家 | 2 个完整大模型 |
| 路由方式 | 门控网络逐 token 路由 | 按时间步阈值切换 |
| 同时激活 | 每次激活 Top-K 专家 | 同时只有 1 个模型 |
| 参数共享 | 专家间参数独立 | 两个完整独立模型 |
| 粒度 | token 级 | 时间步级 |

**设计动机**：去噪过程的不同阶段需要不同能力——高噪声阶段需要把握全局结构，低噪声阶段需要精细化细节。用两个各自专精的模型比一个通用模型效果更好。

### 5.2 Wan 2.2 VAE 升级

Wan 2.2 引入了新版 VAE（`vae2_2.py`），主要变化：

| 对比项 | Wan 2.1 VAE | Wan 2.2 VAE |
|--------|-------------|-------------|
| **z_dim** | 16 | **48** |
| **压缩比** | 4 × 8 × 8 | 4 × **16 × 16** |
| **encoder 基础通道** | 96 | **160** |
| **decoder 基础通道** | 96 | **256** |
| **输入处理** | 直接 RGB | **Patchify (p=2)**：像素重排 |
| **时间下采样** | [False, True, True] | **[True, True, True]** |
| **shortcut** | 学习的 1×1 CausalConv3d | **AvgDown3D / DupUp3D**（无参数） |

#### Patchify 的关键创新

```python
def patchify(x, patch_size=2):
    # [B, 3, T, H, W] → [B, 3*4=12, T, H/2, W/2]
    x = rearrange(x, "b c f (h q) (w r) -> b (c r q) f h w", q=2, r=2)

def unpatchify(x, patch_size=2):
    # [B, 12, T, H/2, W/2] → [B, 3, T, H, W]
    x = rearrange(x, "b (c r q) f h w -> b c f (h q) (w r)", q=2, r=2)
```

**效果**：在输入 VAE 之前先做无损的空间 2×2 像素重排（3ch → 12ch），使 VAE 内部在半分辨率上处理，最终 latent 空间分辨率进一步缩小。

**影响**：对于同样的 720P 视频，Wan 2.2 VAE 产生的 latent token 数是 Wan 2.1 的 **1/4**（空间维度每个方向再 /2），大幅加速 DiT 推理。

### 5.3 TI2V 5B 高效管线

Wan 2.2 提供了一个 **5B 参数**的高效模型（`textimage2video.py`），特点：
- 使用 Wan 2.2 VAE（压缩比 4×16×16）
- 支持 T2V + I2V 双模式
- 720P@24fps
- **可在消费级显卡（4090）上运行**

**关键技术：Per-Token Timestep**

在 I2V 模式下，已知图像对应的 token 不应该被加噪/去噪。Wan 2.2 TI2V 实现了**逐 token 时间步**：

```python
# 源码 textimage2video.py
# 已知区域的 token：timestep = 0（不加噪）
# 待生成区域的 token：timestep = t（正常加噪）
temp_ts = (mask * timestep).flatten()  # mask=0 的位置 timestep=0
```

---

## 六、核心疑难问题专题答疑

### Q1: 3D Causal Conv 详解

**问题**：为什么需要 3D Causal Conv？它是如何实现的？

**为什么需要？**

视频是时间+空间的三维数据。普通 2D 卷积只能处理单帧，无法捕捉帧间的时间关系。3D 卷积的核 `[kT, kH, kW]` 同时在时间和空间上滑动，能学习到**时空联合特征**（如运动模式、帧间连贯性）。

**为什么要"因果"？**

在生成任务中，第 $t$ 帧的特征只应依赖 $\leq t$ 的帧（自回归/因果假设）。如果允许看到未来帧，会导致信息泄露，破坏生成的因果逻辑。

**如何理解 3D 卷积？**

把 3D 卷积想象成一个**小立方体滤波器**在视频"数据立方体"上滑动：

```
普通图像上的 2D 卷积（3×3 核）：
┌─────┐
│ x x x│  ← 3×3 窗口在 H×W 上滑动
│ x o x│     o 是中心像素
│ x x x│
└─────┘

视频上的 3D 卷积（3×3×3 核）：
frame t-1:  frame t:    frame t+1:
┌─────┐   ┌─────┐    ┌─────┐
│ x x x│   │ x x x│    │ x x x│   ← 3×3×3 立方体
│ x x x│   │ x o x│    │ x x x│     在 T×H×W 上滑动
│ x x x│   │ x x x│    │ x x x│
└─────┘   └─────┘    └─────┘

因果 3D 卷积（3×3×3 核，因果 padding）：
frame t-2:  frame t-1:  frame t:
┌─────┐   ┌─────┐    ┌─────┐
│ x x x│   │ x x x│    │ x x x│   ← 只能看到过去2帧
│ x x x│   │ x x x│    │ x o x│     不能看到未来
│ x x x│   │ x x x│    │ x x x│
└─────┘   └─────┘    └─────┘
```

**实现方式**：把时间维 padding 全部放到左侧（过去），右侧为 0。参见 [2.2.3 节](#223-关键组件3d-causal-convolution) 的源码分析。

---

### Q2: VAE 中的卷积操作

**问题**：VAE 在哪里用到卷积？Encoder 还是 Decoder？DiT 有卷积吗？

**答案**：

**Encoder 和 Decoder 都大量使用 3D Causal Conv**。它们的结构是对称的：
- Encoder：3D CausalConv（输入层）→ 多级 ResidualBlock（每个含2个 CausalConv）→ 空间/时间下采样（Resample，内部也是 CausalConv）→ middle → head
- Decoder：CausalConv（输入层）→ middle → 多级 ResidualBlock → 空间/时间上采样 → CausalConv（输出层）

**动机**：卷积是处理网格数据（图像/视频）的经典操作。3D 卷积能高效地捕捉局部时空特征，且参数共享使模型紧凑（仅 127M）。

**你的理解需要修正的地方**：
- VAE 不是"把一个视频帧压缩为一个 latent z"——它是把**整个视频**（多帧）压缩为一个**4D latent tensor**
- 卷积发生在 Encoder（压缩时）**和** Decoder（解压时）
- Encoder 的卷积提取特征并下采样；Decoder 的卷积重建特征并上采样

**DiT 中的卷积**：
- DiT 的 `patch_embedding` 是一个 `Conv3d`（用于将 latent 转换为 token 序列）
- 但 DiT 的主体是 Transformer（自注意力 + 交叉注意力 + FFN），**不使用卷积**
- DiT 的设计哲学是用全局注意力（而非局部卷积）来建模长程依赖

---

### Q3: Feature Cache 机制详解

**问题**：Feature Cache 是什么？chunk by chunk 就是 Feature Cache 吗？为什么需要 cache？

**核心回答**：

"chunk by chunk" 是**分块处理策略**，Feature Cache 是使这个策略**正确工作**的配套机制。

**为什么需要分块？** 一个 81 帧 720P 视频直接通过 3D 卷积网络会**爆显存**。解决方案是把视频按时间切成小 chunk 逐个处理。

**为什么需要 Cache？** 因为 **3D 因果卷积的核需要看到前面几帧**。如果简单切片，每个 chunk 的第一帧就丢失了时间上下文。Cache 的作用是**保存上一个 chunk 末尾的特征帧**，作为下一个 chunk 的"过去上下文"。

**完整工作流程**：

```
视频 81 帧 → 分成 chunks：[第1帧] [第2-5帧] [第6-9帧] ... [第78-81帧]
                           chunk0    chunk1     chunk2        chunk20

处理 chunk0：
  - 每个 CausalConv3d：无 cache（用零 pad 代替），处理 1 帧
  - 保存末尾 2 帧特征到 cache[layer_i]

处理 chunk1：
  - 每个 CausalConv3d：从 cache[layer_i] 取出 2 帧拼到前面，处理 4 帧
  - 更新 cache[layer_i] = 当前 chunk 末尾 2 帧

处理 chunk2：
  - 同 chunk1，用 chunk1 的 cache ...

... 以此类推
```

**两种设置的区别**：
- **Default（无时间下采样）**：`CACHE_T=2`，缓存 2 帧，因为 kernel_size=3 的因果 Conv 左侧需要 2 帧上下文
- **2× temporal downsampling**：stride=2 会使帧数减半，只需缓存 1 帧

参见 [2.2.4 节](#224-feature-cache-机制详解) 的源码分析。

---

### Q4: 视频 latent 如何 token 化

**问题**：视频 latent 是 4D 的 `[C, T, H, W]`，如何变成 1D token 序列？

**答案**：通过一个 `Conv3d(kernel=stride=(1,2,2))` 做"打补丁"，然后展平。

**具体过程**（以 81 帧 720P 为例）：

```
Step 1: VAE 压缩
  视频 [3, 81, 720, 1280] → latent [16, 21, 90, 160]
                               C    T    H    W

Step 2: Patchify (Conv3d)
  [16, 21, 90, 160] → Conv3d(16→5120, k=(1,2,2), s=(1,2,2)) → [5120, 21, 45, 80]
  时间不变，空间各 /2

Step 3: Flatten
  [5120, 21, 45, 80] → flatten → [5120, 21×45×80] = [5120, 75600]
  → transpose → [75600, 5120]

最终：75,600 个 token，每个 5120 维
```

**每个 token 代表什么？**
- 1 个 token = latent 空间中的 1×2×2 小块
- 对应原始像素空间中的 4×16×16 区域（时间4帧，空间16×16像素）
- 所有 token 排列顺序：先 T，再 H，最后 W（按 3D 网格展平）

**Unpatchify（还原）**：

```python
# [L, dim] 中每个 token → [1, 2, 2, C] 的小块
# 重组为 [C, T_lat, H_lat, W_lat]
u = torch.einsum('fhwpqrc->cfphqwr', u)
u = u.reshape(c, *[i*j for i,j in zip(grid_size, patch_size)])
```

---

### Q5: VACE 是对 Wan 的封装吗

**问题**：VACE 是不是对 Wan 的一次封装？需要从头训练吗？

**答案**：

**可以把 VACE 理解为在 Wan-T2V 上"加装"一个条件处理模块**（Context Adapter），类似给手机装了一个带各种镜头的手机壳。

**不需要从头训练！** 这是 VACE 的核心优势：

1. **基座模型（Wan-T2V-14B）完全冻结**——它已经学会了视频生成的全部能力
2. **只训练新增的部分**：
   - Context Embedder（条件输入的 token 化层）
   - 8 个 Context Blocks（从原始 DiT 层复制并独立训练）
3. **关键初始化**：Context Embedder 中 F_c/F_k 对应的权重**从原始 patch_embedding 复制**（因为它们共享同一个 VAE latent 空间），M 对应的权重**初始化为零**

**训练目的是什么？** 训练 Context Adapter 学会：
- 理解 VCU 输入格式（什么是 reactive/inactive 帧）
- 将条件信息编码为 hints
- 在正确的层向正确的位置注入 hints

**为什么冻结基座就够了？** 因为基座已经具备了强大的视频生成能力。Context Adapter 只需学会"告诉基座该在哪里生成什么"——这是一个相对简单的任务，200K 步训练即可收敛。

---

### Q6: VACE 组合任务的训练与泛化

**问题**：VACE 只训练 4 种基础任务吗？组合任务怎么保证能力？能否做目标替换任务？

**答案**：

**训练时确实包含了组合任务！** 论文明确说明：
> "All tasks are randomly combined during training to accommodate a broader range of application scenarios including compositional tasks."

但组合并非穷举所有可能——而是**随机采样**一部分组合作为训练数据（12种基础任务 + 随机组合）。

**泛化到未见组合的原理**：
- VCU 是一个**连续的、统一的表示空间**——任何任务只是 $(F, M)$ 的不同配置
- 模型学会了理解 mask 的语义（1=生成，0=保留）和帧的语义（reactive vs inactive）
- 基础任务的组合本质上就是**拼接不同配置的 F 和 M**，模型能自然泛化

**你的具体需求（目标替换任务）**：

> 给一个 source video, ref image, prompt，对 source video 里指定产品替换为 ref image 的产品

**可以实现！** VCU 配置为：

```
F = [ref_image, source_frame_1, source_frame_2, ..., source_frame_n]
M = [0_{h×w},   m_1,            m_2,            ..., m_n           ]

其中 m_i 是二值 mask：产品所在区域 = 1（需要替换），其他区域 = 0（保留）
```

这其实是 **R2V + MV2V 的组合**（参考图像 + 局部 mask 编辑），正好是 VACE 论文中展示的 "Swap Anything" 场景。

---

### Q7: Wan-Animate vs VACE

**问题**：VACE 不是已经统一了所有视频任务吗？Wan-Animate 存在的意义是什么？

**答案**：VACE 是"瑞士军刀"，Wan-Animate 是"精密手术刀"。

**VACE 在角色动画上的不足**（论文原文）：
- "VACE shows instability in character animation tasks"
- "Issues with identity consistency"
- "Highly dependent on parameter tuning, resulting in a higher barrier to entry"

**Wan-Animate 在角色动画上的专有优势**：

| 能力 | VACE | Wan-Animate |
|------|------|-------------|
| 精细表情控制 | 无专用机制 | Face Adapter（隐式面部特征 + 时间对齐交叉注意力） |
| 骨骼驱动 | 可用深度/姿态控制信号 | Body Adapter（直接加到 latent 上，逐帧精确控制） |
| 身份-表情解耦 | 无 | 1D压缩 + 数据增强 |
| 重打光 | 无 | Relighting LoRA |
| 姿态重定向 | 无 | 骨骼长度调整 + T-pose 辅助 |
| 身份一致性 | 不稳定 | 专门优化 |

**本质原因**：角色动画对**精细控制**要求极高（微妙的嘴唇动作、眼神变化、肢体协调），这需要专门设计的信号通路（Face Adapter / Body Adapter），而非通用的条件注入框架。

---

### Q8: Adapter 机制详解

**问题**：Context Adapter 和 Face Adapter 是什么？如何理解？

**Adapter 的通用概念**：在一个冻结的大模型上，添加少量可训练参数（adapter），使模型获得新能力而不破坏原有能力。类比：给汽车加装导航模块，不改动发动机。

#### Context Adapter（VACE）

**类型**：Res-Tuning 风格的旁路分支

```
                    冻结的 DiT Block
输入 x ──────────→ Self-Attn → Cross-Attn → FFN → 输出
                                                    │
Context tokens ──→ Context Block (可训练) → hint ──→ ⊕（加法注入）
```

**输入**：VCU 编码后的 context tokens（inactive latent + reactive latent + mask）
**输出**：hints（与主干 token 同维度的加性信号）
**特点**：
- 8 个 Context Blocks 分布在 DiT 的不同层（如 [0, 5, 10, 15, 20, 25, 30, 35]）
- 零初始化的投影保证训练初期不影响原模型
- 可插拔：移除 adapter 后模型回到原始 T2V 状态

#### Face Adapter（Wan-Animate）

**类型**：专用交叉注意力模块

```
视频 tokens (Q) ───→ Cross-Attention ───→ 输出（残差加回主干）
                         ↑
面部 embedding (K, V) ──┘
```

**输入**：面部图像经 Face Encoder 编码的 motion vectors（每帧一个）
**输出**：面部表情控制信号（与视频 token 同维度）
**特点**：
- 每 5 层注入一次（40 层 DiT 中 8 个注入点）
- **时间对齐**：每帧的视频 token 只与对应时间步的面部 embedding 做 attention
- Face Encoder 将 512×512 面部图像压缩为 1D 向量，通过 1D 因果卷积做时间降采样（对齐到 latent 时间维）

---

### Q9: MoE 是什么

#### 传统 MoE（Mixture of Experts）

**核心思想**：与其训练一个巨大的模型处理所有类型的输入，不如训练多个小"专家"模型，每个专家擅长处理特定类型的输入。一个"门控网络"负责决定每个输入该由哪些专家处理。

**结构**：
```
输入 x ──→ 门控网络 G(x) ──→ 权重分配 [w_1, w_2, ..., w_N]
  │
  ├──→ Expert 1 ──→ y_1
  ├──→ Expert 2 ──→ y_2
  ...
  └──→ Expert N ──→ y_N

输出 = w_1 * y_1 + w_2 * y_2 + ... + w_N * y_N
```

**关键公式**：

$$y = \sum_{i=1}^{N} G(x)_i \cdot E_i(x)$$

- $E_i(x)$：第 $i$ 个专家的输出
- $G(x)_i$：门控网络对第 $i$ 个专家的权重（通常 softmax 后只保留 Top-K）
- 稀疏激活：每次只有 K 个专家被激活（如 Top-2），其余权重为 0

**门控网络**：
$$G(x) = \text{TopK}(\text{Softmax}(W_g \cdot x + \epsilon))$$

- $W_g$：可学习的门控权重矩阵
- $\epsilon$：可选的噪声（训练时促进专家均衡使用）

**优势**：
- 以少量计算成本获得大模型容量（每次只激活部分专家）
- 不同专家可以专注于不同子任务

**经典应用**：GShard、Switch Transformer、Mixtral

#### Wan 2.2 的"MoE"

Wan 2.2 的实现**不是**传统 MoE，而是**双完整模型按时间步切换**（详见 [5.1 节](#51-双专家模型策略moe)）。

但其思想与 MoE 类似：
- **专家划分维度**：不是按 token 类型，而是按**去噪阶段**（高噪声 vs 低噪声）
- **路由机制**：不是门控网络，而是简单的阈值判断（sigma ≥ 0.875）
- **效果**：高噪声专家擅长把握全局结构，低噪声专家擅长精细化细节——"分工合作"

**为什么这样设计？** 论文的说法是"separating the denoising process across timesteps with specialized powerful expert models, this enlarges the overall model capacity while maintaining the same computational cost"——每一步推理只用一个14B模型（算力不变），但总参数量翻倍（容量增大）。

---

### Q10: Attention 机制全景分析

**问题**：Wan 的 attention 机制在哪些组件中使用？VAE 和 DiT 中各自如何实现？统一时空注意力是如何工作的？

#### 10.1 Attention 在 Wan 各组件中的使用总览

| 组件 | 是否使用 Attention | 类型 | Token 是什么 | 目的 |
|------|-------------------|------|-------------|------|
| **Wan-VAE Encoder** | 是（middle层） | 逐帧空间自注意力 | 单帧像素网格 H×W | 增强空间特征的全局感知 |
| **Wan-VAE Decoder** | 是（middle层） | 逐帧空间自注意力 | 单帧像素网格 H×W | 增强重建的空间一致性 |
| **Wan-DiT** | 是（核心） | 全时空自注意力 + 交叉注意力 | 视频 latent 的 3D patch | 建模完整的时空依赖关系 |
| **VACE Context Blocks** | 是 | 与 DiT 同构 | 条件 latent 的 3D patch | 编码编辑条件 |
| **Wan-Animate Face Block** | 是 | 时间对齐交叉注意力 | 视频 token(Q) / 面部特征(KV) | 注入面部表情控制 |

#### 10.2 VAE 中的 Attention：逐帧空间注意力

VAE 的 Attention 出现在 Encoder 和 Decoder 的 **middle 层**。它是**逐帧独立**的**单头空间**自注意力：

```python
# 源码 vae.py 第375-438行 AttentionBlock
class AttentionBlock(nn.Module):
    def __init__(self, dim):
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)  # 用 1×1 Conv2d 生成 Q/K/V
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):  # x: [B, C, T, H, W]
        b, c, t, h, w = x.size()
        # 关键：把时间维并入 batch，每帧独立做注意力
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.norm(x)
        # Q/K/V: [(B*T), 1, H*W, C]  ← 单头，token数 = H*W
        q, k, v = self.to_qkv(x).reshape(b*t, 1, c*3, -1) \
            .permute(0, 1, 3, 2).chunk(3, dim=-1)
        x = F.scaled_dot_product_attention(q, k, v)  # 每帧内部空间注意力
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)
        return x + identity  # 残差
```

**动机**：VAE 的主要工作由 3D Causal Conv 完成（捕捉局部时空特征）。Attention 只在 middle 层使用一次，作用是让**同一帧内远距离空间位置**能互相感知（弥补局部卷积的感受野不足）。由于 VAE 特征图分辨率在 middle 层已经很小（经过多次下采样），这个注意力的计算量很小。

**为什么不做时空联合注意力？** 因为 VAE 的目标是低级特征的压缩/重建，不需要复杂的跨帧全局推理。3D Causal Conv 已经足够处理帧间的时间依赖。且全时空注意力的 token 数 = T×H×W，在像素/浅层特征空间中太大。

#### 10.3 DiT 中的 Attention：全时空统一注意力

**你的理解完全正确**——Wan-DiT 是在时空维度上**统一做 attention**。

这是 Wan-DiT 最核心的设计。每个 Transformer Block 包含：

```python
# 源码 model.py WanAttentionBlock.forward（第438-495行）
def forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens):
    e = (self.modulation + e).chunk(6, dim=1)

    # (1) 全时空 Self-Attention
    y = self.self_attn(
        self.norm1(x).float() * (1 + e[1]) + e[0],  # AdaLN 调制
        seq_lens, grid_sizes, freqs)  # 3D RoPE + Flash Attention
    x = x + y * e[2]  # gated residual

    # (2) Cross-Attention（文本条件）
    x = x + self.cross_attn(self.norm3(x), context, context_lens)

    # (3) FFN
    y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
    x = x + y * e[5]
    return x
```

**"统一"的含义**：在 Self-Attention 中，所有 75,600 个 token（来自 21×45×80 的 3D 网格）被**拍平成一条序列**，每个 token 与其他**所有** token 做注意力。一个"空间角落第1帧"的 token 可以直接 attend 到"空间中心第20帧"的 token——完全没有时间/空间的人为分割。

**3D RoPE 是关键**：虽然 token 序列是 1D 的，但 3D RoPE 为每个 token 编码了它的 3D 坐标 (f, h, w)，使得注意力权重自然地反映**时空距离**——近邻 token 的注意力权重更大。

```python
# Self-Attention 内部（model.py 第232-271行）
def forward(self, x, seq_lens, grid_sizes, freqs):
    q = self.norm_q(self.q(x)).view(b, s, n, d)
    k = self.norm_k(self.k(x)).view(b, s, n, d)
    v = self.v(x).view(b, s, n, d)

    # 对 Q 和 K 应用 3D RoPE（V 不做 RoPE）
    x = flash_attention(
        q=rope_apply(q, grid_sizes, freqs),  # 3D 旋转位置编码
        k=rope_apply(k, grid_sizes, freqs),
        v=v,
        k_lens=seq_lens)  # 所有 token 互相 attend
    return self.o(x.flatten(2))
```

#### 10.4 时空统一注意力 vs 时空分离注意力：深度对比

##### 时空分离注意力（Temporal-Spatial Separated Attention）

这是 **VDM、Video LDM、AnimateDiff、ModelScope** 等早期视频模型的常见做法：

```python
# 伪代码：时空分离注意力
class SeparatedAttentionBlock:
    def forward(self, x):  # x: [B, T, H, W, C]
        # Step 1: 空间注意力 —— 每帧独立
        for t in range(T):
            x[:, t] = spatial_attn(x[:, t])  # token数 = H*W

        # Step 2: 时间注意力 —— 每个空间位置独立
        for h, w in spatial_positions:
            x[:, :, h, w] = temporal_attn(x[:, :, h, w])  # token数 = T
```

**实际实现方式**：
```
空间注意力：reshape [B*T, H*W, C] → self-attention → reshape back
时间注意力：reshape [B*H*W, T, C] → self-attention → reshape back
```

##### 对比分析

| 对比维度 | 时空统一注意力（Wan） | 时空分离注意力（早期方法） |
|----------|----------------------|--------------------------|
| **Token 数** | T×H×W（如 75,600） | 空间：H×W（如 3,600）；时间：T（如 21） |
| **计算复杂度** | $O((THW)^2)$，非常大 | $O(T \cdot (HW)^2 + HW \cdot T^2)$，小得多 |
| **跨时空建模** | 任意 token 对之间都有直接连接 | 空间和时间分开建模，**无法直接建模斜向运动** |
| **运动建模能力** | 强——如物体从左下角移动到右上角，路径上所有 token 都有直接注意力连接 | 弱——这种运动需要信息先通过空间注意力传播到同帧的其他位置，再通过时间注意力传到其他帧，属于**间接连接** |
| **实现难度** | 需要高效注意力实现（Flash Attention） | 相对简单 |
| **显存占用** | 大（靠 Flash Attention + Context Parallel 缓解） | 小 |
| **位置编码** | 3D RoPE（一次性编码 F/H/W） | 通常 2D PE（空间）+ 1D PE（时间）分别加 |

##### 时空统一注意力是 Wan 首次提出的吗？

**不是。** 时空统一注意力的思想可以追溯到更早的工作：

- **ViViT (2021, Google)**：提出了 Factorized 和 Unfactorized（即统一时空）两种变体
- **TimeSformer (2021, Facebook)**：对比了多种时空注意力模式
- **Latte (2024)**：在视频扩散模型中明确对比了统一 vs 分离注意力
- **Open-Sora / Open-Sora-Plan**：也使用了统一时空注意力
- **Sora (2024, OpenAI)**：据技术报告描述使用统一时空 Transformer

Wan 的贡献不在于"发明"统一时空注意力，而在于**将其高效地工程化到 14B 规模**（通过 Flash Attention + 2D Context Parallelism + FSDP），使 75K+ token 的全注意力在实际中可行。

##### 为什么 Wan 选择统一而非分离？

1. **更强的时空一致性**：视频中的运动是时空耦合的，分离注意力本质上假设时间和空间可以独立处理——这个假设对复杂运动不成立
2. **更简洁的架构**：不需要在空间注意力和时间注意力之间交替，一个 Block 就搞定
3. **Flash Attention 使其可行**：Flash Attention 将注意力的显存从 $O(N^2)$ 降到 $O(N)$，75K token 的全注意力变得实际可行
4. **实验验证**：论文中 Wan 在 VBench 和人类评估中全面领先，间接证明了这种设计的优越性

---

### Q11: VACE 的 DiT 层结构与 Context Block 注入位置

**问题**：VACE 是 40 层 DiT + 8 个 Context Blocks，DiT 的一层到底包含什么？Context Blocks 具体注入在哪里？

#### 11.1 DiT 的"一层"定义

**DiT 的一层 = 一个 `WanAttentionBlock`**，它包含三个子层：

```
WanAttentionBlock（DiT 的一层）:
  ┌─────────────────────────────────────────┐
  │  (1) AdaLN + Self-Attention + Gate      │  ← 全时空自注意力
  │  (2) Cross-Attention                     │  ← 文本条件注入
  │  (3) AdaLN + FFN + Gate                  │  ← 前馈网络
  └─────────────────────────────────────────┘
```

**源码确认**（`model.py` 第685-689行）：

```python
self.blocks = nn.ModuleList([
    WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads, ...)
    for _ in range(num_layers)  # num_layers = 40（14B 模型）
])
```

所以 **40 层 DiT = 40 个 `WanAttentionBlock`**，编号 0~39。

**如何验证有 40 层？** 看配置文件：

```python
# Wan2.1/wan/configs/wan_t2v_14B.py
wan_t2v_14B.num_layers = 40
wan_t2v_14B.dim = 5120
wan_t2v_14B.num_heads = 40
```

1.3B 模型是 30 层（`num_layers=30`），14B 模型是 40 层。

#### 11.2 VACE 的 Context Blocks 注入位置

VACE 版本中，40 层主干 DiT 被替换为 `BaseWanAttentionBlock`（结构与原始 `WanAttentionBlock` 完全相同，只是多了 hint 注入能力）。

**Context Blocks 的数量和位置取决于 `vace_layers` 配置**：

```python
# 源码 vace_model.py 第212-213行
# 默认值：每隔一层注入 → [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]（32层模型）
self.vace_layers = [i for i in range(0, self.num_layers, 2)] if vace_layers is None else vace_layers
```

对于 **Wan2.1-T2V-14B（40层）上的 VACE**，论文中使用的是：
```python
vace_layers = [0, 5, 10, 15, 20, 25, 30, 35]  # 8 个注入点，均匀分布
```

#### 11.3 注入的完整数据流

```
Step 1: 条件分支处理（forward_vace）
─────────────────────────────────

vace_context [C_vace, T, H, W]
    │
    ▼  vace_patch_embedding (Conv3d)
c tokens [B, L, dim]

    ▼  VaceWanAttentionBlock #0:
       c = before_proj(c) + x       ← 与主干 token 对齐
       c = SelfAttn + CrossAttn + FFN(c)
       hints[0] = after_proj(c)      ← 产生第 0 个 hint

    ▼  VaceWanAttentionBlock #1:
       c = SelfAttn + CrossAttn + FFN(c)
       hints[1] = after_proj(c)

    ...（共 8 个 VaceWanAttentionBlock）...

    ▼  VaceWanAttentionBlock #7:
       hints[7] = after_proj(c)

结果：hints = [hint_0, hint_1, ..., hint_7]，每个 shape [B, L, dim]


Step 2: 主干注入
──────────────

DiT Block 0  (block_id=0):  x = Block(x);  x = x + hints[0] * scale  ✅ 注入
DiT Block 1  (block_id=None): x = Block(x)                              不注入
DiT Block 2  (block_id=None): x = Block(x)                              不注入
DiT Block 3  (block_id=None): x = Block(x)                              不注入
DiT Block 4  (block_id=None): x = Block(x)                              不注入
DiT Block 5  (block_id=1):  x = Block(x);  x = x + hints[1] * scale  ✅ 注入
DiT Block 6~9:  不注入
DiT Block 10 (block_id=2):  x = Block(x);  x = x + hints[2] * scale  ✅ 注入
...
DiT Block 35 (block_id=7):  x = Block(x);  x = x + hints[7] * scale  ✅ 注入
DiT Block 36~39:  不注入
```

**注入方式的源码**（`vace_model.py` 第131-151行）：

```python
class BaseWanAttentionBlock(WanAttentionBlock):
    def forward(self, x, hints, context_scale=1.0, **kwargs):
        # 先正常跑完 Self-Attn + Cross-Attn + FFN
        x = super().forward(x, **kwargs)
        # 在指定层做加法注入
        if self.block_id is not None:
            x = x + hints[self.block_id] * context_scale  # 就这一行！
        return x
```

**注入方式极其简洁**：就是一个**加法**（`x = x + hint * scale`），没有任何复杂的门控、投影或融合机制。这也是为什么它能快速收敛——注入的信号路径短而直接。

---

### Q12: 如何理解 hints 与 VACE 不需要从零训练的原因

#### 12.1 Hints 是什么？

**"Hints" 不是一个标准的学术专有名词**，而是 VACE 论文/代码中使用的自定义术语。它可以理解为**编辑提示信号**或**条件残差信号**。

在更广泛的文献中，类似概念有不同叫法：
- **ControlNet** 中叫 "zero convolution output" / "residual"
- **IP-Adapter** 中叫 "image prompt embedding"
- **Res-Tuning** 论文中叫 "auxiliary features"
- VACE 中叫 **hints**

**本质**：hints 是 Context Blocks 对条件信息（源视频、mask、参考图像）的处理结果，编码了"在哪里生成什么样的内容"这类编辑指令。它们以**加法残差**的形式注入到主干 DiT 中，"提示"主干应该如何修改其生成行为。

#### 12.2 为什么 VACE 不需要从零训练？——四层保障机制

**核心原理**：通过精心的初始化设计，保证训练开始时 VACE 模型的行为**与原始 Wan-T2V 完全相同**，然后在训练过程中逐步学会利用条件信息。

**第一层保障：`after_proj` 零初始化**

```python
# vace_model.py 第58-61行
self.after_proj = nn.Linear(self.dim, self.dim)
nn.init.zeros_(self.after_proj.weight)  # 权重全零
nn.init.zeros_(self.after_proj.bias)    # 偏置全零
```

**效果**：训练开始时，`hint = after_proj(c) = 0`。所有 hints 都是零向量，注入到主干的信号为零。

**含义**：主干 DiT 完全感受不到 Context Blocks 的存在，行为与冻结前一模一样。

**第二层保障：`before_proj` 零初始化**

```python
# vace_model.py 第52-56行
self.before_proj = nn.Linear(self.dim, self.dim)
nn.init.zeros_(self.before_proj.weight)
nn.init.zeros_(self.before_proj.bias)
```

**效果**：第一个 Context Block 中 `c = before_proj(c) + x = 0 + x = x`。条件流从主干 token 的值开始。

**第三层保障：主干参数完全冻结**

Context Adapter Tuning 策略下：
- 原始 DiT 的所有参数**不更新**
- 只训练 Context Embedder + Context Blocks

**含义**：无论 Context Blocks 怎么训练，主干的 T2V 能力**永远不会退化**。

**第四层保障：Context Embedder 的权重复制**

```
Context Embedder 中：
  - F_c 对应权重 ← 从原始 patch_embedding 复制（F_c 和 x 共享同一个 VAE latent 空间）
  - F_k 对应权重 ← 从原始 patch_embedding 复制（同理）
  - M 对应权重 ← 零初始化（mask 一开始不产生任何影响）
```

**训练过程的直觉解释**：

```
训练开始（第 0 步）：
  hints = [0, 0, ..., 0]   →   主干行为 = 纯 T2V
  损失来自：主干忽略了条件信息，生成不符合编辑要求

训练中期（~50K 步）：
  hints 逐渐变为非零  →   主干开始"听到"编辑指令
  after_proj 学会了把条件信息编码为有意义的 hint
  before_proj 学会了把 vace_context 对齐到主干空间

训练后期（~200K 步）：
  hints 提供精确的编辑引导  →   主干能准确执行各种编辑任务
  但 hints = 0 时（无条件情况），主干仍然是完美的 T2V 模型
```

这就是为什么不需要从零训练——**整个训练过程是从"纯 T2V"到"T2V + 编辑能力"的平滑过渡**。

---

### Q13: Adapter 范式全面对比与必读论文

#### 13.1 你的理解完全正确

**Adapter 是一种参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）的思想**，核心是：冻结大模型主体，只训练少量新增参数，使模型获得新能力。具体的实现方式有很多种。

#### 13.2 主流 Adapter 实现方式对比

##### (1) LoRA（Low-Rank Adaptation）

**核心思想**：不修改原始权重 $W$，而是学习一个低秩增量 $\Delta W = BA$。

```
原始：y = Wx
LoRA：y = Wx + BAx    其中 B∈R^{d×r}, A∈R^{r×k}, r << min(d,k)

训练时：W 冻结，只训练 A 和 B
推理时：可合并 W' = W + BA，无额外推理开销
```

**实现**：

```python
class LoRALinear(nn.Module):
    def __init__(self, original_linear, rank=16, alpha=16):
        self.original = original_linear  # 冻结
        self.lora_A = nn.Linear(original_linear.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, original_linear.out_features, bias=False)
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)  # B 初始化为零 → 初始 ΔW = 0

    def forward(self, x):
        return self.original(x) + self.lora_B(self.lora_A(x)) * self.scale
```

| 特点 | 说明 |
|------|------|
| 参数量 | 极少（rank=16 时约 0.1% 原始参数） |
| 推理开销 | 可合并，零额外开销 |
| 应用场景 | 风格微调、领域适配、个性化。Wan-Animate 的 Relighting LoRA 即此类 |
| 局限 | 表达能力受 rank 限制；无法注入全新类型的信息（如新模态） |

**必读论文**：
- **LoRA: Low-Rank Adaptation of Large Language Models** (Hu et al., 2022) — 开山之作
- **QLoRA** (Dettmers et al., 2023) — 量化 + LoRA
- **LoRA+** (Hayou et al., 2024) — 改进学习率策略

---

##### (2) Res-Tuning / 旁路分支 Adapter（VACE 的 Context Adapter）

**核心思想**：在冻结的主干旁添加一条**并行分支**，分支的输出以**加法残差**注入主干。

```
主干（冻结）：  x → Block_i(x) → x_out
旁路分支（可训练）：c → AdapterBlock_i(c) → hint
注入：  x_final = x_out + hint * scale
```

**VACE 的具体实现**：

```python
# 条件分支
c, c_skip = VaceWanAttentionBlock(c, x, **kwargs)  # c_skip 即 hint
# 主干注入
x = BaseWanAttentionBlock(x, **kwargs)
x = x + hints[block_id] * context_scale
```

| 特点 | 说明 |
|------|------|
| 参数量 | 中等（8 个完整 Transformer Block 的参数） |
| 推理开销 | 有额外开销（需要同时运行分支） |
| 应用场景 | 注入**全新模态的条件信息**（如视频编辑条件）。需要强大的条件编码能力 |
| 局限 | 参数量比 LoRA 大；分支增加推理延迟 |

**必读论文**：
- **Res-Tuning: A Flexible and Efficient Tuning Paradigm** (Jiang et al., 2023) — VACE 直接引用的基础
- **ControlNet: Adding Conditional Control to Text-to-Image Diffusion Models** (Zhang et al., 2023) — 最知名的此类方法
- **T2I-Adapter** (Mou et al., 2024) — 轻量级版本

---

##### (3) Cross-Attention Adapter（Wan-Animate 的 Face Adapter / IP-Adapter）

**核心思想**：通过**新增的交叉注意力层**将外部特征注入主干。主干 token 作为 Query，外部特征作为 Key/Value。

```
主干 token x (Query) ─── 新增 Cross-Attn ─── 外部特征 f (Key, Value)
                              │
                              ▼
                     x_final = x + cross_attn_output
```

**Wan-Animate Face Adapter 的具体实现**：

```python
# 源码 face_blocks.py
class FaceBlock(nn.Module):
    def forward(self, hidden_states, motion_embeddings):
        # Q 来自视频 token
        q = self.linear1_q(self.norm1(hidden_states))
        # K, V 来自面部运动特征
        kv = self.linear1_kv(self.norm2(motion_embeddings))
        k, v = kv.chunk(2, dim=-1)
        # 时间对齐的交叉注意力
        attn_output = temporal_aligned_cross_attention(q, k, v)
        return hidden_states + self.linear2(attn_output)  # 残差注入
```

| 特点 | 说明 |
|------|------|
| 参数量 | 较少（仅 Q/K/V 投影层 + 输出投影） |
| 推理开销 | 有额外开销（额外的注意力计算） |
| 应用场景 | 注入**密集的逐 token 条件信息**，如面部表情、IP 特征。适合需要**细粒度对齐**的场景 |
| 局限 | 需要外部特征与主干 token 在语义空间上有合理对应关系 |

**必读论文**：
- **IP-Adapter: Text Compatible Image Prompt Adapter** (Ye et al., 2023) — 最经典的此类方法
- **IP-Adapter-FaceID** — 面部 ID 保持变体

---

##### (4) Bottleneck Adapter（经典 Adapter / Houlsby Adapter）

**核心思想**：在 Transformer 的每个子层（Self-Attn / FFN）之后插入一个**瓶颈结构**（降维→非线性→升维）。

```
原始子层输出 x
    │
    ▼
Adapter: Linear(dim→bottleneck) → ReLU → Linear(bottleneck→dim)
    │
    ▼
x_final = x + adapter_output  # 残差
```

| 特点 | 说明 |
|------|------|
| 参数量 | 少（由 bottleneck 维度控制） |
| 推理开销 | 少量额外开销 |
| 应用场景 | NLP 中的任务适配（最早的 adapter 方法），CV 中较少直接使用 |
| 局限 | 表达能力受 bottleneck 限制，且无法注入外部模态信息 |

**必读论文**：
- **Parameter-Efficient Transfer Learning for NLP** (Houlsby et al., 2019) — Adapter 概念的开山之作
- **AdapterFusion** (Pfeiffer et al., 2021) — 多 Adapter 组合

---

##### (5) Prefix Tuning / Prompt Tuning

**核心思想**：在输入序列前添加**可学习的虚拟 token**（prefix），让模型通过注意力机制自动利用这些 prefix。

```
原始输入：[token_1, token_2, ..., token_n]
Prefix Tuning：[prefix_1, prefix_2, ..., prefix_m, token_1, token_2, ..., token_n]
                 ↑ 可学习参数            ↑ 冻结的原始输入
```

| 特点 | 说明 |
|------|------|
| 参数量 | 极少（仅 prefix token 的嵌入） |
| 推理开销 | 略增（序列变长） |
| 应用场景 | NLP 中常见；CV/视频中较少单独使用 |
| 局限 | 表达能力有限；占用序列长度 |

**必读论文**：
- **Prefix-Tuning** (Li & Liang, 2021)
- **Prompt Tuning** (Lester et al., 2021)

---

#### 13.3 综合对比表

| 方法 | 新增参数量 | 推理额外开销 | 能否注入新模态 | 核心原理 | 代表应用 |
|------|-----------|-------------|---------------|---------|---------|
| **LoRA** | ~0.1% | 可合并为零 | 否（仅调整行为） | 低秩分解 | Wan-Animate Relighting |
| **Res-Tuning** | ~10-20% | 有（并行分支） | 是 | 旁路分支+加法残差 | VACE Context Adapter |
| **Cross-Attn Adapter** | ~5% | 有（额外注意力） | 是 | 交叉注意力注入 | Wan-Animate Face Adapter, IP-Adapter |
| **Bottleneck Adapter** | ~1-5% | 少量 | 否 | 瓶颈结构残差 | NLP 任务适配 |
| **Prefix Tuning** | ~0.01% | 略增 | 否 | 虚拟 token | NLP prompt 适配 |

#### 13.4 Adapter 领域必读论文清单

**奠基性工作**：
1. **Houlsby et al. (2019)** — *Parameter-Efficient Transfer Learning for NLP* — Adapter 概念的起源
2. **Hu et al. (2022)** — *LoRA: Low-Rank Adaptation of Large Language Models* — 最广泛使用的 PEFT 方法

**视觉/扩散模型 Adapter**：
3. **Zhang et al. (2023)** — *ControlNet* — 扩散模型条件控制的里程碑
4. **Ye et al. (2023)** — *IP-Adapter* — 图像 prompt adapter，交叉注意力注入
5. **Mou et al. (2024)** — *T2I-Adapter* — 轻量级条件注入
6. **Jiang et al. (2023)** — *Res-Tuning* — VACE 的理论基础

**综述与进阶**：
7. **Pfeiffer et al. (2021)** — *AdapterHub / AdapterFusion* — 多 Adapter 组合框架
8. **He et al. (2022)** — *Towards a Unified View of Parameter-Efficient Transfer Learning* — 统一视角分析各种 PEFT 方法
9. **Lialin et al. (2023)** — *Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning* — 全面综述

---

## 七、面试高频问题

### Q: Diffusion Model vs Flow Matching 的区别？

| 对比项 | DDPM | Flow Matching |
|--------|------|---------------|
| 前向过程 | $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t} \epsilon$ | $x_t = t \cdot x_1 + (1-t) \cdot x_0$ |
| 预测目标 | 噪声 $\epsilon$ | 速度场 $v = x_1 - x_0$ |
| 路径 | 弯曲（基于马尔可夫链） | 直线（rectified flow） |
| 采样 | 逆向 SDE/ODE | 求解 ODE（更简洁） |
| 数学基础 | 随机微分方程 | 常微分方程 |

### Q: 为什么用 VAE 而不直接在像素空间做扩散？

- 81帧720P视频 = 81×720×1280×3 ≈ **2.38亿像素**
- VAE 压缩后 latent = 21×90×160×16 ≈ **483万数值**（约 50 倍压缩）
- DiT 在 latent 空间操作，计算量降低数十倍

### Q: RoPE 在视频生成中如何工作？

将 head_dim 拆成三段，分别对时间/高度/宽度位置做旋转编码。每个 token 的位置由 3D 坐标 (f, h, w) 决定，注意力能感知相对时空距离。

### Q: CFG（Classifier-Free Guidance）原理？

训练时随机丢弃条件（概率 p），让模型同时学有条件和无条件生成。推理时：
$$\text{output} = \text{uncond} + s \cdot (\text{cond} - \text{uncond})$$
$s > 1$ 放大条件影响，提高 prompt 跟随性。

### Q: I2V 中如何保证第一帧与输入图像一致？

通过**通道拼接**：将输入图像的 VAE latent + binary mask 拼接到噪声 latent 的通道维度上。mask 标记第一帧为"已知"，模型学会在去噪时保持第一帧不变。

### Q: VACE 的 Concept Decoupling 为什么有效？

不解耦时，模型需要从混合的 F 中同时理解"哪些是控制信号"和"哪些是保留内容"——这两种语义完全不同。解耦后，$F_c$（reactive）和 $F_k$（inactive）各自语义清晰，模型更容易学习。

### Q: Adapter 相比全参数微调的优势？

1. **收敛更快**（VACE 论文 Figure 5a 证实）
2. **可插拔**：移除后不影响基座模型
3. **参数高效**：只训练少量参数
4. **保护基座能力**：冻结原始权重，T2V 能力完全保留

### Q: Wan-VAE 的 RMSNorm vs GroupNorm？

Wan-VAE 用 RMSNorm 替换了所有 GroupNorm。原因：GroupNorm 在 batch 的通道分组上做归一化，会跨时间帧计算统计量，**破坏时间因果性**。RMSNorm 逐样本/逐通道归一化，与因果卷积兼容，且是 Feature Cache 机制正确工作的前提。

### Q: 视频生成中的主要挑战？

1. **时空一致性**：帧间内容/运动要连贯
2. **长序列建模**：75K+ token 的注意力计算
3. **多模态对齐**：文本/图像条件与视频内容的语义一致
4. **效率**：推理成本高（14B×50步×75K tokens）
5. **评估困难**：缺乏公认的自动评估指标

---

> 本文档基于以下材料编写：
> - Wan: Open and Advanced Large-Scale Video Generative Models (arXiv:2503.20314)
> - VACE: All-in-One Video Creation and Editing (arXiv:2503.07598)
> - Wan-Animate: Unified Character Animation and Replacement with Holistic Replication (arXiv:2509.14055)
> - Wan 2.1 源码 (https://github.com/Wan-Video/Wan2.1)
> - Wan 2.2 源码 (https://github.com/Wan-Video/Wan2.2)
