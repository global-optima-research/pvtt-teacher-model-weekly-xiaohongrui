# 为什么 Norm 就能让训练更稳定？

想象一下没有 Norm 的深层网络：
1.  **第一层输出**：$x_1$ 范围在 $[0, 1]$。
2.  **第二层权重**：$w_2$ 初始化运气不好，全是 10。
3.  **第二层输出**：$x_2 = w_2 \cdot x_1$，范围变成 $[0, 10]$。
4.  **第十层输出**：范围可能变成 $[0, 10^{10}]$（天文数字）。

**后果：**
*   **前向传播（Forward）**：数值爆炸（NaN），计算机存不下了
*   **反向传播（Backward）**：梯度也跟着爆炸。你需要把学习率调得极小（比如 $0.00000001$），否则一步更新参数就飞了。
*   **激活函数饱和**：如果用 Sigmoid/Tanh，大数值会落在导数为 0 的区域，梯度消失，网络“死”了。

**统一“度量衡”**
*   **输入数据**：身高（1.5~2.0米）和 体重（40~100公斤）。
*   如果不 Norm：体重数值大，对 Loss 的影响大，模型会拼命学体重，忽略身高。
*   **Norm 之后**：
    *   身高 $\rightarrow$ $[-1, 1]$
    *   体重 $\rightarrow$ $[-1, 1]$
    *   两者**平权**了。模型能公平地对待每一个特征。


**Norm 的作用：**
不管你上一层传过来的是 10 还是 10000，我这一层**先减去均值，再除以标准差**。
*   强行把数据拉回到均值 0，方差 1 的附近（比如 $[-3, 3]$）。
*   **稳定性**：每层的输入分布都稳定在一个统一的范围内（这叫解决 **Internal Covariate Shift** 内部协变量偏移）。
*   **独立性**：后一层不需要为了适应前一层的剧烈变化而频繁调整参数。每一层都可以“安心”地学习自己的特征。
*   **加速收敛**：因为数值范围统一了，我们可以用更大的学习率，步子迈得更大。

# 核心数学原理（通用版）

几乎所有的 Norm 层，核心步骤都只有两步：

1.  **标准化 (Standardization)**：减去均值，除以标准差。【把所有测量值都换算到同一种单位（比如都换成“标准差单位”）】
    $$ \hat{x} = \frac{x - \mu}{\sigma + \epsilon} $$
    *   $\mu$: 均值 (Mean)
    *   $\sigma$: 标准差 (Standard Deviation)
    *   $\epsilon$: 防止分母为 0 的小量

2.  **仿射变换 (Affine Transformation)**：乘上 Scale，加上 Shift。【每层可以再选择自己想用的标尺与零点（但选择是可学习、由数据驱动的）】
    \[
      \text{Norm}(x)=\gamma \odot \hat{x} + \beta
    \]
    * \(\hat{x}\)：归一化后的向量（均值为0、方差为1，或 RMS 为1，取决于 Norm 类型）
    *   $\gamma$ (Gamma): **Scale (缩放因子)**。让模型自己学习这个分布该多宽。
    *   $\beta$ (Beta): **Shift (平移因子)**。让模型自己学习这个分布该在哪里。
    *   这两兄弟是**可学习参数 (Learnable Parameters)**，模型在训练中自动调整它们。如果只做标准化，网络会被强行约束在固定分布，表达力可能下降。  加入 \(\gamma,\beta\) 后，网络可以学会“必要时把分布再拉回去”，Norm 主要负责稳定训练而不牺牲表示能力
    * \(\odot\)：逐元素乘法

# 为什么标准化之后还要进行仿射变换？（Affine Transformation）

如果只做标准化（减均值除方差），数据就被死死地限制在 $N(0, 1)$（标准正态分布）里了。
*   **限制了表达能力**：如果下一层想学一个偏置（比如所有人都偏胖），或者想把特征放大一点（比如体重的微小变化很重要），标准分布就把这种能力抹杀了。
*   **激活函数不开心**：以 ReLU 为例，如果输入全是 $N(0, 1)$，那有一半的数据（小于0的）会被 ReLU 砍掉变成 0。这会丢失大量信息。

**仿射变换 (Scale & Shift) 的作用：**
**“我先把你们都拉回起跑线 (标准化)，然后允许你们根据需要，自由地向前或向后挪一挪 (Shift)，或者把队伍拉长一点/缩短一点 (Scale)。”**

**如果模型发现标准分布最好**，它会把 $\gamma$ 学成 1，$\beta$ 学成 0（还原回去）。
**如果模型发现某个特征很重要**，它会把 $\gamma$ 变大（放大特征）。
**如果模型发现某个特征需要整体偏移**，它会把 $\beta$ 变大/变小。

# 如何理解 Scale (缩放) ？

*   **对象**：标准化后的数据 $\hat{x}$。
*   **操作**：乘法 $\hat{x} \times \gamma$。
*   **几何意义**：
    *   $\gamma > 1$：把数据分布**拉宽**。原本集中在 $[-1, 1]$，现在变成 $[-2, 2]$。增加了数据的对比度。
    *   $\gamma < 1$：把数据分布**压扁**。原本在 $[-1, 1]$，现在变成 $[-0.5, 0.5]$。减小了数据的波动。
*   **为什么要缩放？**
    *   有些特征（比如边缘检测）需要强烈的对比度，模型会把它的 $\gamma$ 变大。
    *   有些特征（比如背景噪声）不重要，模型会把它的 $\gamma$ 变小，抑制它。

# 如何理解 Shift (平移) ？

*   **对象**：缩放后的数据 $\gamma \hat{x}$。
*   **操作**：加法 $+ \beta$。
*   **几何意义**：
    *   把整个数据分布在数轴上**左右移动**。
    *   原本中心在 0，现在中心变成了 $\beta$。
*   **为什么要平移？**
    *   **配合激活函数**。比如 ReLU 激活函数是 $\max(0, x)$。如果 $\beta$ 是负数，数据就会整体左移，更多的数据落入“死区”（变成0）。如果 $\beta$ 是正数，更多的数据会被激活。模型可以通过调整 $\beta$ 来控制**有多少神经元被激活**。


# 为什么 Scale 和 Shift 是可学习参数？如何学习的？

*   **为什么是可学习的？**
    *   因为我们（人类）不知道每一层到底需要什么样的分布。
    *   第一层可能需要宽分布，第十层可能需要窄分布。
    *   所以，干脆把 $\gamma$ 和 $\beta$ 设置成**变量 (Parameter)**，让神经网络自己去定。

*   **如何学习的？**
    *   **反向传播 (Backpropagation)**。
    *   定义 Loss（预测值和真实值的差距）。
    *   计算 Loss 对 $\gamma$ 和 $\beta$ 的梯度（导数）：
    *   **梯度下降 (Gradient Descent)**：
        *   $\gamma_{new} = \gamma_{old} - \text{lr} \times \text{grad}_{\gamma}$
        *   $\beta_{new} = \beta_{old} - \text{lr} \times \text{grad}_{\beta}$
    *   随着训练进行（比如几万次迭代），$\gamma$ 和 $\beta$ 就会自动调整到最适合当前任务的数值。


# 常见 Norm 层

它们的区别主要在于：**在哪个维度上求均值 ($\mu$) 和标准差 ($\sigma$)**。

## BatchNorm (BN) - 时代的眼泪
**计算方式**：把整个 Batch 看作一个整体，计算所有样本在同一个通道上的均值和方差。

**场景**：CNN (ResNet) 常用，但在 Transformer/RNN 中很少用。

**缺点**：依赖 Batch Size。如果 Batch Size 太小（比如 1 或 2），算出来的均值方差不准，模型就崩了。

```python
import torch
import torch.nn as nn

# 假设输入：Batch=2, Channel=3, Height=2, Width=2
# 模拟一个非常简单的特征图
x = torch.tensor([[[[1., 1.], [1., 1.]],  # Sample 1, Channel 1 (全1)
                   [[2., 2.], [2., 2.]],  # Sample 1, Channel 2 (全2)
                   [[3., 3.], [3., 3.]]], # Sample 1, Channel 3 (全3)
                  
                  [[[10., 10.], [10., 10.]], # Sample 2, Channel 1 (全10)
                   [[20., 20.], [20., 20.]], # Sample 2, Channel 2 (全20)
                   [[30., 30.], [30., 30.]]] # Sample 2, Channel 3 (全30)
                 ])

# 初始化 BatchNorm2d，通道数=3
bn = nn.BatchNorm2d(num_features=3)

# 1. 计算均值 (Mean)
# 对每个通道单独计算，跨越 Batch, Height, Width
# Channel 1 Mean = (1*4 + 10*4) / 8 = 5.5
mean = x.mean(dim=(0, 2, 3), keepdim=True) 

# 2. 计算方差 (Var)
# Channel 1 Var = ((1-5.5)^2 * 4 + (10-5.5)^2 * 4) / 8 = 20.25
var = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)

# 3. 归一化 (Normalize)
# x_hat = (x - mean) / sqrt(var + eps)
# Sample 1, Ch 1: (1 - 5.5) / sqrt(20.25) = -4.5 / 4.5 = -1.0
# Sample 2, Ch 1: (10 - 5.5) / sqrt(20.25) = 4.5 / 4.5 = 1.0
x_norm = (x - mean) / torch.sqrt(var + 1e-5)

# 4. 仿射变换 (Affine)
# y = gamma * x_hat + beta
# bn.weight 就是 gamma (初始化为1), bn.bias 就是 beta (初始化为0)
output = x_norm * bn.weight.view(1, 3, 1, 1) + bn.bias.view(1, 3, 1, 1)

print("手动计算 Output:\n", output[0, 0]) # 应该是全 -1
# PyTorch 官方实现
print("PyTorch Output:\n", bn(x)[0, 0])
```


## LayerNorm (LN) - Transformer 的基石

**计算方式**：在 **Channel (特征)** 维度上求均值。**对每个样本单独计算**。对每个样本、每个 token 的通道向量 \(x \in \mathbb{R}^{C}\)：

\[
\mu = \frac{1}{C}\sum_{i=1}^{C} x_i,\quad
\sigma^2 = \frac{1}{C}\sum_{i=1}^{C}(x_i-\mu)^2
\]

\[
\hat{x}_i = \frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}
\]

\[
\text{LN}(x)=\gamma \odot \hat{x} + \beta
\]

**优点**
- 不依赖 batch 维度大小（小 batch 也稳定）
- 适合序列建模（每个 token 独立归一化）

**在 Transformer 的位置（Pre-LN vs Post-LN）**
- **Pre-LN**：`x = x + Attn(LN(x))`（现代 Transformer 更常用，训练更稳定）
- **Post-LN**：`x = LN(x + Attn(x))`（原始 Transformer，深层会更难训）

```python
import torch
import torch.nn as nn

# 假设输入：Batch=2, Sequence=2, Embedding=4
x = torch.randn(2, 2, 4)

# LayerNorm，指定归一化的维度是最后一维 (Embedding Dim)
ln = nn.LayerNorm(normalized_shape=4)

# 1. 计算均值 (Mean)
# 对每个样本 (Batch) 的每个 Token (Sequence) 单独计算
# mean.shape = [2, 2, 1]
mean = x.mean(dim=-1, keepdim=True)

# 2. 计算方差 (Var)
var = x.var(dim=-1, keepdim=True, unbiased=False)

# 3. 归一化
x_norm = (x - mean) / torch.sqrt(var + 1e-5)

# 4. 仿射变换
# ln.weight (gamma), ln.bias (beta) 形状是 [4]
output = x_norm * ln.weight + ln.bias

print("手动计算 Output Shape:", output.shape)
print("PyTorch Output Shape:", ln(x).shape)
```


## GroupNorm (GN) - CNN/UNet/视频 VAE 常见

**计算方式**：把 Channel 分成 $G$ 个组，在每个 **Group** 在空间维（H,W）和组内通道上统计：

\[
\mu_{g} = \frac{1}{m}\sum x,\quad
\sigma_g^2=\frac{1}{m}\sum (x-\mu_g)^2
\]
这里 \(m=(C/G)\cdot H\cdot W\)（3D则再乘 T）。然后同样 \(\gamma,\beta\) 做仿射。

**优点**
- 不依赖 batch，Batch Size 小的时候比 BN 稳
- 比 LN 更适合卷积特征（考虑空间结构）

**场景**：Detection, Segmentation, Stable Diffusion VAE。Stable Diffusion 的 UNet 大量用 GN。

```python
import torch
import torch.nn as nn

# 输入：Batch=2, Channel=6, H=2, W=2
x = torch.randn(2, 6, 2, 2)

# 分成 3 组 (每组 2 个 Channel)
gn = nn.GroupNorm(num_groups=3, num_channels=6)

# 1. Reshape 分组
# [B, C, H, W] -> [B, G, C//G, H, W]
# [2, 6, 2, 2] -> [2, 3, 2, 2, 2]
x_reshaped = x.view(2, 3, 2, 2, 2)

# 2. 计算均值
# 在 Group 内的所有像素 (C//G, H, W) 上求均值
# mean.shape = [2, 3, 1, 1, 1]
mean = x_reshaped.mean(dim=(2, 3, 4), keepdim=True)

# 3. 计算方差
var = x_reshaped.var(dim=(2, 3, 4), keepdim=True, unbiased=False)

# 4. 归一化
x_norm = (x_reshaped - mean) / torch.sqrt(var + 1e-5)

# 5. 还原形状
# [2, 3, 2, 2, 2] -> [2, 6, 2, 2]
x_norm = x_norm.view(2, 6, 2, 2)

# 6. 仿射变换 (每个 Channel 有独立的 gamma, beta)
output = x_norm * gn.weight.view(1, 6, 1, 1) + gn.bias.view(1, 6, 1, 1)
```

## RMSNorm (Root Mean Square Norm) - LN 的“去均值简化版”，大模型的首选

**计算方式**：**去掉了均值 $\mu$ 的计算，只除以均方根 (RMS)** 来缩放。

\[
\text{rms}(x)=\sqrt{\frac{1}{C}\sum_{i=1}^C x_i^2 + \epsilon}
\]

\[
\hat{x} = \frac{x}{\text{rms}(x)}
\]

\[
\text{RMSNorm}(x)=\gamma \odot \hat{x}\quad (\text{有时不加 }\beta)
\]

**为什么 RMSNorm 越来越常见？**
- 计算速度更快（少了减均值）
- 数值稳定性很好
- 在大规模 Transformer（LLM）里表现非常好（LLaMA 等大量用 RMSNorm）

```python
# PyTorch 没有原生 RMSNorm，通常手写
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        # 只有 Scale (weight)，没有 Shift (bias)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # 1. 计算均方根 (RMS)
        # mean(x^2) -> sqrt
        # 不需要减去 x.mean() !
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        
        # 2. 归一化
        x_norm = x * rms
        
        # 3. 缩放
        return x_norm * self.weight

# 测试
x = torch.randn(2, 10)
rms_norm = RMSNorm(10)
output = rms_norm(x)
print("RMSNorm Output:", output.shape)
```

# AdaLN（Adaptive LayerNorm）：用条件动态生成 Scale/Shift

在 Diffusion Model (如 DiT, Wan2.1) 中，我们不仅要归一化，还要**把时间步 (t) 和 文本条件 (c) 注入到网络里**。
* timestep embedding（最常见）
* 文本/图像条件的聚合向量
* 其他控制信号的 embedding


##  基本形式

给定 token 特征 \(x\)（比如 LN 前的输入），以及条件向量 \(c\)（比如 timestep embedding、文本 embedding 的聚合）：

\[
y = \text{LN}(x)
\]

Scale (γ) 和 Shift (β) 不是固定的参数，而是通过一个 神经网络（MLP），根据条件c预测出来

\[
(\Delta\gamma(c),\Delta\beta(c)) = f(c)
\]

用这两个预测值去调节当前的 Norm 层。用 \(1+\Delta\gamma\) 是为了让初始更接近不改变尺度（常见技巧）

\[
\text{AdaLN}(x,c)= y \odot (1+\Delta\gamma(c)) + \Delta\beta(c)
\]

**作用**：让整个网络“感知”到当前是第几步，要生成什么内容。比如 $t=0$ (高噪) 时 $\gamma$ 可能很大，$t=1000$ (低噪) 时 $\gamma$ 可能很小。


```python
# 代码示例 (模拟 DiT Block 中的 AdaLN)
import torch
import torch.nn as nn

class AdaLNBlock(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        # 标准 LayerNorm (不带 weight/bias，因为要动态预测)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        
        # 预测器 (MLP): 输入条件 -> 预测 gamma, beta
        # 这里的 output_dim = dim * 2 (一个给 gamma, 一个给 beta)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, dim * 2)
        )
        
        # AdaLN-Zero 初始化技巧：
        # 把最后一层 Linear 的权重和偏置初始化为 0
        # 这样初始 gamma=0, beta=0 -> identity mapping
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, condition):
        # x: [Batch, Seq, Dim]
        # condition: [Batch, Cond_Dim] (比如时间步 embedding)
        
        # 1. 预测 Scale (gamma) 和 Shift (beta)
        # shift, scale shape: [Batch, Dim]
        shift, scale = self.adaLN_modulation(condition).chunk(2, dim=1)
        
        # 2. 执行 Norm
        x_norm = self.norm(x)
        
        # 3. 应用动态 Scale 和 Shift
        # 注意维度广播: [B, S, D] * [B, 1, D] + [B, 1, D]
        # (1 + scale) 是为了让初始状态接近 Identity (当 scale=0 时)
        x_out = x_norm * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        
        return x_out

# 测试
dim = 128
cond_dim = 256
block = AdaLNBlock(dim, cond_dim)

x = torch.randn(2, 10, dim)       # Video Tokens
t = torch.randn(2, cond_dim)      # Time Embedding

out = block(x, t)
print("AdaLN Output:", out.shape)
# 初始输出应该接近 0 (因为 AdaLN-Zero 初始化)
print("Initial Mean:", out.mean().item()) 
```


## 在 DiT 里为什么常用 AdaLN？

扩散模型最核心的条件就是 **timestep t**。  
DiT（例如原始 DiT、PixArt、很多视频 DiT）普遍用 AdaLN 让 t 影响每层，从而控制去噪行为。

## DiT架构的AdaLN 如何注入时间步 $t$ ？

1. 数据
   *   **输入 $x$ (Image Tokens)**: $[B, N, D]$。这是图像本身的内容。
   *   **条件 $t$ (Timestep)**: 一个标量（例如 $t=500$），代表当前去噪进行到了哪一步。
        *   $t=1000$: 纯噪声，工人需要“大刀阔斧”地改。
        *   $t=0$: 接近成图，工人需要“小心翼翼”地修。

2. 时间步编码 (Time Embedding)：$t$ 首先通过一个 **MLP (多层感知机)** 变成一个向量 $t_{emb} \in \mathbb{R}^D$。

3. AdaLN 的预测 (Scale & Shift Prediction)：在每个 Block 内部，AdaLN 模块接收 $t_{emb}$，并通过一个线性层 (Linear Layer) **预测出 6 个参数**
$$ (\gamma_1, \beta_1, \alpha_1, \gamma_2, \beta_2, \alpha_2) = \text{Linear}(t_{emb}) $$
   *   $\gamma_1, \beta_1$: 控制 Self-Attention 前的 Norm。
   *   $\alpha_1$: 控制 Self-Attention 后的残差门控 (Gate)。
   *   $\gamma_2, \beta_2$: 控制 FFN 前的 Norm。
   *   $\alpha_2$: 控制 FFN 后的残差门控。

4. 调制 (Modulation) ：AdaLN 对图像 Token $x$ 进行标准化后，应用这些参数：
$$ \text{AdaLN}(x, t) = x_{norm} \cdot (1 + \gamma_1(t)) + \beta_1(t) $$

   *   **如果 $t$ 很大 (噪声多)**: 模型可能会预测出一个**很大的 $\gamma$**（放大特征方差）和**特定的 $\beta$**（偏移均值），告诉 Attention 层：“现在信号很弱，把数值拉大一点，重点关注轮廓！”
   *   **如果 $t$ 很小 (噪声少)**: 模型可能会预测出一个**接近 0 的 $\gamma$**（保持原样）和**微小的 $\beta$**，告诉 Attention 层：“现在细节差不多了，别乱动，微调一下就行。”
5. 影响后续流程：经过 AdaLN 调制后的 $x$，其**数值分布（均值和方差）携带了时间信息**。当它进入 Self-Attention 计算 $Q, K, V$ 时，产生的 Attention Map 就会受到这个分布的影响。
   *   **结果**：不同时间步下，模型关注的图像区域完全不同（初期关注结构，后期关注纹理）。


## AdaLN-Zero（扩散/DiT里超级常见）

在 DiT 的 Residual Block 最后一层，**让条件注入分支在初始化时为 0**，希望训练**从 0 开始**。因为刚开始训练时，我们希望这个 Block **什么都不做**（Identity Mapping），使模型初始行为接近“无条件/恒等”，训练更稳定

典型形式：
\[
x \leftarrow x + g(c)\cdot F(\text{AdaLN}(x,c))
\]
其中 \(g(c)\) 初始为 0（或非常小），训练中逐渐学会打开“门”。

**做法**：AdaLN 预测出来的 $\gamma$ 和 $\beta$，以及一个控制残差权重的门控参数 $\alpha$ (Gate)，**全部初始化为 0**。

**效果**：
  *   初始状态：$y = 0 \cdot \text{Norm}(x) + 0 = 0$。
  *   残差连接：$x_{out} = x_{in} + 0 = x_{in}$。
  *   这使得训练非常稳定，允许模型堆得非常深


```python

# Wan 的 WanAttentionBlock 用 e 分成 6 份，其中两份作为 gate 乘在 self-attn/ffn 输出上再残差相加（属于 AdaLN-Zero 风格的“门控残差”思想）。
# VACE 的 before_proj/after_proj 零初始化也体现“zero injection”思想。

class AdaLNZeroBlock(nn.Module):
    def __init__(self, dim, cond_dim, mlp_ratio=4):
        super().__init__()
        self.ln = nn.LayerNorm(dim, elementwise_affine=False)
        self.to_scale_shift_gate = nn.Linear(cond_dim, 3 * dim)

        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

        # 关键：最后一层或 gate 初始化为 0（这里把线性层权重置0）
        nn.init.zeros_(self.to_scale_shift_gate.weight)
        nn.init.zeros_(self.to_scale_shift_gate.bias)

    def forward(self, x, cond):
        """
        x: [B,L,C]
        cond: [B,cond_dim]
        """
        h = self.ln(x)
        scale, shift, gate = self.to_scale_shift_gate(cond).chunk(3, dim=-1)  # [B,C]*3
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
        gate = gate.unsqueeze(1)

        h = h * (1 + scale) + shift
        y = self.ffn(h)
        out = x + gate * y
        return out

block = AdaLNZeroBlock(dim=C, cond_dim=cond_dim)
y = block(x_seq, cond)
print("y:", y.shape)

# 因为 gate 初始为0，y 会非常接近 x_seq（数值上几乎不变）
print("mean abs diff:", (y - x_seq).abs().mean().item())
```

# 总结对照表

| 名称 | 统计维度 | 是否依赖batch | 是否减均值 | 常见场景 | 主要目的 |
|---|---|---:|---:|---|---|
| BatchNorm | 跨 batch | 是 | 是 | CNN（大batch） | 加速收敛、稳定 |
| LayerNorm | 每 token 的 C 维 | 否 | 是 | Transformer/DiT | 序列稳定训练 |
| GroupNorm | 每样本按通道分组 + 空间 | 否 | 是 | UNet/VAE/CNN | 小batch稳定 |
| RMSNorm | 每 token 的 C 维 | 否 | 否 | LLM/Transformer/注意力QK | 更省、更稳 |
| AdaLN | LN + 条件生成 scale/shift | 否 | 是 | 扩散DiT | 注入条件（t/prompt等） |
| AdaLN-Zero | AdaLN + 残差分支零初始化/门控 | 否 | 是 | 扩散DiT、adapter | 稳定训练、可控注入 |


