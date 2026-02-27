# VACE 想解决什么：一个模型统一“生成 + 编辑 + 组合任务”

论文摘要与图1（第1页）给出了 VACE 覆盖的任务族：  
- **T2V**：文本生成  
- **R2V**：参考图驱动生成（Reference-to-Video）  
- **V2V**：整段视频风格/结构/控制信号编辑（Depth/Gray/Pose/Flow/Layout…）  
- **MV2V**：带 mask 的局部视频编辑（inpaint/outpaint/extension…）  
- **Task Composition**：把上述能力自由组合（图4第7页示例：Move/Animate/Swap/Expand Anything）

关键挑战（论文1-3页）：视频任务比图像任务更难统一，因为必须同时保证 **时序一致性** 和 **空间一致性**，且输入模态非常多（text/image/video/mask）。

VACE 的核心贡献（论文摘要、3章）是两件事：
1) 设计统一输入接口 **VCU**（把各种任务都写成“文本 + 帧序列 + mask序列”）  
2) 用 **Context Adapter** 的结构把这些条件“插件式”注入到视频 DiT 中（支持冻结主干、只训练 adapter，收敛快、可插拔）

# VCU（Video Condition Unit）：统一多任务输入的“协议层”

## 论文定义：
\[
\mathbf{V} = [T; F; M] \tag{1}
\]
- \(T\)：文本 prompt  
- \(F=\{u_1,\dots,u_n\}\)：上下文帧序列（RGB，归一化到[-1,1]）  
- \(M=\{m_1,\dots,m_n\}\)：mask序列（二值；1表示“要编辑/要重新生成”的区域，0表示“保留不变”的区域）  
并强调 **F 和 M 在时间与空间上对齐**（同 \(n, h, w\)）

表1（论文第4页）给出了四种基本任务如何实例化 VCU：

- **T2V**：  
  - \(F = \{0_{h\times w}\}\times n\)（全0帧占位）  
  - \(M = \{1_{h\times w}\}\times n\)（全1表示全部生成）

- **R2V**：  
  - \(F = \{r_1,\dots,r_l\} + \{0\}\times n\)（把参考图插到帧序列前面）  
  - \(M = \{0\}\times l + \{1\}\times n\)（参考图对应mask为0：表示“保留”，不需要重新生成）

- **V2V**：  
  - \(F=\{u_1,\dots,u_n\}\)，\(M=\{1\}\times n\)（整段作为条件，整段可重绘/重渲染）

- **MV2V**：  
  - \(F=\{u_1,\dots,u_n\}\)，\(M=\{m_1,\dots,m_n\}\)（局部区域编辑）

## 对照源码：VCU 在 `vace.py` 里如何落地？
`WanVace.generate()` 的输入就是三件：
- `input_prompt`
- `input_frames`（对应 F）
- `input_masks`（对应 M）
- `input_ref_images`（对应表1里插到 F 前面的 refs）

其中“把 ref 插到前面”的实现，正是 `vace_encode_frames()` 做的：  
```python
latent = torch.cat([*ref_latent, latent], dim=1)  # dim=1 是时间维
```
mask 也对应补零（ref部分mask=0），在 `vace_encode_masks()` 里：
```python
mask_pad = torch.zeros_like(mask[:, :length, :, :])
mask = torch.cat((mask_pad, mask), dim=1)
```
这与论文表1的 R2V 定义完全一致。


# VACE 的整体架构：两路分支 + hints 注入（论文图3；源码 `vace_model.py`）

论文图3（第4页）对比了两种训练/结构策略：
- (a) Fully Finetune：把 context token 与 noisy video token 一起输入主干 DiT，整体微调
- (b) Context Adapter Tuning：新增一条旁路“Context Blocks（Context Adapter）”，只处理 context tokens，然后把输出以 additive signal 加回主干若干层；主干冻结，只训练 adapter + embedder（收敛快、可插拔）

## 代码里 Context Adapter 对应什么？
在 `wan/modules/vace_model.py`：
- `self.vace_blocks`: 一串 **VaceWanAttentionBlock**（相当于论文图3b里的 Context Blocks）
- `self.blocks`: 主干 blocks 被替换成 **BaseWanAttentionBlock**，它支持注入 hints
- `forward_vace()`：跑 vace_blocks 生成 `hints`
- `forward()`：跑主干 blocks，每个注入层执行：
  ```python
  x = x + hints[self.block_id] * context_scale
  ```

对应论文 3.3.2（第5页）所说：
> “Context tokens pass Context Blocks and added back to the original DiT Blocks … output inserted back as addictive signal … main DiT frozen.”

这就是“Context Adapter”。

## 为什么这种设计好？
结合论文 3.3.2 + 5.3 ablation（第8页）：
- 全量微调可行但成本高、收敛慢、难作为插件  
- adapter 方式：
  - **更快收敛**（论文图5a）  
  - **可控的参数规模**：只训练 embedder + context blocks  
  - **可插拔**：不同能力可以通过不同 adapter 组合（论文强调 pluggable）

源码也体现了“尽量不破坏基模”的思想：  
`VaceWanAttentionBlock.before_proj/after_proj` 都 **zero init**，初始 hints≈0，模型行为接近原 WanT2V，只在训练后逐渐学会注入编辑能力（这点对稳定训练非常关键）。

# VCU 如何变成模型吃得下的条件：Concept Decoupling + Context Latent Encode + Context Embedder
论文 3.3.1（第4-5页）给出 Context Tokenization 三步：

## Concept Decoupling（概念解耦）

### 论文定义（第4页）：
\[
F_c = F \times M,\quad F_k = F \times (1-M)
\]
- \(F_c\) reactive：要被修改/要重生成的部分（mask=1保留，mask=0置0）  
- \(F_k\) inactive：要保留不变的部分（mask=0保留，mask=1置0）  
并说明：reference images 和未修改区域进入 \(F_k\)，控制信号/要变化区域进入 \(F_c\)。

#### 对照源码：`vace.py:vace_encode_frames()`
当 `masks is not None`：
```python
inactive = i*(1-m) + 0*m   # Fk
reactive = i*m + 0*(1-m)   # Fc
inactive = vae.encode(inactive)  # 16ch
reactive = vae.encode(reactive)  # 16ch
latent = cat((inactive, reactive), dim=0)  # 通道维拼接 -> 32ch
```
这正是论文的 \(F_k/F_c\) 解耦，只是 **在 latent 空间**表达，并把两者 **沿通道维 concat**（这点论文也提到“reorganize together … hierarchical aligned visual features”）。

## Context Latent Encoding（上下文 latent 编码）

### 论文定义

论文第4页说：DiT 操作的是 noisy latent \(X\in \mathbb{R}^{n'\times h'\times w'\times d}\)，因此 \(F_c,F_k,M\) 也要编码到同一个 latent 空间以保持时空相关性：
- \(F_c,F_k\)：通过 video VAE 编码到与 X 同latent空间  
- reference images：单独编码并沿时间维 concat（避免 image/video mishmash）
- mask \(M\)：直接 reshape + interpolate 到 latent 对齐尺度

### 对照源码：
- \(F_c,F_k\) 的 VAE 编码：如上 `vae.encode(inactive/reactive)`
- reference images 的“单独编码+沿时间维拼”：
  ```python
  ref_latent = vae.encode(refs)          # 每张 ref -> [16,1,H_lat,W_lat]
  latent = torch.cat([*ref_latent, latent], dim=1)  # 时间维拼接
  ```
- mask 的 reshape+interpolate：`vace_encode_masks()`  
  它不是走 VAE，而是把 mask 按 stride_h×stride_w（通常8×8）**展开成 64 个通道**，再对时间维插值到 new_depth，并在 ref 存在时在时间维前补零。

> 这一点非常“VACE味”：mask 既要对齐 latent 的时空分辨率，又要尽量保留 stride block 内的细粒度结构，因此用 64 通道保留 8×8 子格信息。论文只说 reshape+interpolate，源码给出了具体工程实现。

## Context Embedder（上下文嵌入器）
论文第5页说：
> “concatenating \(F_c,F_k,M\) in the channel dimension and tokenizing them into context tokens … weights for \(F_c,F_k\) copied from original embedder, weights for \(M\) init zeros.”

在源码里对应：
- `VaceWanModel.vace_patch_embedding = Conv3d(vace_in_dim -> dim, kernel=stride=patch_size)`
- `vace_in_dim` 就是 \(F_c,F_k,M\) concat 后的通道数（需要与 vace_context 的通道匹配）

注意：在 `vace_model.py` 中，`vace_patch_embedding` 的权重初始化没有显式“copy base embedder + mask zero init”的逻辑（可能在 checkpoint 构建时已完成，或在训练脚本里做了权重拷贝）。但结构上是吻合的：mask 通道可以通过初始化策略不影响基模。

# 模拟 VACE 的一次工作流程（含维度追踪）
选择与 Wan/VACE 默认一致的常见设置：
- 输出视频：**F=81** 帧（4n+1）
- 分辨率：**720p (H=720, W=1280)**
- VAE stride：**(4,8,8)**（论文/Wan2.1）
- patch_size：**(1,2,2)**（WanModel默认）
- latent通道：**16**
- token seq_len（720p/81帧）：**75600**（你在 `WanVace.__init__` 里看到固定值）

输入 VCU（举一个 MV2V + 多ref 组合任务）：
- prompt：T
- source video \(F\)：`input_frames[0]` shape `[3,81,720,1280]`
- mask \(M\)：`input_masks[0]` shape `[3,81,720,1280]`（或1通道也行，但代码里常用3通道）
- ref images：2张，每张 `[3,1,720,1280]`

## Step 1：Concept Decoupling（像素域）
- `inactive_video = video*(1-mask)`：保留未编辑区
- `reactive_video = video*mask`：保留需编辑区

## Step 2：VAE encode 到 latent（时空压缩 4×8×8）
对 81 帧：
- latent 时间长度：
  \[
  T_{lat} = 1 + \frac{81-1}{4}=21
  \]
空间：
- \(H_{lat}=720/8=90,\ W_{lat}=1280/8=160\)

因此：
- `inactive_latent`: `[16,21,90,160]`
- `reactive_latent`: `[16,21,90,160]`
- concat 通道得：
  - `z0_source = [32,21,90,160]`（32=Fk16+Fc16）

ref images（2张）：
- 每张 ref 编码：`[16,1,90,160]`
- 由于 mask 存在，源码把 ref latent 也扩成 32 通道：`[32,1,90,160]`（后16为0）
- 沿时间维拼到最前面：
  - `z0 = [32, (2 + 21)=23, 90, 160]`

## Step 3：mask 编码到 latent对齐尺度（`vace_encode_masks`）
- 把 mask 的 8×8 block 展开成 64通道：
  - `mask_rearranged`: `[64, depth, height', width']`
- 时间插值到 `new_depth = (81+3)//4 = 21`
- ref 存在时在时间维前补 2 个0：
  - `m0`: `[64, 23, 90, 160]`（这里空间裁剪/对齐逻辑略复杂，但目标是与 z0 对齐）

## Step 4：VCU latent 合并（Context Latent Encode输出）
- `vace_context = cat(z0, m0) along channel`
- 通道数：`C_vace = 32 + 64 = 96`
- `vace_context`: `[96, 23, 90, 160]`

注意：这要求 `VaceWanModel.vace_in_dim == 96`，否则 vace_patch_embedding 输入通道不匹配。你们训练/权重里应当已经设好了。

## Step 5：扩散变量（要采样的 latents）形状
`vace.py` 里这样构造噪声：
```python
target_shape = list(z0[0].shape)   # [32,23,90,160]
target_shape[0] = int(target_shape[0]/2)  # -> 16
noise = [randn(16,23,90,160)]
```
也就是扩散采样的未知变量 `x` 是：
- `x`: `[16, 23, 90, 160]`

直觉：在编辑场景里，扩散变量是“要生成/要修改的那部分 latent”（16通道），而 source/ref/mask 作为条件走 vace_context 分支。

## Step 6：主干 patchify（noisy video tokens）
对 `x`（加batch后 `[1,16,23,90,160]`）用 patch_size(1,2,2)：
- `Fp = 23`
- `Hp = 90/2=45`
- `Wp = 160/2=80`
- token length：
  \[
  L = 23 \cdot 45 \cdot 80 = 82{,}800
  \]
并 pad 到 `seq_len`（这里 `seq_len` 会在 `vace.py` 根据 target_shape 算出来，不一定固定75600；但在720p/81帧且无ref时是75600，有ref时会变长，所以 vace.py 动态算 seq_len 是合理的）

token 表示：
- `x_tokens`: `[B=1, seq_len, dim]`

## Step 7：vace_context patchify（context tokens）
`forward_vace()`：
- `c = vace_patch_embedding(vace_context)` 得到 `[1, dim, Fp,Hp,Wp]`
- flatten -> `[1, L_ctx, dim]`  
理论上 L_ctx 应与 x 的 token 网格一致（因为 vace_context 与 x 在时空上对齐），因此 L_ctx≈L
- pad 到同一个 `seq_len`

## Step 8：Context Adapter（vace_blocks）生成 hints
`vace_blocks` 数量 = `len(vace_layers)`（默认每隔一层注入，即16个注入点）
- 第一个 vace_block（block_id=0）做：
  - `c = before_proj(c) + x`（零初始化使初始 c≈x）
- 每个 vace_block 输出：
  - `c_skip = after_proj(c)`（初始≈0）
- 收集 `hints = [hint0, hint1, ...]`，每个 `[1, seq_len, dim]`

## Step 9：主干 blocks 注入 hints
`BaseWanAttentionBlock.forward()`：
- 跑标准 WanAttentionBlock（self-attn + cross-attn(text) + ffn）
- 若当前层在 vace_layers：
  \[
  x \leftarrow x + context\_scale \cdot hints[k]
  \]
这就是论文图3b 的“additive signal insertion”。

## Step 10：head + unpatchify 输出 latent 预测
最终输出：
- `pred_latent`: `[16, 23, 90, 160]`

然后 scheduler 用 flow matching solver 更新 `x`，循环迭代。

## Step 11：decode 时裁掉 ref 时间步
因为 ref 被拼到了时间维前面（23=2+21），最终输出视频不应含 ref “伪帧”：
- `decode_latent()` 里执行：
  - `z = z[:, len(refs):, :, :]` → `[16, 21, 90, 160]`
- VAE decode 还原到像素：`[3,81,720,1280]`


# VACE 与 Wan2.1：区别与联系

## 联系：VACE 是建立在 Wan2.1 的“基础生成引擎”上

从源码看，VACE 复用了 Wan2.1 的核心三件套：
1) **视频 VAE**（`WanVAE`）：同样的 latent 空间、时空压缩比  
2) **DiT 基础块结构**（`WanAttentionBlock` / time modulation / RoPE / flash attention）  
3) **Flow Matching 采样框架**（UniPC / DPM++；scheduler.step 逻辑一致）

因此你可以说：**VACE 没有重新发明生成模型，而是把 Wan 的生成模型“输入协议 + 条件注入方式”扩展成统一编辑框架。**

## 区别 1：输入范式从“单任务接口”变成 VCU
- Wan2.1 的任务入口通常是 T2V / I2V / FLF2V，各自的输入形式不同  
- VACE 强制统一成 `[T; F; M]`，并通过把 ref 插在 F 前面、mask 同步插 0 来支持任务组合（论文表1）

源码对应：
- Wan I2V：图像条件走 CLIP + y(mask+首帧latent) 拼通道（`image2video.py`）
- VACE：所有条件走 vace_context 分支（`vace.py + vace_model.py`）

## 区别 2：条件注入方式从“cross-attn/通道拼接”升级为“Context Adapter（旁路注入）”
Wan2.1（I2V）主要两种条件注入：
- 文本：cross-attn
- 图像：CLIP tokens + cross-attn（`WanI2VCrossAttention`），以及 y 拼通道（首帧latent + mask通道）

VACE 则使用：
- 文本：仍是 cross-attn（注意 `VaceWanModel` 里只用 t2v_cross_attn）
- 编辑条件（视频/控制信号/参考/掩码）：通过 vace_context → vace_blocks → hints 注入主干多层

这就是论文强调的 Context Adapter，优势是：
- 不需要全量改主干输入维度（不像“直接 concat 输入通道”那样要改 patch embedding in_dim 并全量微调）
- 可冻结主干，训练少量模块即可适配多任务
- 插件式：不同 task/能力可以作为不同 adapter 组合（论文强调“pluggable features”）

## 区别 3：显式的 Concept Decoupling（reactive/inactive）
Wan2.1 的 I2V 主要是“首帧条件”，并没有通用的“保留/修改”显式分解。  
VACE 则把 F 与 M 做 \(F_c/F_k\) 解耦，并在 latent 通道里显式存储两类信息（源码 z0 变成 32ch），这对编辑任务非常关键：
- “要改哪里” 与 “要保留哪里”在表示层面就被分开  
- 提升训练收敛与稳定性（论文图5d也证明 concept decouple 降 loss）

