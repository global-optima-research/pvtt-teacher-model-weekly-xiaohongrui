# PVTT  - Task 2: Teacher Model Training

本仓库记录工作计划、技术调研、代码进展及周报。Task 2 的核心目标是基于 Wan2.1-14B 训练一个高质量的 Teacher Model，实现**基于模版的电商视频换品生成**。

---

## 📅 Weekly Schedule & Progress

### Week 1: 熟悉项目背景及已有研究进展

#### 🔬 核心技术调研类 (Core Technical Research)
- [x] 阅读 Wan2.1 技术报告 [Open and Advanced Large-Scale Video Generative Models](https://arxiv.org/abs/2503.20314)
- [x] 阅读 VACE 论文 [All-in-One Video Creation and Editing](https://arxiv.org/abs/2503.07598)


#### 💻 代码工程类 (Engineering)
- [x] 在 SuperPod 中配置 Wan2.1 开发环境，跑通 Wan2.1 T2V 的一次推理过程 | [Log: Wan22在SuperPod中的环境配置](./week01_2026-02-16_to_2026-02-22/Env_config_of_Wan22_in_SuperPod.md)

#### 🧠 基础知识补全类 (Knowledge Supplement)
- [x] 阅读 global-optima-research 各个 repo
- [x] 阅读 Diffusion Transformer 论文 [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)

---

### Week 2: 深入学习 Wan 和 VACE 模型

#### 🔬 核心技术调研类
- [x] 结合Wan2.1技术报告及源码，深入学习 Wan 模型 | [Note: Wan21调研-v1](./week02_2026-02-23_to_2026-03-01/wan21-survey-v1.md)
- [x] 结合VACE论文及源码，深入学习 VACE 模型 | [Note: VACE调研-v1](./week02_2026-02-23_to_2026-03-01/vace-survey-v1.md)

#### 💻 代码工程类
- [x] 在 5090 Server中配置好 DiffSynth Studio 环境
- [x] 使用 DiffSynth Studio 跑通一次 Wan2.1-T2V-1.3B 的 LoRA微调全流程 | [Log: Diffsynth环境搭建及lora流程验证](./week02_2026-02-23_to_2026-03-01/Diffsynth_env_setup_and_LoRa_verification.md)
- [x] VACE 模型能力边界探究 | [Log: 实验记录](./week02_2026-02-23_to_2026-03-01/vace_experiment.md)

#### 🧠 基础知识补全类
- [x] 阅读 Flow matching 论文 [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [ ] 阅读 Rectified Flow 论文 [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003)
- [x] 学习 Normalization 技术 (LayerNorm, RMSNorm, AdaLN, AdaLN-Zero) | [Note: Normalization技术](./week02_2026-02-23_to_2026-03-01/normalization-notes.md)

---

### Week 3: 深入探索 VACE 模型能力边界

#### 🔬 核心技术调研类
- [x] 阅读 Wan-Animate 论文 [Wan-Animate: Unified Character Animation and Replacement with Holistic Replication](https://arxiv.org/abs/2509.14055)
- [x] 结合源码，继续深入学习 wan 系列模型  | [Note: Wan系列模型深度调研文档-v1](./week03_2026-03-02_to_2026-03-08/Wan系列模型深度调研文档.md)
- [x] 阅读“图像条件注入”的相关重要论文
  - [x] 阅读 ControlNet 论文 [Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543)
  - [x] 阅读 T2I-Adapter 论文 [T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models](arxiv.org/abs/2302.08453)
  - [x] 阅读 IP-Adapter 论文 [IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models](https://arxiv.org/abs/2308.06721)
  - [x] 阅读 Uni-ControlNet 论文 [Uni-ControlNet: All-in-One Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2305.16322)
  - [x] 阅读 ControlNet++ 论文 [ControlNet++: Improving Conditional Controls with Efficient Consistency Feedback](https://arxiv.org/abs/2404.07987)
  - [x] 阅读 Ctrl-Adapter 论文[Ctrl-Adapter: An Efficient and Versatile Framework for Adapting Diverse Controls to Any Diffusion Model](https://arxiv.org/abs/2404.09967)

- [ ] 整理大模型微调技术调研路线
- [ ] 整理保持“主体一致性”技术调研路线

#### 💻 代码工程类
- [x] 继续探究 VACE 模型能力边界 | [Code & Log: 实验代码及实验记录](./week03_2026-03-02_to_2026-03-08/val_vace_superpod/experiment_report.md)



### Week 4: 在 pvtt 数据集上验证 VACE 模型能力边界

#### 🔬 核心技术调研类
- [x] 阅读 First Frame Is the Place to Go for Video Content Customization 论文 [First Frame Is the Place to Go for Video Content Customization](https://arxiv.org/abs/2511.15700)  | [Note: FFGO 论文调研文档](./week04_2026-03-09_to_2026-03-15/ffgo_research_report.md)




#### 💻 代码工程类
- [x] 在 pvtt 数据集上验证 VACE 模型能力边界 | [Code & Log: 实验代码及实验记录](./week03_2026-03-02_to_2026-03-08/val_vace_superpod/pvtt_raw_output_experiment_report.md)
- [x] 在 pvtt 数据集上测试 VACE 直接应用 FFGO 思路的效果  | [Code & Log: 实验代码及实验记录](./week03_2026-03-02_to_2026-03-08/val_vace_superpod/pvtt_ffgo_i2v_experiment_report.md)


### Week 5

#### 🔬 核心技术调研类



#### 💻 代码工程类

- [x] 在 pvtt 数据集上测试 First Frame Is the Place to Go for Video Content Customization 论文的效果
  * [Code & Log: 实验代码及实验记录](./week03_2026-03-02_to_2026-03-08/val_vace_superpod/pvtt_ffgo_original_experiment_report.md)
  * [FFGO 转场帧可视化](./week05-2026-3-16_to2026-3-22/pvtt_ffgo_transition_frames.md)
  * [优化/创新方向分析](./week05-2026-3-16_to2026-3-22/pvtt_optimization_research_cn.md) 
- [x] 训练数据集构建计划 [用于 Wan2.2 TI2V 5B + FFGO LoRA 微调的训练数据集](./week05-2026-3-16_to2026-3-22/dataset_construction_plan.md)

