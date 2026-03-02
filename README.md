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

