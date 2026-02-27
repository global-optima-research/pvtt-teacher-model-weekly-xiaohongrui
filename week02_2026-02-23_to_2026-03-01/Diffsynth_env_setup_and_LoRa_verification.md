# 配置环境

```bash
conda create -n diffsynth python=3.10 -y
conda activate diffsynth
# RTX 5090 显卡需 cuda 安装 12.8 以上版本。2.7版本的较为稳定
pip uninstall torch torchvision torchaudio -y
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128


git clone https://github.com/modelscope/DiffSynth-Studio.git  
cd DiffSynth-Studio
pip install -e .

python -c "import diffsynth, torch; print('diffsynth ok'); print('torch', torch.__version__); print('cuda available', torch.cuda.is_available())"
```

# 核心目录结构注释

对于**微调 Wan 模型**，你只需要关注 **20%** 的文件。其他的都是给别的模型（Flux, Qwen, LTX）用的，或者底层的数学实现。

```bash
F:.
│  .gitignore
│  LICENSE
│  pyproject.toml        # <-- 项目依赖配置文件，pip install . 就是读这里
│  README.md             # <-- 说明文档
│
├─diffsynth              # 【核心引擎库】这里的代码通常不用改，除非你要魔改模型结构
│  │  __init__.py
│  │
│  ├─configs             # 预设配置
│  │
│  ├─core                # 底层核心功能（显存管理、FlashAttention、数据加载器）
│  │
│  ├─diffusion           # 扩散算法核心逻辑
│  │      flow_match.py  # <-- Wan 模型使用的是 Flow Matching 算法，核心数学公式在这里
│  │      runner.py      # <-- 训练循环的主控代码（forward, backward, optimizer step）
│  │      training_module.py # <-- 定义了训练时的核心模块
│  │
│  ├─models              # 【模型架构定义】Wan 模型的长相（层、注意力机制）都在这
│  │      ... (Flux, LTX, Qwen 等其他模型，跳过)
│  │      wan_video_dit.py         # <-- 【关键】Wan 的 DiT (Diffusion Transformer) 架构代码
│  │      wan_video_vae.py         # <-- Wan 的 VAE (视频压缩/解压) 架构代码
│  │      wan_video_text_encoder.py # <-- Wan 的文本编码器 (T5/UMT5)
│  │      ...
│  │
│  ├─pipelines           # 【推理流水线】定义了怎么把上面那些零件拼起来生成视频
│  │      wan_video.py   # <-- 【关键】Wan 模型的推理逻辑 (文生视频、图生视频的流程)
│  │      ...
│  │
│  └─utils               # 工具箱
│      ├─lora            # <-- LoRA 的实现逻辑（怎么加载、怎么合并）
│      └─...
│
├─docs                   # 文档（遇到不懂的参数可以来这就查 MD 文件）
│  ...
│
└─examples               # 【你的主战场】所有的训练脚本、启动命令都在这里！！！
    │
    ├─wanvideo           # <--- 【重点关注】你只需要在这个文件夹里干活
    │  ├─acceleration
    │  │      unified_sequence_parallel.py # 序列并行加速代码（多卡训练长视频用）
    │  │
    │  ├─model_inference # 【推理脚本】训练完用来测试生成的
    │  │      Wan2.1-T2V-1.3B.py       # <-- 1.3B 模型文生视频测试脚本
    │  │      Wan2.1-T2V-14B.py        # <-- 14B 模型文生视频测试脚本
    │  │      Wan2.1-I2V-14B-720P.py   # <-- 图生视频测试脚本
    │  │      ...
    │  │
    │  └─model_training  # 【训练脚本】你要改的代码在这里！！！
    │      │  train.py   # <--- ★★★【最核心】通用的训练入口文件，读取参数、启动训练
    │      │
    │      ├─full        # 全量微调 (Full Fine-tuning) 的配置
    │      │      accelerate_config_14B.yaml  # <-- DeepSpeed/Accelerate 配置文件
    │      │      accelerate_config_zero3.yaml # <-- Phase 2/3 需要用到的 DeepSpeed ZeRO3 配置
    │      │      Wan2.1-T2V-14B.sh           # <-- 14B 全量训练的启动脚本例子
    │      │      ...
    │      │
    │      ├─lora        # <--- ★★★【你的 Phase 1 战场】LoRA 微调脚本
    │      │      Wan2.1-T2V-1.3B.sh          # <-- 【推荐】Phase 1 拿这个脚本修改来练手
    │      │      Wan2.1-T2V-14B.sh           # <-- Phase 2 训练 14B LoRA 的启动脚本
    │      │      Wan2.1-I2V-14B-720P.sh      # <-- 如果你要训练图生视频 LoRA，看这个
    │      │
    │      └─validate_lora # 验证脚本（加载训练好的lora模型参数，生成看效果）
    │             ...
    │
    └─... (Flux, Z-Image 等其他模型的例子，完全不用看)
```

以`\examples\wanvideo\model_training\lora\Wan2.1-T2V-1.3B.sh`为例，解释lora训练各参数的含义
```bash
# 使用 accelerate 启动训练: accelerate 是 HuggingFace 的库，负责自动处理显卡分配、混合精度（fp16/bf16）等环境配置
accelerate launch examples/wanvideo/model_training/train.py \
  \
  # 【视频文件的根目录】: 程序会去这个目录下找 metadata.csv/json 里提到的文件名
  --dataset_base_path data/example_video_dataset \
  \
  # 【数据集的“索引文件”】 (支持 csv 或 json),里面记录了：视频文件名、提示词(prompt)、宽、高、帧数等信息
  --dataset_metadata_path data/example_video_dataset/metadata.csv \
  \
  # 【分辨率设置】
  # Wan 模型对分辨率敏感，推荐：480x832 (竖屏) 或 832x480 (横屏)
  # 如果你的视频比例不一致，会被裁剪或缩放
  --height 480 \
  --width 832 \
  \
  # 【数据集重复次数】
  # 如果你的数据很少（比如只有 5 个视频），一个 epoch 瞬间就跑完了，显卡显存频繁加载/卸载很慢
  # 设置为 100 表示把这 5 个视频重复练 100 次算作 1 个 epoch，减少保存 checkpoint 的频率
  --dataset_repeat 100 \
  \
  # 【模型路径映射】
  # 格式：ID:文件规则1,ID:文件规则2...
  # 作用：告诉 DiffSynth 去哪里找底模的权重文件
  #  - diffusion_pytorch_model*.safetensors：Wan 1.3B 的主干扩散/DiT权重（通配符匹配）
  #  - models_t5_umt5-xxl-enc-bf16.pth：文本编码器（umt5-xxl encoder）权重
  #  - Wan2.1_VAE.pth：VAE（视频/图像latent编解码）权重
  --model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth" \
  \
  # 【学习率】
  # LoRA 训练通常在 1e-4 到 5e-5 之间。如果过拟合（画面崩坏），调低它；如果不收敛（学不像），调高它
  --learning_rate 1e-4 \
  \
  # 【训练轮数】
  # 跑完所有数据算 1 个 epoch。结合 dataset_repeat 使用
  --num_epochs 5 \
  \
  # 【保存权重时的前缀清理】
  # 为了让保存出来的 LoRA 权重名字更干净，兼容性更好
  --remove_prefix_in_ckpt "pipe.dit." \
  \
  # 【输出目录】
  # 训练好的 .safetensors 文件会保存在这里
  --output_path "./models/train/Wan2.1-T2V-1.3B_lora" \
  \
  # 【LoRA 基础模型】
  # 指定我们要训练的是 DiT (Diffusion Transformer) 部分，而不是 VAE 或 T5
  --lora_base_model "dit" \
  \
  # 【LoRA 注入目标】
  # LoRA 注入的目标层（按模块/子层名匹配）：
  #  - q,k,v,o：attention 的 Q/K/V/out 投影线性层
  #  - ffn.0, ffn.2：前馈网络 FFN 中第1/第3个线性层（常见结构是 Linear -> 激活 -> Linear）
  # 是否命中要看 wan_video_dit.py 里模块命名；注入数=0 说明匹配失败
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  \
  # 【LoRA Rank (秩)】
  # 控制 LoRA 的大小和容量。16 或 32 是常规选择；64 或 128 适合学习非常复杂的风格，但显存占用会增加
  --lora_rank 32
```

# DiffSynth-Studio 到底做了什么？

**结论：不仅仅是 Copy 源码，它是“重构 + 优化 + 统一接口”。**

如果只是简单的 Copy，那它毫无价值，确实不如直接去官方仓库。DiffSynth-Studio (DSS) 的集成方式主要是“**统一框架 + 模型适配层 + 权重转换层 + 通用训练/推理管线**”，而不只是把各家源码粗暴复制进来。

## 把共性（训练循环、日志、显存管理、数据接口）做成框架，把差异（模型结构、权重加载）做成适配器

不管你训练哪个模型，启动命令、参数名字、数据格式都是一模一样的。

- **统一训练/推理框架**：`diffsynth/diffusion/*`  
  这里提供了通用的 runner、loss、training_module、logger 等。不同模型复用同一套训练骨架。

- **统一 pipeline 抽象**：`diffsynth/diffusion/base_pipeline.py` + `diffsynth/pipelines/*`  
  每个模型家族一个 pipeline（如 `wan_video.py`），负责把组件拼起来并对外提供一致的调用方式。

- **模型组件适配层**：`diffsynth/models/*`  
  这里不是“整个官方 repo 原封不动搬进来”，而是把模型拆成统一可组装的模块（text encoder / VAE / DiT / 控制模块等），并按 DiffSynth 的框架接口组织。

- **权重 key 映射/转换层（非常关键）**：`diffsynth/utils/state_dict_converters/*`  
  这是很多“只下官方源码”并不需要做、但“统一框架集成”必须做的工作：不同 repo 的权重命名、层级结构不同，Studio 通过 converter 把权重映射到自己这套模块命名上。

- **显存/设备/加速统一能力**：`diffsynth/core/vram/*`、`core/gradient/*`、`core/attention/*`、`examples/wanvideo/acceleration/*`  
  这些属于“跨模型复用的系统优化”，官方单个模型 repo 往往不会为别的模型考虑这么通用的 VRAM 管理。

## 显存管理的魔改 (Memory Optimization)
这是 DSS 最大的卖点
*   **官方源码：** 往往默认你有 H800 或者 A100 (80GB) 显卡。代码里写得很奔放，模型一次性全部加载到显存。
*   **DSS 的做法：** 它的 `core/vram` 模块重写了加载逻辑。它能做到：
    *   **Layer-wise Loading：** 算哪一层，加载哪一层，算完立刻卸载。
    *   **CPU Offload：** 把暂时不用的参数扔到内存（RAM）里，而不是显存（VRAM）里。
    *   **结果：** 官方代码需要 80GB 显存才能跑的任务，DSS 可能 24GB 就能跑。**这是你自己去改官方源码很难做到的，因为涉及到底层架构。**

## 训练脚本的标准化
*   官方通常只提供推理（生成）代码，训练代码往往藏着掖着，或者非常简陋。
*   DSS 提供了工业级的训练脚本（支持 Checkpointing, Logging, 各种 Loss 计算），让你开箱即用。

# Wan 模型 LoRA 微调需要操作的文件

## Phase 1 - 1.3B 模型验证
*   **位置：** `examples/wanvideo/model_training/lora/`
*   **目标文件：** `Wan2.1-T2V-1.3B.sh` (或者类似的 .sh 文件)
*   **操作：**
    *   复制一份，重命名为 `my_train_phase1.sh`。
    *   用记事本或 VS Code 打开它。
    *   修改里面的参数：
        *   `--pretrained_model_name_or_path`: 改成你下载好的 Wan 1.3B 模型路径。
        *   `--dataset_path`: 改成你准备好的 JSON 数据集文件路径。
        *   `--output_dir`: 改成你想保存 LoRA 的路径。
    *   **运行：** 在终端执行 `bash examples/wanvideo/model_training/lora/my_train_phase1.sh`。

## Phase 2 - 14B 模型 + DeepSpeed
*   **位置：** `examples/wanvideo/model_training/full/`
*   **借用目标文件：** `accelerate_config_zero3.yaml` (或者 `zero2`)
    *   它不是“训练脚本”，而是 **DeepSpeed 的配置文件**，定义了显存优化策略
    *   它不管你是在练 LoRA 还是练全量微调，它只管一件事：**如何把这个巨大的 14B 模型切碎了塞进你的显卡里**。它的内容通常是：有多少张卡？用不用 DeepSpeed？用 ZeRO 几阶段？用不用 fp16？
    *  **为什么要借用？**： Wan 14B 模型非常巨大。即使你只练 LoRA（只更新 1% 的参数），但在训练过程中，你需要**加载整个 14B 的底模**到显存里进行前向传播。如果没有 DeepSpeed ZeRO-3 这种“切片技术”，单张显卡根本装不下 14B 的底模。
*   **操作：**
    *   回到你的 `my_train_phase1.sh`，在命令最前面加上 `accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_zero3.yaml ...`
    *   把模型路径改成 14B 的路径

# 快速走通 Wan 模型 LoRA 微调全流程

## 第一步：造数据集

DiffSynth-Studio **不包含** 现成的视频数据集（因为视频太大了，放进 GitHub 会爆炸）。

modelscope提供了一个样例视频数据集，以方便进行测试，通过以下命令可以下载这个数据集：
```bash
pip install modelscope

modelscope download --dataset DiffSynth-Studio/example_video_dataset --local_dir ./data/example_video_dataset
```

## 第二步：修改并运行训练脚本
找到 `examples/wanvideo/model_training/lora/Wan2.1-T2V-1.3B.sh`

复制一份出来改：
```bash
cp examples/wanvideo/model_training/lora/Wan2.1-T2V-1.3B.sh my_train.sh
chmod +x my_train.sh
```

**编辑 `my_train.sh` (使用 vim 或 nano):**
```bash
1. 必改项
    --dataset_base_path: 你的视频文件所在的文件夹。
    --dataset_metadata_path: 你的 json/csv 索引文件。

2. 建议修改项

    --output_path: 改成你自己喜欢的目录 (比如 ./output_my_test)。
        原因： 默认路径可能会覆盖掉之前的实验结果，或者混在一起让你分不清哪个是哪个。
    --height 和 --width: 改成和你视频接近的分辨率。
        原因： 脚本默认可能是 480x832。如果你的视频是 1024x1024 的，强制压缩会导致画面变形；如果你的视频很小，强制放大浪费显存。
        Wan 模型推荐分辨率： 480x832 (竖屏) 或 832x480 (横屏)。

```

## 第三步：启动并观察结果

```bash
CUDA_VISIBLE_DEVICES=2 bash examples/wanvideo/model_training/lora/my_train.sh
```

观察
1. `nvidia-smi`查看GPU占用情况
2. **看输出目录：** 去 `--output_path`参数对应的文件夹看，如果生成了对应的lora参数权重，说明训练成功

