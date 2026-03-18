#!/bin/bash
#SBATCH --job-name=ffgo_pvtt            # 作业名称
#SBATCH --output=logs/ffgo_%j.log       # 标准输出日志
#SBATCH --error=logs/ffgo_err_%j.log    # 标准错误日志
#SBATCH --nodes=1                       # 单节点运行
#SBATCH --gpus=1                        # 1 张 GPU（14B 模型推荐 H100/H200 80G+）
#SBATCH --time=12:00:00                 # 时间限制（12 小时，14B 模型推理较慢）
#SBATCH --partition=normal              # 分区名称
#SBATCH --account=mscaisuperpod         # 账户名称

# =============================================================================
# FFGo 原版模型在 PVTT 数据集上的推理
#
# 模型：Wan2.2-I2V-A14B + FFGo LoRA adapter
#
# 用法：
#   sbatch ffgo_original.sbatch
#
#   # 自定义分辨率（降低显存需求）
#   sbatch --export=ALL,HEIGHT=480,WIDTH=640 ffgo_original.sbatch
#
#   # 跑全部 199 个任务
#   sbatch --export=ALL,SAMPLED=0 ffgo_original.sbatch
# =============================================================================

# =============================================================================
# 1. 环境配置
# =============================================================================

cd /home/hxiaoap/val_vace_superpod
mkdir -p logs

# 加载模块
module purge
module load Anaconda3/2023.09-0
module load cuda12.2/toolkit/12.2.2

# 激活 环境
ENV_PATH="/home/hxiaoap/.conda/envs/diffsynth"
export PATH="$ENV_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_PATH/lib:$LD_LIBRARY_PATH"

# HF 镜像（国内服务器加速下载）
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "=== 环境信息 ==="
echo "节点: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Python 路径: $(which python)"
echo "Python 版本: $(python --version)"

python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"

# =============================================================================
# 2. 实验配置
# =============================================================================

echo ""
echo "=== 实验配置 ==="
echo "实验:      FFGo 原版模型（Wan2.2-I2V-A14B + LoRA）"
echo "分辨率:    ${WIDTH:-832}x${HEIGHT:-480}"
echo "帧数:      ${VIDEO_LENGTH:-81}"
echo "采样模式:  ${SAMPLED:-1}"

# 修复 Windows 换行符
sed -i 's/\r$//' ./scripts/*.sh
chmod +x ./scripts/*.sh

# =============================================================================
# 3. 执行
# =============================================================================

TOTAL_START=$(date +%s)

echo ""
echo "============================================================"
echo "开始执行: FFGo 原版模型 PVTT 推理"
echo "开始时间: $(date +'%Y-%m-%d %H:%M:%S')"
echo "============================================================"

bash ./scripts/run_pvtt_ffgo_original.sh
RET_CODE=$?

# =============================================================================
# 4. 汇总
# =============================================================================

TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))
SECONDS=$((TOTAL_DURATION % 60))

echo ""
echo "============================================"
if [ $RET_CODE -eq 0 ]; then
    echo "FFGo 原版模型 PVTT 推理完成。"
else
    echo "FFGo 原版模型 PVTT 推理失败，退出码: $RET_CODE"
fi
echo "总耗时: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "输出目录: experiments/results/ffgo_original/pvtt/"
echo "============================================"

exit $RET_CODE
