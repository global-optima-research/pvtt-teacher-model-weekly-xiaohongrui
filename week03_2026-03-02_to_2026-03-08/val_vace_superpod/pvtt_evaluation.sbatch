#!/bin/bash
#SBATCH --job-name=pvtt_eval           # 作业名称
#SBATCH --output=logs/pvtt_%j.log      # 标准输出日志
#SBATCH --error=logs/pvtt_err_%j.log   # 标准错误日志
#SBATCH --nodes=1                      # 单节点
#SBATCH --gpus=1                       # 1 张 GPU
#SBATCH --time=24:00:00                # 24 小时（199 个任务需要较长时间）
#SBATCH --partition=normal
#SBATCH --account=mscaisuperpod

# =============================================================================
# PVTT 评估流程：掩码提取 + Canny 首帧粘贴推理
#
# 用法：
#   # 完整流程（掩码提取 + 推理）
#   sbatch pvtt_evaluation.sbatch
#
#   # 仅推理（掩码已提取）
#   SKIP_MASK_EXTRACTION=1 sbatch pvtt_evaluation.sbatch
#
#   # 仅提取掩码
#   SKIP_INFERENCE=1 sbatch pvtt_evaluation.sbatch
#
#   # 测试模式（仅前 5 个任务）
#   MAX_TASKS=5 sbatch pvtt_evaluation.sbatch
#
#   # 断点续跑
#   START_IDX=50 SKIP_EXISTING=1 sbatch pvtt_evaluation.sbatch
# =============================================================================

cd /home/hxiaoap/val_vace_superpod
mkdir -p logs

# =============================================================================
# 1. 环境配置
# =============================================================================

module purge
module load Anaconda3/2023.09-0
module load cuda12.2/toolkit/12.2.2

ENV_PATH="/home/hxiaoap/.conda/envs/diffsynth"
export PATH="$ENV_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_PATH/lib:$LD_LIBRARY_PATH"

echo "=== 环境信息 ==="
echo "节点: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Python: $(python --version)"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"

# Python 版本检查
if [[ "$(python --version)" != *"3.10"* ]]; then
    echo "Python 版本不匹配！预期 3.10，实际 $(python --version)"
    exit 1
fi

# =============================================================================
# 2. 配置
# =============================================================================

# 模型大小
export WAN_VACE_MODEL_SIZE="${WAN_VACE_MODEL_SIZE:-1.3B}"

# 控制参数
SKIP_MASK_EXTRACTION="${SKIP_MASK_EXTRACTION:-0}"
SKIP_INFERENCE="${SKIP_INFERENCE:-0}"

# 传递给推理脚本的参数
export START_IDX="${START_IDX:-0}"
export MAX_TASKS="${MAX_TASKS:-}"
export TASK_IDS="${TASK_IDS:-}"
export SKIP_EXISTING="${SKIP_EXISTING:-0}"
export NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
export CFG_SCALE="${CFG_SCALE:-7.5}"

# GroundingDINO + SAM2 模型路径（根据服务器环境修改）
export SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-/home/hxiaoap/models/sam2.1_hiera_large.pt}"
export SAM2_CONFIG="${SAM2_CONFIG:-/home/hxiaoap/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml}"
export GDINO_CONFIG="${GDINO_CONFIG:-/home/hxiaoap/models/GroundingDINO_SwinT_OGC.py}"
export GDINO_CHECKPOINT="${GDINO_CHECKPOINT:-/home/hxiaoap/models/groundingdino_swint_ogc.pth}"

echo ""
echo "=== PVTT 评估配置 ==="
echo "模型大小:          $WAN_VACE_MODEL_SIZE"
echo "跳过掩码提取:      $SKIP_MASK_EXTRACTION"
echo "跳过推理:          $SKIP_INFERENCE"
echo "起始索引:          $START_IDX"
echo "最大任务数:        ${MAX_TASKS:-全部}"
echo "跳过已完成:        $SKIP_EXISTING"
echo ""

# 修复 Windows 换行符
sed -i 's/\r$//' ./scripts/*.sh
chmod +x ./scripts/*.sh

TOTAL_START=$(date +%s)

# =============================================================================
# 3. Step 1: 掩码提取
# =============================================================================

if [ "$SKIP_MASK_EXTRACTION" != "1" ]; then
    echo "============================================================"
    echo "Step 1: 掩码提取 (GroundingDINO + SAM2)"
    echo "============================================================"

    STEP1_START=$(date +%s)

    bash ./scripts/run_pvtt_extract_masks.sh
    RET=$?

    STEP1_END=$(date +%s)
    STEP1_DUR=$((STEP1_END - STEP1_START))
    echo "掩码提取完成，耗时: $((STEP1_DUR/60))m $((STEP1_DUR%60))s, 退出码: $RET"

    if [ $RET -ne 0 ]; then
        echo "掩码提取失败！退出码: $RET"
        # 不终止，部分视频可能已成功
    fi
else
    echo "跳过掩码提取（SKIP_MASK_EXTRACTION=1）"
fi

# =============================================================================
# 4. Step 2: Canny 首帧粘贴推理
# =============================================================================

if [ "$SKIP_INFERENCE" != "1" ]; then
    echo ""
    echo "============================================================"
    echo "Step 2: PVTT Canny 首帧粘贴推理"
    echo "============================================================"

    STEP2_START=$(date +%s)

    bash ./scripts/run_pvtt_canny_paste.sh
    RET=$?

    STEP2_END=$(date +%s)
    STEP2_DUR=$((STEP2_END - STEP2_START))
    echo "推理完成，耗时: $((STEP2_DUR/60))m $((STEP2_DUR%60))s, 退出码: $RET"
else
    echo "跳过推理（SKIP_INFERENCE=1）"
fi

# =============================================================================
# 5. 汇总
# =============================================================================

TOTAL_END=$(date +%s)
TOTAL_DUR=$((TOTAL_END - TOTAL_START))
HOURS=$((TOTAL_DUR / 3600))
MINUTES=$(((TOTAL_DUR % 3600) / 60))
SECONDS=$((TOTAL_DUR % 60))

echo ""
echo "============================================"
echo "PVTT 评估流程完成"
echo "总耗时: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "============================================"
