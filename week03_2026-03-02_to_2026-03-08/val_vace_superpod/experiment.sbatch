#!/bin/bash
#SBATCH --job-name=vace_exp            # 作业名称
#SBATCH --output=logs/vace_%j.log      # 标准输出日志
#SBATCH --error=logs/vace_err_%j.log   # 标准错误日志
#SBATCH --nodes=1                      # 单节点运行
#SBATCH --gpus=1                       # 1 张 GPU（14B 推荐 A100/H100 80G）
#SBATCH --time=08:00:00                # 时间限制（8 小时）
#SBATCH --partition=normal             # 分区名称
#SBATCH --account=mscaisuperpod        # 账户名称

# =============================================================================
# 1. 环境配置
# =============================================================================

cd /home/hxiaoap/val_vace_superpod
mkdir -p logs

# 加载模块
module purge
module load Anaconda3/2023.09-0
module load cuda12.2/toolkit/12.2.2

# 激活 conda 环境
ENV_PATH="/home/hxiaoap/.conda/envs/diffsynth"
export PATH="$ENV_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_PATH/lib:$LD_LIBRARY_PATH"

echo "=== 环境信息 ==="
echo "节点: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Python 路径: $(which python)"
echo "Python 版本: $(python --version)"

# Python 版本检查
if [[ "$(python --version)" != *"3.10"* ]]; then
    echo "Python 版本不匹配！预期 3.10，实际 $(python --version)"
    echo "请检查 ENV_PATH 是否正确。"
    exit 1
fi
echo "环境检查通过: Python 3.10"

python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"

# =============================================================================
# 2. 实验配置
# =============================================================================

# 模型运行模式：1.3B / 14B / both
#   1.3B  = 只用 1.3B 模型跑所有实验
#   14B   = 只用 14B 模型跑所有实验
#   both  = 先用 1.3B 跑完所有实验，再用 14B 跑完所有实验
MODEL_RUN_MODE="${MODEL_RUN_MODE:-both}"

case "$MODEL_RUN_MODE" in
    "1.3B") MODEL_SIZES=("1.3B") ;;
    "14B")  MODEL_SIZES=("14B") ;;
    "both") MODEL_SIZES=("1.3B" "14B") ;;
    *)
        echo "错误: MODEL_RUN_MODE 只能是 1.3B / 14B / both，当前值: $MODEL_RUN_MODE"
        exit 1
        ;;
esac

# 通用参数（可在此处覆盖默认值）
export SAMPLE_NAME="${SAMPLE_NAME:-teapot}"
export WIDTH="${WIDTH:-480}"
export HEIGHT="${HEIGHT:-848}"
export NUM_FRAMES="${NUM_FRAMES:-49}"
export SEED="${SEED:-42}"

echo ""
echo "=== 实验配置 ==="
echo "运行模式:  $MODEL_RUN_MODE -> 模型列表: ${MODEL_SIZES[*]}"
echo "样本:      $SAMPLE_NAME"
echo "分辨率:    ${WIDTH}x${HEIGHT}"
echo "帧数:      $NUM_FRAMES"
echo "种子:      $SEED"

# 要运行的实验脚本列表
# 注：可根据需要注释/取消注释来选择要运行的实验
SCRIPTS=(
    # --- 第一轮实验（基础） ---
    # "./scripts/run_precise_mask.sh"
    # "./scripts/run_bbox_mask.sh"
    # "./scripts/run_reactive_weight.sh"
    # "./scripts/run_reactive_weight_bbox.sh"

    # --- 第二轮实验（Neutral Fill 预处理） ---
    # "./scripts/run_neutral_fill.sh"
    # "./scripts/run_neutral_fill_bbox.sh"
    # "./scripts/run_neutral_fill_growmask.sh"
    # "./scripts/run_comfyui_baseline.sh"

    # --- 第四轮实验（Canny 结构控制） ---
    "./scripts/run_canny_control.sh"
    "./scripts/run_canny_paste.sh"
)

# 修复 Windows 换行符（CRLF → LF），防止 bash 解析错误
sed -i 's/\r$//' ./scripts/*.sh

chmod +x ./scripts/*.sh

# =============================================================================
# 3. 执行循环（按模型大小分轮运行）
# =============================================================================

TOTAL_START=$(date +%s)

for MODEL in "${MODEL_SIZES[@]}"; do
    export WAN_VACE_MODEL_SIZE="$MODEL"

    echo ""
    echo "============================================================"
    echo "========== 开始使用 $MODEL 模型运行所有实验 =========="
    echo "============================================================"

    for SCRIPT in "${SCRIPTS[@]}"; do
        echo ""
        echo "############################################################"
        echo "[$MODEL] 开始执行: $SCRIPT"
        echo "############################################################"

        EXP_START=$(date +%s)
        echo "开始时间: $(date +'%Y-%m-%d %H:%M:%S')"

        bash "$SCRIPT"
        RET_CODE=$?

        EXP_END=$(date +%s)
        DURATION=$((EXP_END - EXP_START))
        MM=$((DURATION / 60))
        SS=$((DURATION % 60))

        echo "------------------------------------------------------------"
        if [ $RET_CODE -eq 0 ]; then
            echo "[$MODEL] 成功: $SCRIPT 执行完毕。"
        else
            echo "[$MODEL] 失败: $SCRIPT 退出码 $RET_CODE。"
            # 如需在第一个失败时停止，取消下行注释：
            # exit 1
        fi
        echo "结束时间: $(date +'%Y-%m-%d %H:%M:%S')  耗时: ${MM}m ${SS}s"
        echo "############################################################"
    done

    echo ""
    echo "========== $MODEL 模型所有实验完成 =========="
done

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
echo "所有实验执行完毕。"
echo "总耗时: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "输出目录: experiments/results/"
echo "============================================"
