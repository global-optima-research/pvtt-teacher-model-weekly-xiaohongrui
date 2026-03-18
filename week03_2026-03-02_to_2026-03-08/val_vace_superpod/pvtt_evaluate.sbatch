#!/bin/bash
#SBATCH --job-name=pvtt_eval             # 作业名称
#SBATCH --output=logs/eval_%j.log        # 标准输出日志
#SBATCH --error=logs/eval_err_%j.log     # 标准错误日志
#SBATCH --nodes=1                        # 单节点
#SBATCH --gpus=1                         # 1 张 GPU（评估模型较小，任意 GPU 即可）
#SBATCH --time=02:00:00                  # 时间限制（2 小时，30 个任务足够）
#SBATCH --partition=normal               # 分区名称
#SBATCH --account=mscaisuperpod          # 账户名称

# =============================================================================
# PVTT 评估脚本：计算 FiVE-Bench / Edit Success / VBench 三组指标
#
# 需要 GPU（CLIP、LPIPS、DINOv2 推理），但显存需求很低（<4GB）。
#
# 用法：
#   sbatch pvtt_evaluate.sbatch
#
#   # 评估指定目录
#   sbatch --export=ALL,GENERATED_DIR=experiments/results/ffgo_original/pvtt/20260317_185832 pvtt_evaluate.sbatch
#
#   # 同时评估两个批次（合并）
#   sbatch --export=ALL,EVAL_BOTH=1 pvtt_evaluate.sbatch
# =============================================================================

# =============================================================================
# 1. 环境配置
# =============================================================================

cd /home/hxiaoap/val_vace_superpod
mkdir -p logs

module purge
module load Anaconda3/2023.09-0
module load cuda12.2/toolkit/12.2.2

# 使用 diffsynth 环境（评估依赖较轻，复用即可）
ENV_PATH="/home/hxiaoap/.conda/envs/diffsynth"
export PATH="$ENV_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_PATH/lib:$LD_LIBRARY_PATH"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "=== 环境信息 ==="
echo "节点: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Python: $(python --version)"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# =============================================================================
# 2. 评估配置
# =============================================================================

DATASET_ROOT="samples/pvtt_evaluation_datasets"
SKIP_FRAMES="${SKIP_FRAMES:-4}"
EVAL_H="${EVAL_H:-480}"
EVAL_W="${EVAL_W:-832}"

# 默认评估两个批次
RUN1="experiments/results/ffgo_original/pvtt/20260317_165346"
RUN2="experiments/results/ffgo_original/pvtt/20260317_185832"

# 可通过 GENERATED_DIR 覆盖为单个目录
GENERATED_DIR="${GENERATED_DIR:-}"

# 修复 Windows 换行符
sed -i 's/\r$//' ./scripts/*.sh 2>/dev/null
sed -i 's/\r$//' ./pvtt_evaluation/metrics/*.py 2>/dev/null

# =============================================================================
# 3. 执行评估
# =============================================================================

TOTAL_START=$(date +%s)

if [ -n "$GENERATED_DIR" ]; then
    # 评估单个目录
    echo ""
    echo "============================================================"
    echo "评估目录: $GENERATED_DIR"
    echo "跳过转场帧: $SKIP_FRAMES"
    echo "============================================================"

    python pvtt_evaluation/metrics/evaluate_pvtt.py \
        --generated_dir "$GENERATED_DIR" \
        --dataset_root "$DATASET_ROOT" \
        --skip_frames "$SKIP_FRAMES" \
        --eval_h "$EVAL_H" \
        --eval_w "$EVAL_W" \
        --verbose

else
    # 分别评估两个批次
    for RUN_DIR in "$RUN1" "$RUN2"; do
        if [ ! -d "$RUN_DIR" ]; then
            echo "目录不存在，跳过: $RUN_DIR"
            continue
        fi

        TIMESTAMP=$(basename "$RUN_DIR")
        OUTPUT_CSV="$RUN_DIR/evaluation_results.csv"
        OUTPUT_JSON="$RUN_DIR/evaluation_summary.json"

        echo ""
        echo "============================================================"
        echo "评估目录: $RUN_DIR"
        echo "输出 CSV: $OUTPUT_CSV"
        echo "输出 JSON: $OUTPUT_JSON"
        echo "跳过转场帧: $SKIP_FRAMES"
        echo "============================================================"

        python pvtt_evaluation/metrics/evaluate_pvtt.py \
            --generated_dir "$RUN_DIR" \
            --dataset_root "$DATASET_ROOT" \
            --output_csv "$OUTPUT_CSV" \
            --output_summary "$OUTPUT_JSON" \
            --skip_frames "$SKIP_FRAMES" \
            --eval_h "$EVAL_H" \
            --eval_w "$EVAL_W" \
            --verbose

        echo ""
        echo "评估完成: $RUN_DIR"
    done
fi

# =============================================================================
# 4. 汇总
# =============================================================================

TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))
MINUTES=$((TOTAL_DURATION / 60))
SECONDS=$((TOTAL_DURATION % 60))

echo ""
echo "============================================"
echo "所有评估完成。"
echo "总耗时: ${MINUTES}m ${SECONDS}s"
echo "============================================"
