#!/bin/bash
# ===========================================================================
# PVTT 批量推理：中性色填充 + Bbox Mask（原始输出，不合成）
#
# 与 run_pvtt_neutral_fill_bbox.sh 的唯一区别：
#   输出视频是模型直接生成的原始视频，不经过 composite_with_mask 合成。
#   用于验证"割裂感"是否来自合成步骤。
#
# 用法：
#   bash scripts/run_pvtt_neutral_fill_bbox_raw.sh
#
#   # 跑全部 199 个任务
#   SAMPLED=0 bash scripts/run_pvtt_neutral_fill_bbox_raw.sh
# ===========================================================================

set -e

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# --- 路径配置 ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PVTT_DIR="$PROJECT_DIR/pvtt_evaluation"
DATASET_ROOT="$PROJECT_DIR/samples/pvtt_evaluation_datasets"

# --- 模型 ---
MODEL_SIZE="${WAN_VACE_MODEL_SIZE:-1.3B}"

# --- 输出目录（带时间戳）---
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/experiments/results/$MODEL_SIZE/pvtt_neutral_fill_bbox_raw/$TIMESTAMP}"

# --- 视频参数 ---
NUM_FRAMES="${NUM_FRAMES:-81}"
SEED="${SEED:-42}"
FPS="${FPS:-16}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
CFG_SCALE="${CFG_SCALE:-7.5}"

# --- 实验特有参数 ---
FILL_VALUE="${FILL_VALUE:-128}"
BBOX_MARGIN_LR="${BBOX_MARGIN_LR:-20}"
BBOX_MARGIN_TOP="${BBOX_MARGIN_TOP:-20}"
BBOX_MARGIN_BOTTOM="${BBOX_MARGIN_BOTTOM:-20}"

# --- 采样任务（每种产品 3 个，共 23 个） ---
SAMPLED_TASK_IDS="0016-bracelet1_to_bracelet2,0017-bracelet2_scene01_to_bracelet1,0017-bracelet2_scene02_to_bracelet1,0021-earring1_to_earring2,0022-earring2_to_earring1,0023-earring3_scene01_to_earring1,0006-handbag1_scene01_to_handbag2,0006-handbag1_scene06_to_handbag2,0007-handbag2_to_handbag1,0001-handfan1_to_handfan2,0002-handfan2_to_handfan1,0026-necklace1_to_necklace2,0027-necklace2_to_necklace1,0028-necklace3_scene01_to_necklace1,0012-purse1_to_purse2,0013-purse2_to_purse1,0014-purse3_scene01_to_purse1,0003-sunglasses1_scene01_to_sunglasses2,0003-sunglasses1_scene02_to_sunglasses2,0004-sunglasses2_to_sunglasses1,0031-watch1_to_watch2,0032-watch2_to_watch1,0033-watch3_scene02_to_watch1"

# 是否使用采样子集
TASK_IDS_FLAG=""
if [ "${SAMPLED:-1}" = "1" ]; then
    TASK_IDS_FLAG="--task_ids $SAMPLED_TASK_IDS"
elif [ -n "$TASK_IDS" ]; then
    TASK_IDS_FLAG="--task_ids $TASK_IDS"
fi

# --- 控制参数 ---
SKIP_FLAG=""
if [ "${SKIP_EXISTING:-1}" = "1" ]; then
    SKIP_FLAG="--skip_existing"
fi

START_IDX="${START_IDX:-0}"
MAX_TASKS_FLAG=""
if [ -n "$MAX_TASKS" ]; then
    MAX_TASKS_FLAG="--max_tasks $MAX_TASKS"
fi

# ===========================================================================
echo "============================================================"
echo "PVTT 批量推理：中性色填充 + Bbox Mask（原始输出，不合成）"
echo "============================================================"
echo "数据集目录:    $DATASET_ROOT"
echo "输出目录:      $OUTPUT_ROOT"
echo "模型大小:      $MODEL_SIZE"
echo "帧数:          $NUM_FRAMES"
echo "填充值:        $FILL_VALUE"
echo "Bbox 边距:     lr=$BBOX_MARGIN_LR, top=$BBOX_MARGIN_TOP, bottom=$BBOX_MARGIN_BOTTOM"
echo "采样模式:      ${SAMPLED:-1}"
echo "============================================================"

python3 "$PVTT_DIR/run_pvtt_neutral_fill_bbox_raw.py" \
    --dataset_root "$DATASET_ROOT" \
    --output_root "$OUTPUT_ROOT" \
    --model_size "$MODEL_SIZE" \
    --num_frames "$NUM_FRAMES" \
    --seed "$SEED" \
    --fps "$FPS" \
    --num_inference_steps "$NUM_INFERENCE_STEPS" \
    --cfg_scale "$CFG_SCALE" \
    --fill_value "$FILL_VALUE" \
    --bbox_margin_lr "$BBOX_MARGIN_LR" \
    --bbox_margin_top "$BBOX_MARGIN_TOP" \
    --bbox_margin_bottom "$BBOX_MARGIN_BOTTOM" \
    --start_idx "$START_IDX" \
    $MAX_TASKS_FLAG \
    $SKIP_FLAG \
    $TASK_IDS_FLAG

EXIT_CODE=$?
echo ""
echo "推理完成，退出码: $EXIT_CODE"
exit $EXIT_CODE
