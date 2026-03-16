#!/bin/bash
# ===========================================================================
# PVTT 数据集批量推理：Canny 首帧粘贴（实验 10 工作流）
#
# 默认采样每种产品 3 个任务（共 23 个），不跑全部 199 个。
# 前置条件：masks 已通过 run_pvtt_extract_masks.sh 提取完成。
#
# 用法：
#   bash scripts/run_pvtt_canny_paste.sh
#
#   # 跑全部 199 个任务
#   SAMPLED=0 bash scripts/run_pvtt_canny_paste.sh
#
#   # 指定模型大小
#   WAN_VACE_MODEL_SIZE=14B bash scripts/run_pvtt_canny_paste.sh
#
#   # 断点续跑（跳过已完成的）
#   SKIP_EXISTING=1 bash scripts/run_pvtt_canny_paste.sh
#
#   # 仅运行指定任务
#   TASK_IDS="0001-handfan1_to_handfan2,0002-handfan2_to_handfan1" \
#       bash scripts/run_pvtt_canny_paste.sh
# ===========================================================================

set -e

# HuggingFace 镜像
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# --- 路径配置 ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PVTT_DIR="$PROJECT_DIR/pvtt_evaluation"
DATASET_ROOT="$PROJECT_DIR/samples/pvtt_evaluation_datasets"

# --- 模型配置 ---
MODEL_SIZE="${WAN_VACE_MODEL_SIZE:-1.3B}"

# --- 输出目录 ---
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/experiments/results/${MODEL_SIZE}/pvtt_canny_paste/${TIMESTAMP}}"

# --- 视频参数 ---
NUM_FRAMES="${NUM_FRAMES:-81}"
SEED="${SEED:-42}"
FPS="${FPS:-16}"

# --- VACE 参数 ---
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
CFG_SCALE="${CFG_SCALE:-7.5}"

# --- Canny / Mask 参数 ---
FILL_VALUE="${FILL_VALUE:-128}"
GROW_PIXELS="${GROW_PIXELS:-10}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"
CANNY_LOW="${CANNY_LOW:-50}"
CANNY_HIGH="${CANNY_HIGH:-150}"
BG_THRESHOLD="${BG_THRESHOLD:-240}"
BLUR_KSIZE="${BLUR_KSIZE:-3}"

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
echo "PVTT 批量推理：Canny 首帧粘贴（实验 10）"
echo "============================================================"
echo "数据集目录:      $DATASET_ROOT"
echo "输出目录:        $OUTPUT_ROOT"
echo "模型大小:        $MODEL_SIZE"
echo "帧数:            $NUM_FRAMES"
echo "Canny 阈值:      low=$CANNY_LOW, high=$CANNY_HIGH"
echo "填充值:          $FILL_VALUE"
echo "Mask 处理:       GrowMask(${GROW_PIXELS}px) + BlockifyMask(${BLOCK_SIZE}px)"
echo "采样模式:        ${SAMPLED:-1}"
echo "============================================================"

python3 "$PVTT_DIR/run_pvtt_canny_paste.py" \
    --dataset_root "$DATASET_ROOT" \
    --output_root "$OUTPUT_ROOT" \
    --model_size "$MODEL_SIZE" \
    --num_frames $NUM_FRAMES \
    --seed $SEED \
    --fps $FPS \
    --num_inference_steps $NUM_INFERENCE_STEPS \
    --cfg_scale $CFG_SCALE \
    --fill_value $FILL_VALUE \
    --grow_pixels $GROW_PIXELS \
    --block_size $BLOCK_SIZE \
    --canny_low $CANNY_LOW \
    --canny_high $CANNY_HIGH \
    --bg_threshold $BG_THRESHOLD \
    --blur_ksize $BLUR_KSIZE \
    --start_idx $START_IDX \
    $MAX_TASKS_FLAG \
    $SKIP_FLAG \
    $TASK_IDS_FLAG

EXIT_CODE=$?
echo ""
echo "推理完成，退出码: $EXIT_CODE"
exit $EXIT_CODE
