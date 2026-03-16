#!/bin/bash
# ===========================================================================
# PVTT 数据集批量推理：VACE I2V + FFGo 首帧参考
#
# 输入 = ref img（左侧产品 RGBA + 右侧去物体首帧）
#       + mask seq（首帧全0 + 后续帧全1）
#       + prompt
#
# 这是 I2V 模式：首帧固定，后续帧全部由 VACE 从头生成。
#
# 用法：
#   bash scripts/run_pvtt_ffgo_i2v.sh
#
#   # 仅处理采样的 23 个任务
#   SAMPLED=1 bash scripts/run_pvtt_ffgo_i2v.sh
#
#   # 自定义物体移除方式
#   INPAINT_METHOD=neutral_fill bash scripts/run_pvtt_ffgo_i2v.sh
#
#   # 添加场景转换前缀
#   TRANSITION_PREFIX="the scene transitions to" bash scripts/run_pvtt_ffgo_i2v.sh
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

# --- 输出目录（带时间戳） ---
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/experiments/results/$MODEL_SIZE/pvtt_ffgo_i2v/$TIMESTAMP}"

# --- 视频参数 ---
NUM_FRAMES="${NUM_FRAMES:-81}"
SEED="${SEED:-42}"
FPS="${FPS:-16}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
CFG_SCALE="${CFG_SCALE:-7.5}"

# --- FFGo 特有参数 ---
INPAINT_METHOD="${INPAINT_METHOD:-lama}"   # lama(推荐) / cv2 / neutral_fill
FILL_VALUE="${FILL_VALUE:-128}"
DILATE_PIXELS="${DILATE_PIXELS:-3}"        # mask 膨胀像素（物体移除前，默认 3px 仅消除边缘锯齿）
LAMA_CKPT="${LAMA_CKPT:-}"                 # 本地 big-lama.pt 路径（避免从 GitHub 下载）
TRANSITION_PREFIX="${TRANSITION_PREFIX:-The camera view suddenly changes.}"
DISCARD_FRAMES="${DISCARD_FRAMES:-4}"      # 丢弃前 N 帧转场帧（参考 FFGo Fc=4）
USE_REF_IMAGE="${USE_REF_IMAGE:-1}"        # 是否同时传 reference image

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

REF_IMAGE_FLAG=""
if [ "$USE_REF_IMAGE" = "0" ]; then
    REF_IMAGE_FLAG="--no_ref_image"
fi

LAMA_CKPT_FLAG=""
if [ -n "$LAMA_CKPT" ]; then
    LAMA_CKPT_FLAG="--lama_ckpt $LAMA_CKPT"
fi

# ===========================================================================
echo "============================================================"
echo "PVTT 批量推理：VACE I2V + FFGo 首帧参考"
echo "============================================================"
echo "数据集目录:    $DATASET_ROOT"
echo "输出目录:      $OUTPUT_ROOT"
echo "模型大小:      $MODEL_SIZE"
echo "帧数:          $NUM_FRAMES"
echo "物体移除方式:  $INPAINT_METHOD (膨胀 ${DILATE_PIXELS}px)"
echo "LaMa ckpt:     ${LAMA_CKPT:-自动下载}"
echo "丢弃转场帧:    前 ${DISCARD_FRAMES} 帧"
echo "使用 ref img:  $USE_REF_IMAGE"
echo "转场前缀:      $TRANSITION_PREFIX"
echo "采样模式:      ${SAMPLED:-1}"
echo "============================================================"

python3 "$PVTT_DIR/run_pvtt_ffgo_i2v.py" \
    --dataset_root "$DATASET_ROOT" \
    --output_root "$OUTPUT_ROOT" \
    --model_size "$MODEL_SIZE" \
    --num_frames "$NUM_FRAMES" \
    --seed "$SEED" \
    --fps "$FPS" \
    --num_inference_steps "$NUM_INFERENCE_STEPS" \
    --cfg_scale "$CFG_SCALE" \
    --inpaint_method "$INPAINT_METHOD" \
    --fill_value "$FILL_VALUE" \
    --dilate_pixels "$DILATE_PIXELS" \
    --discard_frames "$DISCARD_FRAMES" \
    --transition_prefix "$TRANSITION_PREFIX" \
    --start_idx "$START_IDX" \
    $MAX_TASKS_FLAG \
    $SKIP_FLAG \
    $TASK_IDS_FLAG \
    $REF_IMAGE_FLAG \
    $LAMA_CKPT_FLAG

EXIT_CODE=$?
echo ""
echo "推理完成，退出码: $EXIT_CODE"
exit $EXIT_CODE
