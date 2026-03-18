#!/bin/bash
# ===========================================================================
# PVTT 数据集批量推理：FFGo 原版模型（Wan2.2-I2V-A14B + LoRA）
#
# 使用 FFGO-Video-Customization 论文的原版模型和 LoRA adapter，
# 在 PVTT 数据集上进行首帧引导的视频生成测试。
#
# 前置条件：
#   1. FFGo 环境已安装（见 scripts/README.md 中的 FFGo 环境配置指南）
#   2. 模型已下载（Wan2.2-I2V-A14B + FFGo LoRA adapter）
#
# 用法：
#   bash scripts/run_pvtt_ffgo_original.sh
#
#   # 跑全部 199 个任务
#   SAMPLED=0 bash scripts/run_pvtt_ffgo_original.sh
# ===========================================================================

set -e

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# --- 路径配置 ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PVTT_DIR="$PROJECT_DIR/pvtt_evaluation"
DATASET_ROOT="$PROJECT_DIR/samples/pvtt_evaluation_datasets"
FFGO_ROOT="$PROJECT_DIR/FFGO-Video-Customization"

# --- 模型路径（默认使用 FFGO_ROOT 下的 Models 目录） ---
# MODEL_NAME="${MODEL_NAME:-$FFGO_ROOT/Models/Wan2.2-I2V-A14B}"
# LORA_LOW="${LORA_LOW:-$FFGO_ROOT/Models/Lora/10_LargeMixedDatset_wan_14bLow_f81_LongCaption_StartMatch_run_r128_a128_3obj_Longrun_B4/checkpoint-600.safetensors}"
# LORA_HIGH="${LORA_HIGH:-$FFGO_ROOT/Models/Lora/10_LargeMixedDatset_wan_14bHigh_f81_LongCaption_StartMatch_run_r128_a128_3obj_Longrun_B4/checkpoint-600.safetensors}"
# CONFIG_PATH="${CONFIG_PATH:-$FFGO_ROOT/VideoX-Fun/config/wan2.2/wan_civitai_i2v.yaml}"
MODEL_NAME="${MODEL_NAME:-/home/hxiaoap/val_vace_superpod/models/Wan-AI/Wan2.2-I2V-A14B}"
LORA_LOW="${LORA_LOW:-/home/hxiaoap/val_vace_superpod/models/Wan-AI/Lora/10_LargeMixedDatset_wan_14bLow_f81_LongCaption_StartMatch_run_r128_a128_3obj_Longrun_B4/checkpoint-600.safetensors}"
LORA_HIGH="${LORA_HIGH:-/home/hxiaoap/val_vace_superpod/models/Wan-AI/Lora/10_LargeMixedDatset_wan_14bHigh_f81_LongCaption_StartMatch_run_r128_a128_3obj_Longrun_B4/checkpoint-600.safetensors}"
CONFIG_PATH="${CONFIG_PATH:-/home/hxiaoap/val_vace_superpod/models/Wan-AI/wan_civitai_i2v.yaml}"

# --- 输出目录（带时间戳）---
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/experiments/results/ffgo_original/pvtt/$TIMESTAMP}"

# --- 生成参数 ---
# FFGo 原版使用 480x832 或 832x480（适合大部分 GPU）
# 如有 H200 可改为 720x1280 或 1280x720
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
VIDEO_LENGTH="${VIDEO_LENGTH:-81}"
SEED="${SEED:-42}"
FPS="${FPS:-16}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-6.0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"

# --- Inpainting ---
INPAINT_METHOD="${INPAINT_METHOD:-lama}"
DILATE_PIXELS="${DILATE_PIXELS:-3}"
LAMA_CKPT="${LAMA_CKPT:-}"

# --- 转场 ---
PROMPT_PREFIX="${PROMPT_PREFIX:-ad23r2 the camera view suddenly changes. }"
SKIP_TRANSITION_FRAMES="${SKIP_TRANSITION_FRAMES:-4}"

# --- 采样任务（每种产品 2 个，共 16 个；handfan 仅 2 个已全部包含） ---
SAMPLED_TASK_IDS="0016-bracelet1_to_bracelet3,0016-bracelet1_to_bracelet4,0021-earring1_to_earring3,0021-earring1_to_earring4,0006-handbag1_scene01_to_handbag3,0006-handbag1_scene01_to_handbag4,0026-necklace1_to_necklace3,0026-necklace1_to_necklace4,0012-purse1_to_purse3,0012-purse1_to_purse4,0003-sunglasses1_scene01_to_sunglasses3,0004-sunglasses2_to_sunglasses3,0031-watch1_to_watch3,0031-watch1_to_watch4"

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

LAMA_CKPT_FLAG=""
if [ -n "$LAMA_CKPT" ]; then
    LAMA_CKPT_FLAG="--lama_ckpt $LAMA_CKPT"
fi

# ===========================================================================
echo "============================================================"
echo "PVTT 批量推理：FFGo 原版模型（Wan2.2-I2V-A14B + LoRA）"
echo "============================================================"
echo "数据集目录:    $DATASET_ROOT"
echo "输出目录:      $OUTPUT_ROOT"
echo "FFGo 根目录:   $FFGO_ROOT"
echo "模型:          $MODEL_NAME"
echo "分辨率:        ${WIDTH}x${HEIGHT}"
echo "帧数:          $VIDEO_LENGTH"
echo "Prompt 前缀:   $PROMPT_PREFIX"
echo "采样模式:      ${SAMPLED:-1}"
echo "============================================================"

python3 "$PVTT_DIR/run_pvtt_ffgo_original.py" \
    --dataset_root "$DATASET_ROOT" \
    --output_root "$OUTPUT_ROOT" \
    --ffgo_root "$FFGO_ROOT" \
    --model_name "$MODEL_NAME" \
    --lora_low "$LORA_LOW" \
    --lora_high "$LORA_HIGH" \
    --config_path "$CONFIG_PATH" \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --video_length "$VIDEO_LENGTH" \
    --seed "$SEED" \
    --fps "$FPS" \
    --guidance_scale "$GUIDANCE_SCALE" \
    --num_inference_steps "$NUM_INFERENCE_STEPS" \
    --inpaint_method "$INPAINT_METHOD" \
    --dilate_pixels "$DILATE_PIXELS" \
    --prompt_prefix "$PROMPT_PREFIX" \
    --skip_transition_frames "$SKIP_TRANSITION_FRAMES" \
    --start_idx "$START_IDX" \
    $MAX_TASKS_FLAG \
    $SKIP_FLAG \
    $TASK_IDS_FLAG \
    $LAMA_CKPT_FLAG

EXIT_CODE=$?
echo ""
echo "推理完成，退出码: $EXIT_CODE"
exit $EXIT_CODE
