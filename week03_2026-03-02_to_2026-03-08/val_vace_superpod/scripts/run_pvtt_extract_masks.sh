#!/bin/bash
# ===========================================================================
# PVTT 数据集掩码提取：GroundingDINO + SAM2
#
# 从 easy_new.json 中的所有视频提取商品掩码序列。
# 需要先安装：pip install groundingdino-py sam2 opencv-python-headless
#
# 用法：
#   bash scripts/run_pvtt_extract_masks.sh
#
#   # 仅处理指定视频
#   VIDEO_NAMES="0001-handfan1,0002-handfan2" bash scripts/run_pvtt_extract_masks.sh
#
#   # 自定义模型路径
#   SAM2_CHECKPOINT=/path/to/sam2.pt bash scripts/run_pvtt_extract_masks.sh
# ===========================================================================

set -e

# HuggingFace 镜像（国内加速）
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# --- 路径配置 ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PVTT_DIR="$PROJECT_DIR/pvtt_evaluation"
DATASET_ROOT="$PROJECT_DIR/samples/pvtt_evaluation_datasets"

# --- 模型路径（根据你的服务器环境修改） ---
# SAM2
# SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-/home/hxiaoap/models/sam2.1_hiera_large.pt}"
# SAM2_CONFIG="${SAM2_CONFIG:-/home/hxiaoap/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml}"
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-/data/xiaohongrui/val_vace_superpod/models/sam2.1_hiera_large.pt}"
SAM2_CONFIG="${SAM2_CONFIG:-/data/xiaohongrui/packages/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml}"

# GroundingDINO
# 后端选择：transformers（推荐，无需编译）或 source（需源码安装）
# 使用 transformers 后端时 GDINO_CONFIG 可留空
GDINO_BACKEND="${GDINO_BACKEND:-}"  # 留空=自动检测, 可选: transformers / source
# GDINO_CONFIG="${GDINO_CONFIG:-/home/hxiaoap/models/GroundingDINO_SwinT_OGC.py}"
# GDINO_CHECKPOINT="${GDINO_CHECKPOINT:-/home/hxiaoap/models/groundingdino_swint_ogc.pth}"
GDINO_CONFIG="${GDINO_CONFIG:-/data/xiaohongrui/val_vace_superpod/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py}"
GDINO_CHECKPOINT="${GDINO_CHECKPOINT:-/data/xiaohongrui/val_vace_superpod/models/groundingdino_swint_ogc.pth}"

# --- 检测参数 ---
BOX_THRESHOLD="${BOX_THRESHOLD:-0.3}"
TEXT_THRESHOLD="${TEXT_THRESHOLD:-0.25}"
MAX_FRAMES="${MAX_FRAMES:-81}"

# --- 可选：仅处理指定视频 ---
VIDEO_NAMES_FLAG=""
if [ -n "$VIDEO_NAMES" ]; then
    VIDEO_NAMES_FLAG="--video_names $VIDEO_NAMES"
fi

# --- 是否跳过已有 mask ---
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SKIP_FLAG=""
if [ "$SKIP_EXISTING" = "1" ]; then
    SKIP_FLAG="--skip_existing"
else
    SKIP_FLAG="--no_skip_existing"
fi

# ===========================================================================
echo "============================================================"
echo "PVTT 掩码提取：GroundingDINO + SAM2"
echo "============================================================"
echo "数据集目录:      $DATASET_ROOT"
echo "SAM2 权重:       $SAM2_CHECKPOINT"
echo "SAM2 配置:       $SAM2_CONFIG"
echo "GDINO 配置:      $GDINO_CONFIG"
echo "GDINO 权重:      $GDINO_CHECKPOINT"
echo "Box 阈值:        $BOX_THRESHOLD"
echo "Text 阈值:       $TEXT_THRESHOLD"
echo "跳过已有:        $SKIP_EXISTING"
echo "============================================================"

# 前置检查
if [ ! -f "$SAM2_CHECKPOINT" ]; then
    echo "警告: SAM2 权重不存在: $SAM2_CHECKPOINT"
    echo "请设置 SAM2_CHECKPOINT 环境变量指向正确路径"
fi

if [ ! -f "$GDINO_CHECKPOINT" ]; then
    echo "警告: GroundingDINO 权重不存在: $GDINO_CHECKPOINT"
    echo "请设置 GDINO_CHECKPOINT 环境变量指向正确路径"
fi

# --- 构造 GroundingDINO 参数 ---
GDINO_FLAGS="--gdino_checkpoint $GDINO_CHECKPOINT"
if [ -n "$GDINO_BACKEND" ]; then
    GDINO_FLAGS="$GDINO_FLAGS --gdino_backend $GDINO_BACKEND"
fi
if [ -n "$GDINO_CONFIG" ] && [ -f "$GDINO_CONFIG" ]; then
    GDINO_FLAGS="$GDINO_FLAGS --gdino_config $GDINO_CONFIG"
fi

# 运行
python3 "$PVTT_DIR/extract_masks.py" \
    --dataset_root "$DATASET_ROOT" \
    --sam2_checkpoint "$SAM2_CHECKPOINT" \
    --sam2_config "$SAM2_CONFIG" \
    $GDINO_FLAGS \
    --max_frames "$MAX_FRAMES" \
    --box_threshold "$BOX_THRESHOLD" \
    --text_threshold "$TEXT_THRESHOLD" \
    $SKIP_FLAG \
    $VIDEO_NAMES_FLAG

EXIT_CODE=$?
echo ""
echo "掩码提取完成，退出码: $EXIT_CODE"
exit $EXIT_CODE
