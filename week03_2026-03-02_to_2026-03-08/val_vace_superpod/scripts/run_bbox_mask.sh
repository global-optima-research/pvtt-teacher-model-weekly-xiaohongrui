#!/bin/bash
# =============================================================================
# Bbox Mask vs 精确 Mask 对比实验
#
# 对应 Python: baseline/wan2.1-vace/exp_bbox_mask.py
#
# 所有参数均可通过环境变量覆盖：
#   BBOX_MARGIN_LR=30 BBOX_MARGIN_TOP=50 bash scripts/run_bbox_mask.sh
# =============================================================================
set -e

# ---- HuggingFace 镜像（国内服务器加速） ----
export HF_ENDPOINT=https://hf-mirror.com

# ---- 路径设置 ----
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BASELINE_DIR="$PROJECT_DIR/baseline/wan2.1-vace"

# ---- 参数（均可通过环境变量覆盖） ----
SAMPLE_NAME="${SAMPLE_NAME:-teapot}"
SAMPLE_DIR="$PROJECT_DIR/samples/$SAMPLE_NAME"

# 模型大小（提前设置，用于构建输出路径）
if [ -z "$WAN_VACE_MODEL_SIZE" ]; then
    export WAN_VACE_MODEL_SIZE="1.3B"
fi
MODEL_SIZE="$WAN_VACE_MODEL_SIZE"

# 时间戳（防止重复运行时结果覆盖）
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/experiments/results/${MODEL_SIZE}/bbox_mask/${TIMESTAMP}}"

WIDTH="${WIDTH:-480}"
HEIGHT="${HEIGHT:-848}"
NUM_FRAMES="${NUM_FRAMES:-49}"
SEED="${SEED:-42}"
FPS="${FPS:-15}"
PROMPT="${PROMPT:-yellow rubber duck toy, product display, studio lighting}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-low quality, blurry, deformed}"
REFERENCE_IMAGE="${REFERENCE_IMAGE:-ref_rubber_duck.png}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
CFG_SCALE="${CFG_SCALE:-7.5}"

# Bbox 专有参数
BBOX_MARGIN_LR="${BBOX_MARGIN_LR:-20}"
BBOX_MARGIN_TOP="${BBOX_MARGIN_TOP:-20}"
BBOX_MARGIN_BOTTOM="${BBOX_MARGIN_BOTTOM:-20}"

# ---- 打印配置 ----
echo "============================================"
echo "Bbox Mask vs 精确 Mask 对比实验"
echo "============================================"
echo "模型:        $MODEL_SIZE"
echo "样本:        $SAMPLE_NAME"
echo "分辨率:      ${WIDTH}x${HEIGHT}"
echo "帧数:        $NUM_FRAMES"
echo "种子:        $SEED"
echo "Bbox 边距:   lr=$BBOX_MARGIN_LR, top=$BBOX_MARGIN_TOP, bottom=$BBOX_MARGIN_BOTTOM"
echo ""

# ---- 环境检查 ----
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
else
    echo "警告: 未检测到 nvidia-smi"
fi

# ---- 数据检查 ----
MISSING=0
for dir in "$SAMPLE_DIR/video_frames" "$SAMPLE_DIR/masks"; do
    if [ ! -d "$dir" ]; then
        echo "[缺失] $dir"
        MISSING=1
    fi
done
if [ ! -f "$SAMPLE_DIR/reference_images/$REFERENCE_IMAGE" ]; then
    echo "[缺失] $SAMPLE_DIR/reference_images/$REFERENCE_IMAGE"
    MISSING=1
fi
if [ $MISSING -eq 1 ]; then
    echo "错误: 缺少必要的输入数据，实验中止。"
    exit 1
fi
echo "[OK] 所有输入数据就绪。"

# ---- 运行实验 ----
echo ""
echo "开始运行实验..."
mkdir -p "$OUTPUT_DIR"

python3 "$BASELINE_DIR/exp_bbox_mask.py" \
    --sample_dir "$SAMPLE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --width $WIDTH \
    --height $HEIGHT \
    --num_frames $NUM_FRAMES \
    --seed $SEED \
    --fps $FPS \
    --prompt "$PROMPT" \
    --negative_prompt "$NEGATIVE_PROMPT" \
    --reference_image "$REFERENCE_IMAGE" \
    --model_size "$MODEL_SIZE" \
    --num_inference_steps $NUM_INFERENCE_STEPS \
    --cfg_scale $CFG_SCALE \
    --bbox_margin_lr $BBOX_MARGIN_LR \
    --bbox_margin_top $BBOX_MARGIN_TOP \
    --bbox_margin_bottom $BBOX_MARGIN_BOTTOM

# ---- 汇总 ----
echo ""
echo "============================================"
echo "实验完成！"
echo "输出目录: $OUTPUT_DIR"
echo "============================================"
