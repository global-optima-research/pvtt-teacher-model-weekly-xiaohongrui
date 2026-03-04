#!/bin/bash
# =============================================================================
# 中性色填充 + 精确 Mask 商品替换实验
#
# 对应 Python: baseline/wan2.1-vace/exp_neutral_fill.py
#
# 所有参数均可通过环境变量覆盖：
#   FILL_VALUE=128 SEED=123 bash scripts/run_neutral_fill.sh
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

OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/experiments/results/${MODEL_SIZE}/neutral_fill/${TIMESTAMP}}"

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

# 中性色填充专有参数
FILL_VALUE="${FILL_VALUE:-128}"

# ---- 打印配置 ----
echo "============================================"
echo "中性色填充 + 精确 Mask 商品替换实验"
echo "============================================"
echo "模型:      $MODEL_SIZE"
echo "样本:      $SAMPLE_NAME"
echo "分辨率:    ${WIDTH}x${HEIGHT}"
echo "帧数:      $NUM_FRAMES"
echo "种子:      $SEED"
echo "填充值:    $FILL_VALUE"
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

python3 "$BASELINE_DIR/exp_neutral_fill.py" \
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
    --fill_value $FILL_VALUE

# ---- 汇总 ----
echo ""
echo "============================================"
echo "实验完成！"
echo "输出目录: $OUTPUT_DIR"
echo "============================================"
