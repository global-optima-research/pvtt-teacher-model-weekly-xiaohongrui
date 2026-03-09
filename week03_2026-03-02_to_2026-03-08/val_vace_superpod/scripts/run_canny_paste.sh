#!/bin/bash
# =============================================================================
# Canny 首帧粘贴（白线/黑线对比）+ 中性色填充 + GrowMask + BlockifyMask 商品替换实验
#
# 对应 Python: baseline/wan2.1-vace/exp_canny_paste.py
#
# 仅在视频首帧的 mask 区域嵌入参考图像的 Canny 边缘线条，
# 同一次运行中依次测试白色和黑色两种线条颜色。
#
# 所有参数均可通过环境变量覆盖：
#   CANNY_LOW=80 CANNY_HIGH=180 bash scripts/run_canny_paste.sh
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

# 模型大小
if [ -z "$WAN_VACE_MODEL_SIZE" ]; then
    export WAN_VACE_MODEL_SIZE="1.3B"
fi
MODEL_SIZE="$WAN_VACE_MODEL_SIZE"

# 时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/experiments/results/${MODEL_SIZE}/canny_paste/${TIMESTAMP}}"

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

# 中性色填充参数
FILL_VALUE="${FILL_VALUE:-128}"

# Mask 处理参数
GROW_PIXELS="${GROW_PIXELS:-10}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"

# Canny 参数
CANNY_LOW="${CANNY_LOW:-50}"
CANNY_HIGH="${CANNY_HIGH:-150}"
BG_THRESHOLD="${BG_THRESHOLD:-240}"
BLUR_KSIZE="${BLUR_KSIZE:-3}"

# rembg 前景分割（默认开启；设 NO_REMBG=1 可禁用，回退到亮度阈值去背景）
NO_REMBG="${NO_REMBG:-0}"
REMBG_FLAG=""
if [ "$NO_REMBG" = "1" ]; then
    REMBG_FLAG="--no_rembg"
fi

# ---- 打印配置 ----
echo "============================================"
echo "Canny 首帧粘贴（白线/黑线对比）商品替换实验"
echo "============================================"
echo "模型:            $MODEL_SIZE"
echo "样本:            $SAMPLE_NAME"
echo "分辨率:          ${WIDTH}x${HEIGHT}"
echo "帧数:            $NUM_FRAMES"
echo "种子:            $SEED"
echo "填充值:          $FILL_VALUE"
echo "Mask 膨胀像素:   $GROW_PIXELS"
echo "Mask 网格大小:   $BLOCK_SIZE"
echo "Canny 阈值:      low=$CANNY_LOW, high=$CANNY_HIGH"
echo "背景阈值:        $BG_THRESHOLD"
echo "高斯模糊核:      $BLUR_KSIZE"
echo "rembg 前景分割:  $([ "$NO_REMBG" = "1" ] && echo '禁用' || echo '启用')"
echo "变体:            白色 + 黑色（依次执行）"
echo ""

# ---- 环境检查 ----
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
else
    echo "警告: 未检测到 nvidia-smi"
fi

# ---- OpenCV 检查 ----
python3 -c "import cv2" 2>/dev/null || {
    echo "错误: 未安装 OpenCV。请执行: pip install opencv-python-headless"
    exit 1
}
echo "[OK] OpenCV 可用。"

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
echo "开始运行实验（白色 + 黑色两个变体）..."
mkdir -p "$OUTPUT_DIR"

python3 "$BASELINE_DIR/exp_canny_paste.py" \
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
    --fill_value $FILL_VALUE \
    --grow_pixels $GROW_PIXELS \
    --block_size $BLOCK_SIZE \
    --canny_low $CANNY_LOW \
    --canny_high $CANNY_HIGH \
    --bg_threshold $BG_THRESHOLD \
    --blur_ksize $BLUR_KSIZE \
    $REMBG_FLAG

# ---- 汇总 ----
echo ""
echo "============================================"
echo "实验完成！"
echo "输出目录: $OUTPUT_DIR"
echo "============================================"
