#!/bin/bash
# =============================================================================
# 1:1 复刻 ComfyUI VACE 商品替换工作流
#
# 对应 Python: baseline/wan2.1-vace/exp_comfyui_baseline.py
#
# 此实验复刻 ComfyUI 工作流的全部关键配置：
#   - 参考图去背景（BiRefNetUltra → rembg birefnet-general）
#   - 输入帧 mask 区域中性色填充
#   - Mask 预处理：GrowMask(10px) + BlockifyMask(32px)
#   - LightX2V 蒸馏 LoRA (rank32, strength=1.0)
#   - 单步采样 (steps=1)
#   - CFG=6.0, CFG_Star=5.0
#
# 与 ComfyUI 的唯一差异：
#   - 模型精度 bf16（而非 fp8 量化），对生成质量影响极小
#   - 推理框架 DiffSynth-Studio（而非 ComfyUI + kijai WanVideoWrapper）
#
# 所有参数均可通过环境变量覆盖：
#   LORA_PATH=/path/to/lora.safetensors bash scripts/run_comfyui_baseline.sh
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

# 模型大小：ComfyUI 工作流使用 14B，这里默认也用 14B
if [ -z "$WAN_VACE_MODEL_SIZE" ]; then
    export WAN_VACE_MODEL_SIZE="14B"
fi
MODEL_SIZE="$WAN_VACE_MODEL_SIZE"

# 时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/experiments/results/${MODEL_SIZE}/comfyui_baseline/${TIMESTAMP}}"

# 视频参数
WIDTH="${WIDTH:-480}"
HEIGHT="${HEIGHT:-848}"
NUM_FRAMES="${NUM_FRAMES:-49}"
SEED="${SEED:-42}"
FPS="${FPS:-15}"
PROMPT="${PROMPT:-yellow rubber duck toy, product display, studio lighting}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-low quality, blurry, deformed}"
REFERENCE_IMAGE="${REFERENCE_IMAGE:-ref_rubber_duck.png}"

# ---- ComfyUI 工作流对齐参数（核心差异点） ----

# 采样步数：ComfyUI 使用 1 步（蒸馏 LoRA 支持）
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-1}"

# CFG：ComfyUI 使用 cfg=6.0
CFG_SCALE="${CFG_SCALE:-6.0}"

# CFG Star：ComfyUI 使用 cfg_star=5.0
EMBEDDED_CFG_SCALE="${EMBEDDED_CFG_SCALE:-5.0}"

# 中性色填充值
FILL_VALUE="${FILL_VALUE:-128}"

# Mask 处理参数（与 ComfyUI GrowMask + BlockifyMask 一致）
GROW_PIXELS="${GROW_PIXELS:-10}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"

# LightX2V 蒸馏 LoRA 路径
LORA_PATH="${LORA_PATH:-$PROJECT_DIR/models/LightX2V/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors}"
LORA_STRENGTH="${LORA_STRENGTH:-1.0}"

# 参考图去背景模型（birefnet-general 对应 ComfyUI BiRefNetUltra）
REMBG_MODEL="${REMBG_MODEL:-birefnet-general}"

# ---- 打印配置 ----
echo "============================================"
echo "ComfyUI VACE 工作流 1:1 复刻实验"
echo "============================================"
echo "模型:              $MODEL_SIZE"
echo "样本:              $SAMPLE_NAME"
echo "分辨率:            ${WIDTH}x${HEIGHT}"
echo "帧数:              $NUM_FRAMES"
echo "种子:              $SEED"
echo "采样步数:          $NUM_INFERENCE_STEPS"
echo "CFG:               $CFG_SCALE"
echo "CFG Star:          $EMBEDDED_CFG_SCALE"
echo "填充值:            $FILL_VALUE"
echo "Mask 膨胀像素:     $GROW_PIXELS"
echo "Mask 网格大小:     $BLOCK_SIZE"
echo "LoRA 路径:         $LORA_PATH"
echo "LoRA 强度:         $LORA_STRENGTH"
echo "去背景模型:        $REMBG_MODEL"
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

# ---- LoRA 检查 ----
if [ -f "$LORA_PATH" ]; then
    echo "[OK] LoRA 文件存在: $(basename $LORA_PATH)"
else
    echo "[警告] LoRA 文件不存在: $LORA_PATH"
    echo "       蒸馏 LoRA 是 ComfyUI 工作流的核心组件，缺失将显著影响结果。"
    echo "       下载方法:"
    echo "         huggingface-cli download Kijai/WanVideo_comfy \\"
    echo "           Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors \\"
    echo "           --local-dir $PROJECT_DIR/models/LightX2V"
    echo ""
fi

# ---- 依赖检查 ----
python3 -c "import rembg" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[警告] rembg 未安装，参考图去背景步骤将被跳过。"
    echo "       安装方法: pip install rembg[gpu]"
fi

# ---- 运行实验 ----
echo ""
echo "开始运行实验..."
mkdir -p "$OUTPUT_DIR"

python3 "$BASELINE_DIR/exp_comfyui_baseline.py" \
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
    --embedded_cfg_scale $EMBEDDED_CFG_SCALE \
    --fill_value $FILL_VALUE \
    --grow_pixels $GROW_PIXELS \
    --block_size $BLOCK_SIZE \
    --lora_path "$LORA_PATH" \
    --lora_strength $LORA_STRENGTH \
    --rembg_model "$REMBG_MODEL"

# ---- 汇总 ----
echo ""
echo "============================================"
echo "实验完成！"
echo "输出目录: $OUTPUT_DIR"
echo "============================================"
