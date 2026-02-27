# 环境配置

```sh
conda create -n wan22 python=3.10 -y
conda activate wan22

# Ensure torch >= 2.4.0
# If the installation of `flash_attn` fails, try installing the other packages first and install `flash_attn` last
pip install -r requirements.txt

# If you want to use CosyVoice to synthesize speech for Speech-to-Video Generation, please install requirements_s2v.txt additionally
pip install -r requirements_s2v.txt
```

```sh
pip install torch>=2.4.0 torchvision>=0.19.0 torchaudio --index-url https://download.pytorch.org/whl/cu121
# sudo apt update && sudo apt install -y gcc g++ nvcc build-essential python3-dev
# # 推荐方式：指定版本 + 禁用构建隔离（使用当前环境的 Torch）
# pip install flash-attn==2.8.3 --no-build-isolation
cat requirements.txt | grep -v -E '^torch|^torchvision|^torchaudio|^flash_attn' | xargs pip install
pip install peft easydict decord moviepy imageio librosa --no-cache-dir
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

| Models              | Download Links                                                                                                                              | Description |
|--------------------|---------------------------------------------------------------------------------------------------------------------------------------------|-------------|
| T2V-A14B    | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B)    | Text-to-Video MoE model, supports 480P & 720P |
| I2V-A14B    | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B)    | Image-to-Video MoE model, supports 480P & 720P |
| TI2V-5B     | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)     🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B)     | High-compression VAE, T2V+I2V, supports 720P |
| S2V-14B     | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B)     🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-S2V-14B)     | Speech-to-Video model, supports 480P & 720P |
| Animate-14B | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B) 🤖 [ModelScope](https://www.modelscope.cn/models/Wan-AI/Wan2.2-Animate-14B)  | Character animation and replacement | |


```sh
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B

huggingface-cli download Wan-AI/Wan2.2-I2V-A14B --local-dir ./Wan2.2-I2V-A14B

huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./Wan2.2-TI2V-5B
```

# sbatch文件

```sh
#!/bin/bash
#SBATCH --job-name=wan_inference  # Create a short name for your job
#SBATCH --output=logs/wan_output_%j.log  # Log output file, saved in the logs directory (%j is the job ID)
#SBATCH --error=logs/wan_error_%j.log   # Error log file
#SBATCH --nodes=1                # Number of nodes
#SBATCH --gpus=1                 # Number of GPUs per node (only valid for large/normal partitions)
#SBATCH --time=01:00:00         # Total run time limit (HH:MM:SS)
#SBATCH --partition=normal  # Partition (large/normal/cpu) to submit to
#SBATCH --account=mscaisuperpod      # Required only for multiple projects

# Navigate to the project directory
cd /home/hxiaoap/Wan2.2
# Prepare log directory
mkdir -p logs

# Load environment
module purge                     # Clear inherited environment modules
module load Anaconda3/2023.09-0  # Load the required modules
module load cuda12.2/toolkit/12.2.2

# 环境在：/home/hxiaoap/.conda/envs/wan22
ENV_PATH="/home/hxiaoap/.conda/envs/wan22"
# 强制将环境的 bin 目录插入到 PATH 的最前面,这样当输入 python 时，系统只能看到你环境里的那个，看不到系统的
export PATH="$ENV_PATH/bin:$PATH"
# 为了保险，将库路径也加进去
export LD_LIBRARY_PATH="$ENV_PATH/lib:$LD_LIBRARY_PATH"
# 验证时刻
echo "当前 Python 路径: $(which python)"
echo "当前 Python 版本: $(python --version)"

# 4. 自动熔断机制：如果不是 Python 3.10，直接报错退出，不执行后面代码
if [[ "$(python --version)" != *"3.10"* ]]; then
    echo "❌ 错误：环境切换失败！当前依然是 Python $(python --version)"
    echo "请检查 ENV_PATH 变量是否正确指向了你的 Conda 环境目录。"
    exit 1
fi
echo "✅ 环境锁定成功：Python 3.10"


# echo ""
# echo "=== 安装/检查依赖 ==="
# # 安装 EasyDict (修复 ModuleNotFoundError)
# pip install easydict --no-cache-dir
# pip install peft decord moviepy imageio librosa --no-cache-dir

# # 安装 Flash-Attention (使用预编译包，跳过编译)
# echo "正在安装 Flash-Attention..."
# pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# 安装flash-attn（推荐--no-build-isolation避免编译依赖问题）
# pip install flash-attn==2.6.3 --no-build-isolation -i https://pypi.tuna.tsinghua.edu.cn/simple

echo ""
echo "=== Checking installed packages ==="
python - <<'PYCODE'
import torch, diffusers, transformers, accelerate
print(f"torch version        : {torch.__version__}")
print(f"diffusers version    : {diffusers.__version__}")
print(f"transformers version : {transformers.__version__}")
print(f"accelerate version   : {accelerate.__version__}")
PYCODE
echo "==================================="


# Execute inference command
# 记录开始时间
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo -e "\n $START_TIME 推理开始..." >> logs/wan_output_${SLURM_JOB_ID}.log

python generate.py  \
  --task t2v-A14B \
  --size 1280*720 \
  --ckpt_dir ./Wan2.2-T2V-A14B \
  --offload_model True \
  --convert_model_dtype \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."

# 记录结束时间
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "$END_TIME 推理结束" >> logs/wan_output_${SLURM_JOB_ID}.log

# 计算耗时（转换为时间戳计算差值）
START_TIMESTAMP=$(date -d "$START_TIME" +%s)
END_TIMESTAMP=$(date -d "$END_TIME" +%s)
DURATION=$((END_TIMESTAMP - START_TIMESTAMP))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))
echo "推理总耗时: ${HOURS}小时${MINUTES}分钟${SECONDS}秒" >> logs/wan_output_${SLURM_JOB_ID}.log


if [ $? -eq 0 ]; then
  # Output on success
  echo "Inference task completed."
else
  # Output on failure
  echo "Inference task failed! Check the error log: cat logs/wan_error_"$SLURM_JOB_ID".log"
fi
```

```sh
sed -i 's/\r$//' inference.sbatch
sbatch inference.sbatch
squeue -u hxiaoap
```