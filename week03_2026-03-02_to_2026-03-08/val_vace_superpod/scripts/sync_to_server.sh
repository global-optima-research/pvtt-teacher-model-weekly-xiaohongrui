#!/bin/bash
# 同步项目到 GPU 服务器

# ============================================
# 配置 - 根据实际情况修改
# ============================================
SERVER="5090"                    # SSH 别名或 user@host
REMOTE_DIR="/data/xuhao/pvtt-training-pipeline"

# ============================================
# 同步
# ============================================
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "Syncing to $SERVER:$REMOTE_DIR"
echo "Local: $LOCAL_DIR"
echo ""

# 排除临时文件，保留样本数据
rsync -avz --progress \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    --exclude='experiments/results/wan2.1-vace/' \
    "$LOCAL_DIR/" "$SERVER:$REMOTE_DIR/"

echo ""
echo "============================================"
echo "同步完成！"
echo "============================================"
echo ""
echo "在服务器上运行："
echo "  ssh $SERVER"
echo "  cd $REMOTE_DIR"
echo "  bash scripts/run_vace_test.sh"
