#!/bin/bash

# 设置本地下载目录
TARGET_DIR="dataset/Objaverse-2"
DATASET_NAME="allenai/objaverse"

# 创建目标目录
mkdir -p "$TARGET_DIR"

# 下载数据集（避免使用软链接）
huggingface-cli download $DATASET_NAME \
  --repo-type dataset \
  --local-dir "$TARGET_DIR" \
  --local-dir-use-symlinks False \
  --resume-download

echo "✅ 数据集下载完成，保存在: $TARGET_DIR"