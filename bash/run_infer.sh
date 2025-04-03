#!/bin/bash

# 设置 Hugging Face 镜像（可选）
export HF_ENDPOINT=https://hf-mirror.com

# 设置推理参数
EXPORT_VIDEO=true
EXPORT_MESH=true
INFER_CONFIG="./configs/infer-l.yaml"
MODEL_NAME="zxhezexin/openlrm-mix-large-1.1"
IMAGE_INPUT="assets/sample_input/381743595798_.pic_hd.jpg"

# 执行推理命令
python -m openlrm.launch infer.lrm \
  --infer "$INFER_CONFIG" \
  model_name="$MODEL_NAME" \
  image_input="$IMAGE_INPUT" \
  export_video="$EXPORT_VIDEO" \
  export_mesh="$EXPORT_MESH"