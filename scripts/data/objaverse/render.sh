#!/bin/bash

# Blender可执行文件路径（确保替换为你自己的路径）
BLENDER_EXEC=/mnt/public/yuchen.xi/app/blender-4.0.0-linux-x64/blender

# 要渲染的GLB文件路径
GLB_PATH="dataset/Objaverse/glbs/000-000/000a3d9fa4ff4c888e71e698694eb0b0.glb"

# 输出目录
OUTPUT_DIR="dataset/Objaverse/rendered"

# OpenLRM的Blender渲染脚本路径
SCRIPT_PATH="scripts/data/objaverse/blender_script.py"

# 执行渲染
$BLENDER_EXEC -b -P "$SCRIPT_PATH" -- \
    --object_path "$GLB_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --engine "CYCLES" \
    --num_images 8 \
    --resolution 512 