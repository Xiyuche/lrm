#!/bin/bash

# === 路径设置 ===
ACC_CONFIG=./configs/accelerate-train.yaml
TRAIN_CONFIG=./configs/train-sample.yaml

# === 启动训练 ===
accelerate launch --config_file $ACC_CONFIG -m openlrm.launch train.lrm --config $TRAIN_CONFIG