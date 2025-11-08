#!/bin/bash

# 设置Python不缓冲输出，以便立即看到日志
export PYTHONUNBUFFERED=1

# 1. 安装依赖
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# 2. 运行训练脚本
# 检查configs/base.yaml是否存在
if [ ! -f "configs/base.yaml" ]; then
    echo "Error: configs/base.yaml not found!"
    exit 1
fi

echo "Starting training..."
python src/train.py --config configs/base.yaml