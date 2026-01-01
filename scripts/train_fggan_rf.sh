#!/bin/bash
# FG-GAN 使用RF模型作为代理训练脚本

# 训练参数配置
DATA_PATH="/data_2_mnt/xuwenhai/wfzoo/src/20000/"
GENERATOR_G1="/data_2_mnt/xuwenhai/wfzoo/src/checkpoints/generator/best_generator.pth"
PROXY_MODEL="/data_2_mnt/xuwenhai/wfzoo/src/checkpoints/best_model.pth"
PROXY_MODEL_TYPE="rf"  # 代理模型类型: 'rf' 或 'df'
CHECKPOINTS="/data_2_mnt/xuwenhai/wfzoo/checkpoints/fggan_rf"

# 日志文件配置
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="./logs"
TRAIN_LOG="${LOG_DIR}/fggan_rf_train_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/fggan_rf_train.pid"

# 创建日志目录
mkdir -p "$LOG_DIR"

echo "========================================"
echo "FG-GAN 使用RF代理模型训练"
echo "========================================"
echo "训练日志: $TRAIN_LOG"
echo "PID文件: $PID_FILE"
echo "========================================"

# 完全后台运行训练
nohup python src/run_fggan.py \
    --data-path "$DATA_PATH" \
    --mon-classes 100 \
    --mon-inst 100 \
    --generator-g1 "$GENERATOR_G1" \
    --proxy-model "$PROXY_MODEL" \
    --proxy-model-type "$PROXY_MODEL_TYPE" \
    --checkpoints "$CHECKPOINTS" \
    --batch-size 32 \
    --epochs 50 \
    --lr0 0.0002 \
    --n-critic 5 \
    --workers 4 \
    --use-gpu \
    --gpu 2 \
    --save-interval 10 \
    --seed 42 \
    --monitor-real-insertion \
    > "$TRAIN_LOG" 2>&1 &

TRAIN_PID=$!
echo $TRAIN_PID > "$PID_FILE"

echo "✅ 训练进程已启动 (PID: $TRAIN_PID)"
echo ""
echo "常用命令:"
echo "  查看实时日志: tail -f $TRAIN_LOG"
echo "  查看进程状态: ps -p $TRAIN_PID"
echo "  停止训练: kill $TRAIN_PID  或  kill \$(cat $PID_FILE)"
echo "  强制停止: kill -9 $TRAIN_PID"
echo ""
echo "========================================"