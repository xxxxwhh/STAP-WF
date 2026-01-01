#!/bin/bash
# 改进FG-GAN训练示例脚本 - 支持后台运行

# 训练参数配置
DATA_PATH="E:/PythonFile/wfzoo/data/DF19"
GENERATOR_G1="/data_2_mnt/xuwenhai/wfzoo/src/checkpoints/generator/best_generator.pth"
PROXY_RF="/data_2_mnt/xuwenhai/wfzoo/src/checkpoints/best_model.pth"
CHECKPOINTS="./checkpoints/fggan_improved"

# 日志文件配置
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="./logs"
TRAIN_LOG="${LOG_DIR}/fggan_train_${TIMESTAMP}.log"
GEN_ADV_LOG="${LOG_DIR}/fggan_generate_${TIMESTAMP}.log"

# 创建日志目录
mkdir -p "$LOG_DIR"

echo "========================================"
echo "FG-GAN 后台训练脚本"
echo "========================================"
echo "训练日志: $TRAIN_LOG"
echo "生成日志: $GEN_ADV_LOG"
echo "========================================"

# 训练 (后台运行)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始训练FG-GAN..."
nohup python src/run_fggan.py \
    --data-path "$DATA_PATH" \
    --mon-classes 95 \
    --mon-inst 1000 \
    --generator-g1 "$GENERATOR_G1" \
    --proxy-rf "$PROXY_RF" \
    --checkpoints "$CHECKPOINTS" \
    --batch-size 32 \
    --epochs 100 \
    --lr0 0.0002 \
    --n-critic 5 \
    --workers 4 \
    --use-gpu \
    --gpu 0 \
    --save-interval 10 \
    --seed 42 \
    > "$TRAIN_LOG" 2>&1 &

TRAIN_PID=$!
echo "训练进程已启动 (PID: $TRAIN_PID)"
echo "查看训练日志: tail -f $TRAIN_LOG"
echo "停止训练: kill $TRAIN_PID"

# 等待训练完成
echo "等待训练完成..."
wait $TRAIN_PID
TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "❌ 训练失败 (退出码: $TRAIN_EXIT_CODE)"
    echo "查看日志: cat $TRAIN_LOG"
    exit 1
fi

echo "✅ 训练完成!"

# 生成对抗样本 (后台运行)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始生成对抗样本..."
nohup python src/generate_adversarial.py \
    --data-path "$DATA_PATH" \
    --mon-classes 95 \
    --mon-inst 100 \
    --fggan-model "$CHECKPOINTS/fggan_final.pth" \
    --generator-g1 "$GENERATOR_G1" \
    --proxy-rf "$PROXY_RF" \
    --output-dir "./adversarial_samples" \
    --num-samples 1000 \
    --use-gpu \
    --gpu 0 \
    > "$GEN_ADV_LOG" 2>&1 &

GEN_PID=$!
echo "生成进程已启动 (PID: $GEN_PID)"
echo "查看生成日志: tail -f $GEN_ADV_LOG"
echo "停止生成: kill $GEN_PID"

# 等待生成完成
echo "等待对抗样本生成完成..."
wait $GEN_PID
GEN_EXIT_CODE=$?

if [ $GEN_EXIT_CODE -ne 0 ]; then
    echo "❌ 对抗样本生成失败 (退出码: $GEN_EXIT_CODE)"
    echo "查看日志: cat $GEN_ADV_LOG"
    exit 1
fi

echo "✅ 对抗样本生成完成!"
echo "========================================"
echo "所有任务完成!"
echo "训练日志: $TRAIN_LOG"
echo "生成日志: $GEN_ADV_LOG"
echo "========================================"
