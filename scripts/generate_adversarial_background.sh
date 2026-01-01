#!/bin/bash
# 生成对抗样本脚本 (完全后台运行)
# 在训练完成后单独运行此脚本

# 参数配置
DATA_PATH="/data_2_mnt/xuwenhai/wfzoo/raw-data-50-1000_cell"
GENERATOR_G1="/data_2_mnt/xuwenhai/wfzoo/src/checkpoints/generator_df_ds_2000/best_generator.pth"
PROXY_RF="/data_2_mnt/xuwenhai/wfzoo/src/checkpoints/rf_best_model_df_ds.pth"
CHECKPOINTS="/data_2_mnt/xuwenhai/wfzoo/checkpoints/fggan_rf_df_ds_noise_20251229_165233/"

# 目标攻击模型路径
MODEL_RF="/data_2_mnt/xuwenhai/wfzoo/src/checkpoints/rf_best_model_df_ds_adv.pth"
MODEL_DF="/data_2_mnt/xuwenhai/wfzoo/src/checkpoints/df_best_model_df_ds_adv.pth"
MODEL_TIKTAK="/data_2_mnt/xuwenhai/wfzoo/src/checkpoints/tiktok_best_model_df_ds_adv.pth"
MODEL_VARCNN="/data_2_mnt/xuwenhai/wfzoo/src/checkpoints/varcnn_best_model_df_ds_adv.pth"
MODEL_AWF="/data_2_mnt/xuwenhai/wfzoo/src/checkpoints/awf_best_model_df_ds_adv.pth"
MODEL_ARES="/data_2_mnt/xuwenhai/wfzoo/src/checkpoints/ares_best_model_df_ds_adv.pth"

# 评估哪些模型 (可修改)
TARGET_MODELS="rf df tiktak varcnn awf ares"  # 可选: rf, df, tiktak

# 可选: 指定使用哪个epoch的模型 (默认使用final)
MODEL_NAME="${1:-fggan_final.pth}"  # 可以传参: bash script.sh fggan_epoch_50.pth
FGGAN_MODEL="${CHECKPOINTS}/${MODEL_NAME}"

# 日志配置
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="./logs"
GEN_LOG="${LOG_DIR}/generate_adversarial_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/generate_adversarial.pid"

# 创建日志目录
mkdir -p "$LOG_DIR"

echo "========================================"
echo "FG-GAN 对抗样本生成与评估 (后台运行)"
echo "========================================"
echo "模型路径: $FGGAN_MODEL"
echo "目标模型: $TARGET_MODELS"
echo "生成日志: $GEN_LOG"
echo "PID文件: $PID_FILE"
echo "========================================"

# 检查模型文件是否存在
if [ ! -f "$FGGAN_MODEL" ]; then
    echo "❌ 错误: 模型文件不存在: $FGGAN_MODEL"
    echo "请确认训练已完成或指定正确的模型文件"
    echo "可用模型:"
    ls -lh "$CHECKPOINTS"/*.pth 2>/dev/null || echo "  (无模型文件)"
    exit 1
fi

echo "✅ 模型文件存在,开始生成..."
echo "将评估以下攻击模型: $TARGET_MODELS"
echo ""

# 完全后台运行
nohup python src/generate_adversarial.py \
    --data-path "$DATA_PATH" \
    --mon-classes 95 \
    --mon-inst 10 \
    --fggan-model "$FGGAN_MODEL" \
    --generator-g1 "$GENERATOR_G1" \
    --proxy-model "$PROXY_RF" \
    --target-models $TARGET_MODELS \
    --model-rf "$MODEL_RF" \
    --model-df "$MODEL_DF" \
    --model-varcnn "$MODEL_VARCNN" \
    --model-awf "$MODEL_AWF" \
    --model-ares "$MODEL_ARES" \
    --model-tiktak "$MODEL_TIKTAK" \
    --output-dir "./adversarial_samples" \
    --num-samples 1000 \
    --seq-length 2000 \
    --use-gpu \
    --gpu 3 \
    > "$GEN_LOG" 2>&1 &

GEN_PID=$!
echo $GEN_PID > "$PID_FILE"

echo "✅ 生成进程已启动 (PID: $GEN_PID)"
echo ""
echo "常用命令:"
echo "  查看实时日志: tail -f $GEN_LOG"
echo "  查看进程状态: ps -p $GEN_PID"
echo "  停止生成: kill $GEN_PID  或  kill \$(cat $PID_FILE)"
echo "  强制停止: kill -9 $GEN_PID"
echo ""
echo "生成完成后,结果保存在:"
echo "  ./adversarial_samples/adversarial_samples.npz  - 对抗样本TAM"
echo "  ./adversarial_samples/traces/*.cell            - 对抗trace文件"
echo "  ./adversarial_samples/evaluation.json         - 评估结果(ASR+开销)"
echo ""
echo "evaluation.json包含:"
echo "  - ASR: 各模型的攻击成功率"
echo "  - Overhead: 数据包开销统计(dummy包数量、相对开销、时间开销等)"
echo ""
echo "========================================"
