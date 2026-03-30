#!/bin/bash

# PEPP Predictor 持续训练脚本
# Checkpoint路径包含数据集和模型信息

set -e
# 配置参数

# MODEL_PATH="/data1/gx/MoE-predict/models/Qwen3-30B-A3B"
MODEL_PATH="/data1/gx/MoE-predict/models/Mixtral-8x7B-v0.1"
# MODEL_PATH="/data1/gx/MoE-predict/models/DeepSeek-V2-Lite-Chat"
# MODEL_PATH="/data1/gx/MoE-predict/models/Phi-tiny-MoE-instruct"
DATASET_PATH="/data1/gx/MoE-predict/dataset/processed/train/mmlu.jsonl"

# 从路径中提取模型名称和数据集名称
MODEL_NAME=$(basename "$MODEL_PATH")
DATASET_NAME=$(basename "$DATASET_PATH" .jsonl)

# 创建包含模型和数据集信息的checkpoint目录
SAVE_DIR="/data1/gx/MoE-predict/predict_models/PEPP/checkpoints_${MODEL_NAME}_${DATASET_NAME}"

# 训练参数
PATTERN="attn_gate"
BUFFER_SIZE_GB=4.0
BATCH_SIZE=1
MAX_SEQ_LENGTH=2048
NUM_EPOCHS=1
LEARNING_RATE=1e-3
WEIGHT_DECAY=1e-4
TOP_K_ROUTING=8
LAMBDA_RANK=1.0
MARGIN=0.1
DEVICE="cuda"
CHECKPOINT_INTERVAL=200  # 每x个样本保存一次checkpoint
MAX_SAMPLES_PER_EPOCH=""  # 空字符串表示不限制，持续训练

# 打印配置信息
echo "=========================================="
echo "PEPP Predictor 持续训练"
echo "=========================================="
echo "模型路径: $MODEL_PATH"
echo "模型名称: $MODEL_NAME"
echo "数据集路径: $DATASET_PATH"
echo "数据集名称: $DATASET_NAME"
echo "Checkpoint目录: $SAVE_DIR"
echo "=========================================="
echo ""

# 进入脚本目录
cd /data1/gx/MoE-predict/src/predict/PEPP

# 运行训练（不设置max_batches_per_epoch，持续训练整个数据集）
python train_predictor.py \
    --model_path "$MODEL_PATH" \
    --dataset_path "$DATASET_PATH" \
    --pattern "$PATTERN" \
    --buffer_size_gb "$BUFFER_SIZE_GB" \
    --batch_size "$BATCH_SIZE" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --num_epochs "$NUM_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --weight_decay "$WEIGHT_DECAY" \
    --top_k_routing "$TOP_K_ROUTING" \
    --lambda_rank "$LAMBDA_RANK" \
    --margin "$MARGIN" \
    --save_dir "$SAVE_DIR" \
    --checkpoint_interval "$CHECKPOINT_INTERVAL" \
    --device "$DEVICE" \
    ${MAX_SAMPLES_PER_EPOCH:+--max_samples_per_epoch "$MAX_SAMPLES_PER_EPOCH"}

echo ""
echo "=========================================="
echo "训练完成！"
echo "Checkpoints保存在: $SAVE_DIR"
echo "=========================================="
