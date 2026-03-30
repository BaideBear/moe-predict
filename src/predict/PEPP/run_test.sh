#!/bin/bash

# PEPP Predictor 评估脚本
# Checkpoint路径包含数据集和模型信息

set -e

# 配置参数

MODEL_PATH="/data1/gx/MoE-predict/models/Mixtral-8x7B-v0.1"
CHECKPOINT_PATH="/data1/gx/MoE-predict/predict_models/PEPP/checkpoints_Mixtral-8x7B-v0.1_mmlu/predictor_final.pt"
DATASET_PATH="/data1/gx/MoE-predict/dataset/processed/test/mmlu.jsonl"

# 从路径中提取模型名称和数据集名称
MODEL_NAME=$(basename "$MODEL_PATH")
DATASET_NAME=$(basename "$DATASET_PATH" .jsonl)

# 评估参数
PATTERN="attn_gate"
BUFFER_SIZE_GB=4.0
BATCH_SIZE=16
MAX_SEQ_LENGTH=2048
TOP_K=4
MAX_SAMPLES=""  # 空字符串表示不限制，评估所有数据
DEVICE="cuda"

# 打印配置信息
echo "=========================================="
echo "PEPP Predictor 评估"
echo "=========================================="
echo "模型路径: $MODEL_PATH"
echo "模型名称: $MODEL_NAME"
echo "Checkpoint路径: $CHECKPOINT_PATH"
echo "数据集路径: $DATASET_PATH"
echo "数据集名称: $DATASET_NAME"
echo "=========================================="
echo ""

# 进入脚本目录
cd /data1/gx/MoE-predict/src/predict/PEPP

# 运行评估
python test_predictor.py \
    --model_path "$MODEL_PATH" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --dataset_path "$DATASET_PATH" \
    --pattern "$PATTERN" \
    --buffer_size_gb "$BUFFER_SIZE_GB" \
    --batch_size "$BATCH_SIZE" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --top_k "$TOP_K" \
    --device "$DEVICE" \
    ${MAX_SAMPLES:+--max_samples "$MAX_SAMPLES"}

echo ""
echo "=========================================="
echo "评估完成！"
echo "=========================================="
