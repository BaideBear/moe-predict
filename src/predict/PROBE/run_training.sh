#!/bin/bash
# PROBE Predictor Training Script
# Before running, edit MODEL_PATH and DATASET_PATH below.

SOURCE_DIR="$(cd "$(dirname "$0")" && pwd)"

# ===== 配置参数（按需修改） =====
MODEL_PATH="/path/to/deepseek-ai/DeepSeek-V2-Lite"
DATASET_PATH="${SOURCE_DIR}/../../../dataset/processed/train"
CHECKPOINT_DIR="${SOURCE_DIR}/checkpoints"
BUFFER_SIZE_GB=4.0
BATCH_SIZE=1
MAX_SEQ_LENGTH=2048
NUM_EPOCHS=10
LEARNING_RATE=1e-3
WEIGHT_DECAY=1e-4
TOP_K_ROUTING=8
CHECKPOINT_INTERVAL=50
# =================================

echo "================================================"
echo "  PROBE Predictor Training"
echo "================================================"
echo "  Model:        ${MODEL_PATH}"
echo "  Dataset:      ${DATASET_PATH}"
echo "  Checkpoints:  ${CHECKPOINT_DIR}"
echo "  Buffer:       ${BUFFER_SIZE_GB} GB"
echo "  Epochs:       ${NUM_EPOCHS}"
echo "  LR:           ${LEARNING_RATE}"
echo "  Top-k:        ${TOP_K_ROUTING}"
echo "================================================"

python "${SOURCE_DIR}/train_predictor.py" \
    --model_path "${MODEL_PATH}" \
    --dataset_path "${DATASET_PATH}" \
    --buffer_size_gb ${BUFFER_SIZE_GB} \
    --batch_size ${BATCH_SIZE} \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --num_epochs ${NUM_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay ${WEIGHT_DECAY} \
    --top_k_routing ${TOP_K_ROUTING} \
    --checkpoint_interval ${CHECKPOINT_INTERVAL} \
    --save_dir "${CHECKPOINT_DIR}" \
    --device cuda
