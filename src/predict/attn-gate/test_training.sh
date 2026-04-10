#!/bin/bash

# Test script for MoE gate predictor training
# This script tests the training module with minimal samples

PROJECT_ROOT="/data1/gx/MoE-predict"
DATASET_PATH="${PROJECT_ROOT}/dataset/processed/test/mmlu.jsonl"
SCRIPT_PATH="${PROJECT_ROOT}/src/predict/attn-gate/train_predictor.py"

# Test configuration
MODEL_PATH="${PROJECT_ROOT}/models/DeepSeek-V2-Lite-Chat"
PATTERN="attn_gate"
BATCH_SIZE=1
MAX_SEQ_LENGTH=2048
BUFFER_SIZE_GB=2.0
MAX_SAMPLES=30

# Training configuration
EPOCHS=1
TRAIN_BATCH_SIZE=5
LEARNING_RATE=1e-4
WEIGHT_DECAY=0.01
USE_WANDB=false
CHECKPOINT_DIR="${PROJECT_ROOT}/src/predict/attn-gate/test_checkpoints"
CHECKPOINT_INTERVAL=20

echo "=========================================="
echo "MoE Gate Predictor Training Test"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "Dataset: ${DATASET_PATH}"
echo "Pattern: ${PATTERN}"
echo "Sampling batch size: ${BATCH_SIZE}"
echo "Max sequence length: ${MAX_SEQ_LENGTH}"
echo "Buffer size: ${BUFFER_SIZE_GB} GB"
echo "Max samples per epoch: ${MAX_SAMPLES}"
echo "Epochs: ${EPOCHS}"
echo "Train batch size: ${TRAIN_BATCH_SIZE}"
echo "Learning rate: ${LEARNING_RATE}"
echo "Weight decay: ${WEIGHT_DECAY}"
echo "Use wandb: ${USE_WANDB}"
echo "Checkpoint dir: ${CHECKPOINT_DIR}"
echo "Checkpoint interval: ${CHECKPOINT_INTERVAL}"
echo "=========================================="

# Check if dataset exists
if [ ! -f "${DATASET_PATH}" ]; then
    echo "Error: Dataset not found at ${DATASET_PATH}"
    exit 1
fi

# Check if model exists
if [ ! -d "${MODEL_PATH}" ]; then
    echo "Error: Model not found at ${MODEL_PATH}"
    exit 1
fi

# Check if script exists
if [ ! -f "${SCRIPT_PATH}" ]; then
    echo "Error: Training script not found at ${SCRIPT_PATH}"
    exit 1
fi

# Create checkpoint directory
mkdir -p "${CHECKPOINT_DIR}"

# Run training test
echo ""
echo "Starting training test..."
echo ""

python "${SCRIPT_PATH}" \
    --model_path "${MODEL_PATH}" \
    --dataset_path "${DATASET_PATH}" \
    --pattern "${PATTERN}" \
    --batch_size "${BATCH_SIZE}" \
    --max_seq_length "${MAX_SEQ_LENGTH}" \
    --buffer_size_gb "${BUFFER_SIZE_GB}" \
    --max_samples "${MAX_SAMPLES}" \
    --epochs "${EPOCHS}" \
    --train_batch_size "${TRAIN_BATCH_SIZE}" \
    --learning_rate "${LEARNING_RATE}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --checkpoint_interval "${CHECKPOINT_INTERVAL}"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ Training test completed successfully"
    echo ""
    echo "Checkpoints saved:"
    ls -lh "${CHECKPOINT_DIR}" 2>/dev/null || echo "  No checkpoints found"
else
    echo "✗ Training test failed with exit code ${EXIT_CODE}"
fi
echo "=========================================="

exit ${EXIT_CODE}
