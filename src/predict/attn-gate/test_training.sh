#!/bin/bash

# Test script for MoE gate predictor training
# This script tests the training module with minimal samples

export CUDA_VISIBLE_DEVICES=0,2
PROJECT_ROOT="/data1/gx/MoE-predict"
MODEL_NAME="Qwen3-30B-A3B"
# MODEL_NAME="Mixtral-8x7B-Instruct-v0.1"
# MODEL_NAME="DeepSeek-V2-Lite-Chat"
DATA_NAME="mmlu"
# DATA_NAME="wikitext"
DATASET_PATH="${PROJECT_ROOT}/dataset/processed/train/${DATA_NAME}.jsonl"
SCRIPT_PATH="${PROJECT_ROOT}/src/predict/attn-gate/train_predictor.py"

# Loss function configuration
LOSS_TYPE="ranking_aware_bce"
# LOSS_TYPE="weighted_bce"
TOP_K=8
LAMBDA_RANKING=0.3
MARGIN=0.1
WEIGHT_TOP10=3.0
WEIGHT_TOP11_30=1.5
WEIGHT_OTHERS=0.5
TOP_N_FOR_RANKING=10

# Test configuration
MODEL_PATH="${PROJECT_ROOT}/models/${MODEL_NAME}"
PATTERN="attn_gate"
BATCH_SIZE=1
MAX_SEQ_LENGTH=2048
BUFFER_SIZE_GB=2.0
MAX_SAMPLES=12000

# Training configuration
EPOCHS=3
TRAIN_BATCH_SIZE=15
LEARNING_RATE=1e-3
WEIGHT_DECAY=0.01
USE_WANDB=true
CHECKPOINT_DIR="${PROJECT_ROOT}/predict_models/attn-gate/${MODEL_NAME}/${DATA_NAME}-${LOSS_TYPE}-epoch3_5"
CHECKPOINT_INTERVAL=2000


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
echo "Loss type: ${LOSS_TYPE}"
echo "Top-k: ${TOP_K}"
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
    --checkpoint_interval "${CHECKPOINT_INTERVAL}" \
    --loss_type "${LOSS_TYPE}" \
    --top_k "${TOP_K}" \
    --lambda_ranking "${LAMBDA_RANKING}" \
    --margin "${MARGIN}" \
    --weight_top10 "${WEIGHT_TOP10}" \
    --weight_top11_30 "${WEIGHT_TOP11_30}" \
    --weight_others "${WEIGHT_OTHERS}" \
    --top_n_for_ranking "${TOP_N_FOR_RANKING}" \
    --use_wandb \
    --wandb_project "moe-gate-predictor" \
    --wandb_run_name "${MODEL_NAME}-${DATA_NAME}-${LOSS_TYPE}-epoch3_5" \
    --load_checkpoint "/data1/gx/MoE-predict/predict_models/attn-gate/Qwen3-30B-A3B/mmlu-ranking_aware_bce/predictor_sample_28000.pt"

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
