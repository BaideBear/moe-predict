#!/bin/bash

# Test script for MoE gate predictor evaluation
# This script tests the evaluation module with trained checkpoints

export CUDA_VISIBLE_DEVICES=0,2
PROJECT_ROOT="/data1/gx/MoE-predict"
# MODEL_NAME="Qwen3-30B-A3B"
# MODEL_NAME="Mixtral-8x7B-Instruct-v0.1"
MODEL_NAME="DeepSeek-V2-Lite-Chat"
DATA_NAME="mmlu"
# DATA_NAME="wikitext"
DATASET_PATH="${PROJECT_ROOT}/dataset/processed/test/${DATA_NAME}.jsonl"
SCRIPT_PATH="${PROJECT_ROOT}/src/predict/attn-gate/test_predictor.py"

# Model configuration
MODEL_PATH="${PROJECT_ROOT}/models/${MODEL_NAME}"
PATTERN="attn_gate"
MODEL_TYPE="simple_mlp"
BATCH_SIZE=1
MAX_SEQ_LENGTH=2048
BUFFER_SIZE_GB=2.0
MAX_SAMPLES=12000

# Evaluation configuration
EPOCHS=1
EVAL_BATCH_SIZE=3
TOP_K_VALUES="1,2,6"
NUM_ACTIVE_EXPERTS=6
USE_WANDB=true

# Checkpoint configuration
CHECKPOINT_NAME="mmlu"
CHECKPOINT_ID="70000"
CHECKPOINT_FILE="${PROJECT_ROOT}/predict_models/attn-gate/${MODEL_NAME}/${CHECKPOINT_NAME}/predictor_sample_${CHECKPOINT_ID}.pt"

echo "=========================================="
echo "MoE Gate Predictor Evaluation Test"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "Dataset: ${DATASET_PATH}"
echo "Pattern: ${PATTERN}"
echo "Sampling batch size: ${BATCH_SIZE}"
echo "Max sequence length: ${MAX_SEQ_LENGTH}"
echo "Buffer size: ${BUFFER_SIZE_GB} GB"
echo "Max samples per epoch: ${MAX_SAMPLES}"
echo "Epochs: ${EPOCHS}"
echo "Eval batch size: ${EVAL_BATCH_SIZE}"
echo "Top-k values: ${TOP_K_VALUES}"
echo "Num active experts: ${NUM_ACTIVE_EXPERTS}"
echo "Use wandb: ${USE_WANDB}"
echo "Checkpoint file: ${CHECKPOINT_FILE}"
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
    echo "Error: Evaluation script not found at ${SCRIPT_PATH}"
    exit 1
fi

# Check if checkpoint exists
if [ ! -f "${CHECKPOINT_FILE}" ]; then
    echo "Error: Checkpoint not found at ${CHECKPOINT_FILE}"
    echo "Available checkpoints in ${CHECKPOINT_DIR}:"
    ls -lh "${CHECKPOINT_DIR}" 2>/dev/null || echo "  Directory does not exist"
    exit 1
fi

# Run evaluation test
echo ""
echo "Starting evaluation test..."
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
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --top_k_values "${TOP_K_VALUES}" \
    --num_active_experts "${NUM_ACTIVE_EXPERTS}" \
    --load_checkpoint "${CHECKPOINT_FILE}" \
    --model_type "${MODEL_TYPE}" \
    --use_wandb \
    --wandb_project "moe-gate-predictor-eval" \
    --wandb_run_name "${MODEL_NAME}-${CHECKPOINT_NAME}"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ Evaluation test completed successfully"
    echo ""
    echo "Results logged to wandb project: moe-gate-predictor-eval"
else
    echo "✗ Evaluation test failed with exit code ${EXIT_CODE}"
fi
echo "=========================================="

exit ${EXIT_CODE}
