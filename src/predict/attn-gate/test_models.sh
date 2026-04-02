#!/bin/bash

# Test script for MoE gate predictor
# Tests all models except Phi-tiny-MoE

PROJECT_ROOT="/data1/gx/MoE-predict"
DATASET_PATH="${PROJECT_ROOT}/dataset/processed/test/mmlu.jsonl"
SCRIPT_PATH="${PROJECT_ROOT}/src/predict/attn-gate/train_predictor.py"

# Models to test (excluding Phi-tiny-MoE)
MODELS=(
    "${PROJECT_ROOT}/models/DeepSeek-V2-Lite-Chat"
    # "${PROJECT_ROOT}/models/Mixtral-8x7B-v0.1"
    # "${PROJECT_ROOT}/models/Qwen3-30B-A3B"
)

# Test configuration
PATTERN="attn_gate"
BATCH_SIZE=1
MAX_SEQ_LENGTH=2048
BUFFER_SIZE_GB=4.0
MAX_SAMPLES=10

echo "=========================================="
echo "MoE Gate Predictor Model Testing"
echo "=========================================="
echo "Dataset: ${DATASET_PATH}"
echo "Pattern: ${PATTERN}"
echo "Batch size: ${BATCH_SIZE}"
echo "Max sequence length: ${MAX_SEQ_LENGTH}"
echo "Buffer size: ${BUFFER_SIZE_GB} GB"
echo "Max samples per model: ${MAX_SAMPLES}"
echo "=========================================="

# Check if dataset exists
if [ ! -f "${DATASET_PATH}" ]; then
    echo "Error: Dataset not found at ${DATASET_PATH}"
    exit 1
fi

# Check if script exists
if [ ! -f "${SCRIPT_PATH}" ]; then
    echo "Error: Training script not found at ${SCRIPT_PATH}"
    exit 1
fi

# Test each model
for MODEL_PATH in "${MODELS[@]}"; do
    MODEL_NAME=$(basename "${MODEL_PATH}")
    
    echo ""
    echo "=========================================="
    echo "Testing model: ${MODEL_NAME}"
    echo "Path: ${MODEL_PATH}"
    echo "=========================================="
    
    # Check if model exists
    if [ ! -d "${MODEL_PATH}" ]; then
        echo "Warning: Model directory not found at ${MODEL_PATH}"
        echo "Skipping this model..."
        continue
    fi
    
    # Run the test
    python "${SCRIPT_PATH}" \
        --model_path "${MODEL_PATH}" \
        --dataset_path "${DATASET_PATH}" \
        --pattern "${PATTERN}" \
        --batch_size "${BATCH_SIZE}" \
        --max_seq_length "${MAX_SEQ_LENGTH}" \
        --buffer_size_gb "${BUFFER_SIZE_GB}" \
        --max_samples "${MAX_SAMPLES}"
    
    EXIT_CODE=$?
    
    if [ ${EXIT_CODE} -eq 0 ]; then
        echo "✓ Model ${MODEL_NAME} test completed successfully"
    else
        echo "✗ Model ${MODEL_NAME} test failed with exit code ${EXIT_CODE}"
    fi
    
    echo ""
    echo "Waiting 5 seconds before next test..."
    sleep 5
done

echo ""
echo "=========================================="
echo "All model tests completed!"
echo "=========================================="
