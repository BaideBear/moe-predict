#!/bin/bash

# Batch evaluation script for MoE gate predictor
# Runs multiple evaluation tasks sequentially

export CUDA_VISIBLE_DEVICES=0,2
PROJECT_ROOT="/data1/gx/MoE-predict"
SCRIPT_PATH="${PROJECT_ROOT}/src/predict/attn-gate/test_predictor.py"

# Fixed configuration
PATTERN="attn_gate"
BATCH_SIZE=1
MAX_SEQ_LENGTH=2048
BUFFER_SIZE_GB=2.0
MAX_SAMPLES=12000
EPOCHS=1
EVAL_BATCH_SIZE=3
USE_WANDB=true
WANDB_PROJECT="moe-gate-predictor-eval"

# Task list: each line is "MODEL_NAME|DATA_NAME|MODEL_TYPE|TOP_K_VALUES|NUM_ACTIVE_EXPERTS|CHECKPOINT_NAME|CHECKPOINT_ID"
TASKS=(
    "DeepSeek-V2-Lite-Chat|mmlu|lstm|1,2,6|6|mmlu-ce-lstm|70000"
    "DeepSeek-V2-Lite-Chat|mmlu|mlp_without_dropout|1,2,6|6|mmlu-mlp-without-dropout|70000"
    "DeepSeek-V2-Lite-Chat|mmlu|simple_mlp|1,2,6|6|mmlu-weighted_bce|70000"
    "DeepSeek-V2-Lite-Chat|mmlu|simple_mlp|1,2,6|6|mmlu-ranking_aware_bce|70000"
)

TOTAL_TASKS=${#TASKS[@]}
COMPLETED=0
FAILED=0

echo "=========================================="
echo "MoE Gate Predictor Batch Evaluation"
echo "=========================================="
echo "Total tasks: ${TOTAL_TASKS}"
echo "=========================================="

for TASK in "${TASKS[@]}"; do
    IFS='|' read -r MODEL_NAME DATA_NAME MODEL_TYPE TOP_K_VALUES NUM_ACTIVE_EXPERTS CHECKPOINT_NAME CHECKPOINT_ID <<< "${TASK}"

    MODEL_PATH="${PROJECT_ROOT}/models/${MODEL_NAME}"
    DATASET_PATH="${PROJECT_ROOT}/dataset/processed/test/${DATA_NAME}.jsonl"
    CHECKPOINT_FILE="${PROJECT_ROOT}/predict_models/attn-gate/${MODEL_NAME}/${CHECKPOINT_NAME}/predictor_sample_${CHECKPOINT_ID}.pt"
    WANDB_RUN_NAME="${MODEL_NAME}-${CHECKPOINT_NAME}"

    echo ""
    echo "=========================================="
    echo "Task $((COMPLETED + FAILED + 1))/${TOTAL_TASKS}"
    echo "=========================================="
    echo "Model: ${MODEL_NAME}"
    echo "Dataset: ${DATA_NAME}"
    echo "Model type: ${MODEL_TYPE}"
    echo "Top-k values: ${TOP_K_VALUES}"
    echo "Num active experts: ${NUM_ACTIVE_EXPERTS}"
    echo "Checkpoint: ${CHECKPOINT_NAME}/predictor_sample_${CHECKPOINT_ID}.pt"
    echo "=========================================="

    if [ ! -f "${DATASET_PATH}" ]; then
        echo "Error: Dataset not found at ${DATASET_PATH}"
        FAILED=$((FAILED + 1))
        continue
    fi

    if [ ! -d "${MODEL_PATH}" ]; then
        echo "Error: Model not found at ${MODEL_PATH}"
        FAILED=$((FAILED + 1))
        continue
    fi

    if [ ! -f "${CHECKPOINT_FILE}" ]; then
        echo "Error: Checkpoint not found at ${CHECKPOINT_FILE}"
        FAILED=$((FAILED + 1))
        continue
    fi

    echo "Starting evaluation..."
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
        --wandb_project "${WANDB_PROJECT}" \
        --wandb_run_name "${WANDB_RUN_NAME}"

    EXIT_CODE=$?

    echo ""
    if [ ${EXIT_CODE} -eq 0 ]; then
        echo "✓ Task completed: ${MODEL_NAME}-${CHECKPOINT_NAME}"
        COMPLETED=$((COMPLETED + 1))
    else
        echo "✗ Task failed (exit code ${EXIT_CODE}): ${MODEL_NAME}-${CHECKPOINT_NAME}"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=========================================="
echo "Batch Evaluation Summary"
echo "=========================================="
echo "Total: ${TOTAL_TASKS}"
echo "Completed: ${COMPLETED}"
echo "Failed: ${FAILED}"
echo "=========================================="

if [ ${FAILED} -gt 0 ]; then
    exit 1
fi
