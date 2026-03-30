#!/bin/bash

# Example 1: Sample Mixtral-8x7B-v0.1 on mmlu dataset
# python src/sample/sample.py \
#     --model-name Mixtral-8x7B-v0.1 \
#     --dataset-name mmlu \
#     --model-path /data1/gx/MoE-predict/models/Mixtral-8x7B-v0.1 \
#     --batch-size 4 \
#     --max-seq-length 2048 \
#     --chunk-size 100

# Example 2: Sample DeepSeek-V2-Lite on gsm8k dataset
# python src/sample/sample.py \
#     --model-name DeepSeek-V2-Lite-Chat \
#     --dataset-name gsm8k \
#     --model-path /data1/gx/MoE-predict/models/DeepSeek-V2-Lite-Chat \
#     --batch-size 1 \
#     --max-seq-length 2048 \
#     --chunk-size 10

# Example 3: Sample Qwen3-30B-A3B on wikitext dataset
python src/sample/sample.py \
    --model-name Qwen3-30B-A3B \
    --dataset-name wikitext \
    --model-path /data1/gx/MoE-predict/models/Qwen3-30B-A3B \
    --batch-size 1 \
    --max-seq-length 2048 \
    --chunk-size 10


# Example 4: Use custom input path
# python src/sample/sample.py \
#     --model-name Mixtral-8x7B-v0.1 \
#     --dataset-name mmlu \
#     --model-path /path/to/model \
#     --input-path /path/to/custom/dataset.jsonl \
#     --output-dir /path/to/output \
#     --batch-size 2 \
#     --max-seq-length 4096 \
#     --chunk-size 50
