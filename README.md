# MoE门控函数预测项目

## 项目概述

本项目旨在对MoE模型的门控函数进行预测，通过采集推理过程中的隐藏层信息，实现对门控函数输出的预测能力。

## 项目结构

```
MoE-predict/
├── dataset/                    # 数据集目录
│   ├── raw/                   # 原始数据集
│   │   ├── mmlu/
│   │   └── gsm8k/
│   └── processed/             # 预处理后的数据
│       ├── train/            # 训练集（90%）
│       └── test/             # 测试集（10%）
├── models/                    # 模型参数目录
│   ├── Mixtral-8x7B-v0.1/
│   └── deepseek-v2-Lite/
├── predict_models/                    # 预测模型目录
│   ├── PEPP/                    # PEPP预测器目录
│       ├── checkpoints_${MODEL_NAME}_${DATASET_NAME}/
├── samples/                   # 采样数据目录
│   ├── Mixtral-8x7B-v0.1/
│   └── deepseek-v2-Lite/
├── src/                   # 脚本目录
│   ├── preprocess       # 数据集预处理
│   ├── sample            # 训练数据采样
│   ├── online_sample     # 在线采样模块
│   └── predict          # 门控函数预测
└── README.md
```

## 数据流程

### 1. 数据集预处理（preprocess）

**输入**: 原始数据集（MMLU、GSM8K等）

**输出**: 统一格式的训练文本结构

**说明**: 将原始数据集转换为可直接用于prefill推理的格式，按90%/10%比例划分训练集和测试集

### 2. 训练数据采样（sample）

**输入**: 预处理后的训练集

**输出**: 采集的隐藏层数据

**说明**: 在不同模型（Mixtral、DeepSeek）上进行推理，采集以下信息：
- 每层门控函数的输出
- 每层Attn层之前的隐藏层
- Token序列（token编码）
- 每层门控函数的输入

### 3. 门控函数预测（predict）

**输入**: 采样的隐藏层数据

**输出**: 门控函数预测结果

**说明**: 使用训练或特殊算法对门控函数输出进行预测，可直接在测试集上验证

### 4. 在线采样（online_sample）

**输入**: 预处理后的训练集

**输出**: 实时采集的激活值（通过共享缓冲区）

**说明**: 在线采样模块提供了一种高效的实时采样方案，解决了传统采样方法中IO开销大和存储空间占用高的问题：

- **零磁盘IO**：所有激活值保持在GPU HBM中，避免CPU-GPU数据传输
- **异步并行**：采样和训练可以同时进行，提高整体效率
- **内存管理**：环形缓冲区实现内存复用，防止OOM
- **可扩展性**：支持多种数据格式模式

**支持的数据模式**：
1. `attn_gate`: Attention层前激活值 + Gate Logits
2. `gate_input`: Gate输入激活值 + Gate Logits
3. `token_gate`: Token序列 + Gate Logits

**使用示例**：
```python
from online_sample import create_buffer, extract_model_config, OnlineSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

model_config = extract_model_config(model, "Mixtral-8x7B-v0.1", 2048)

buffer = create_buffer(
    model_config=model_config,
    pattern="attn_gate",
    buffer_size_gb=4.0,
    device="cuda"
)

sampler = OnlineSampler(
    model=model,
    tokenizer=tokenizer,
    dataset_path="dataset/processed/train/mmlu.jsonl",
    buffer=buffer,
    pattern="attn_gate",
    batch_size=1,
    max_seq_length=2048
)

sampler.start()
```

详细文档请参考 [src/online_sample/README.md](src/online_sample/README.md)

## 数据格式定义

### 1. 预处理数据格式

**文件路径**: `dataset/processed/{train|test}/{dataset_name}.jsonl`

**格式**: JSON Lines，每行一个样本

**数据结构**:
```json
{
  "text": "完整的文本内容",
  "metadata": {
    "source": "mmlu",
    "category": "mathematics"
  }
}
```

**说明**:
- `text`: 原始文本内容
- `metadata`: 元数据信息（可选）

### 2. 采样数据格式

**文件路径**: `samples/{model_name}/{dataset_name}_sample.pt`

**格式**: PyTorch格式（.pt）

**数据结构**:
```python
{
  "gate_outputs": tensor,      # shape: (num_samples, num_layers, num_experts), dtype: torch.bfloat16
  "gate_inputs": tensor,       # shape: (num_samples, num_layers, hidden_dim), dtype: torch.bfloat16
  "attn_hidden_states": tensor, # shape: (num_samples, num_layers, seq_len, hidden_dim), dtype: torch.bfloat16
  "tokens": tensor,            # shape: (num_samples, seq_len), dtype: torch.int32
  "sample_indices": tensor     # shape: (num_samples,), dtype: torch.int32
}
```

**说明**:
- `gate_outputs`: 门控函数输出，每层对每个专家的权重
- `gate_inputs`: 门控函数输入，每层的隐藏层状态
- `attn_hidden_states`: Attention层之前的隐藏层状态
- `tokens`: 对应的token序列
- `sample_indices`: 原始数据集中的样本索引
- 所有tensor保持原始精度（bfloat16或int32）

## 数据存储规范

- 数组数据使用PyTorch格式存储，保持原始精度（bfloat16）
- 文本数据使用JSON Lines格式，便于流式读取
- 采样数据按模型和数据集分别存储，便于管理和复用
- 所有数据文件命名遵循统一规范，便于自动化处理