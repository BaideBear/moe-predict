# MoE模型采样工具

本工具用于从MoE模型中采样隐藏层状态，用于门控函数预测。

## 功能特性

- 支持多种MoE模型架构（Mixtral、DeepSeek等）
- 自动检测MoE层和门控函数
- 高性能推理，使用Flash Attention 2
- 自动GPU分配，充分利用所有可用GPU
- 分块保存机制，防止内存溢出
- 符合项目规范的输出格式

## 使用方法

### 基本用法

```bash
python src/sample/sample.py \
    --model-name Mixtral-8x7B-v0.1 \
    --dataset-name mmlu \
    --model-path /path/to/model
```

### 参数说明

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--model-name` | 是 | - | 模型名称（如Mixtral-8x7B-v0.1） |
| `--dataset-name` | 是 | - | 数据集名称（如mmlu） |
| `--model-path` | 是 | - | 模型路径 |
| `--input-path` | 否 | `{project_dir}/dataset/processed/train/{dataset_name}.jsonl` | 输入数据集路径 |
| `--output-dir` | 否 | `/data1/gx/MoE-predict/samples` | 输出目录 |
| `--batch-size` | 否 | 1 | 推理批次大小 |
| `--max-seq-length` | 否 | 2048 | 最大序列长度 |
| `--chunk-size` | 否 | 100 | 每个chunk的样本数 |
| `--trust-remote-code` | 否 | False | 是否信任远程代码 |

### 输出格式

采样结果将保存为`.pt`文件（PyTorch格式），包含以下字段：

- `gate_outputs`: 门控函数输出，形状为 `(num_samples, num_layers, num_experts)`，dtype: `torch.bfloat16`
- `gate_inputs`: 门控函数输入，形状为 `(num_samples, num_layers, hidden_dim)`，dtype: `torch.bfloat16`
- `attn_hidden_states`: Attention层之前的隐藏层状态，形状为 `(num_samples, num_layers, seq_len, hidden_dim)`，dtype: `torch.bfloat16`
- `tokens`: Token序列，形状为 `(num_samples, seq_len)`，dtype: `torch.int32`
- `sample_indices`: 原始数据集中的样本索引，形状为 `(num_samples,)`，dtype: `torch.int32`

输出文件路径：`{output_dir}/{model_name}/{dataset_name}_sample.pt`

**读取示例**：
```python
import torch
data = torch.load("samples/Mixtral-8x7B-v0.1/mmlu_sample.pt")
gate_outputs = data["gate_outputs"]
```

## 示例

### 对Mixtral模型在MMLU数据集上采样

```bash
python src/sample/sample.py \
    --model-name Mixtral-8x7B-v0.1 \
    --dataset-name mmlu \
    --model-path /data1/gx/MoE-predict/models/Mixtral-8x7B-v0.1 \
    --batch-size 1 \
    --chunk-size 100
```

### 对DeepSeek模型在GSM8K数据集上采样

```bash
python src/sample/sample.py \
    --model-name deepseek-v2-Lite-Chat \
    --dataset-name gsm8k \
    --model-path /data1/gx/MoE-predict/models/deepseek-v2-Lite-Chat \
    --batch-size 1 \
    --chunk-size 50
```

## 性能优化建议

1. **批次大小**: 根据GPU显存调整`--batch-size`，更大的批次可以提高吞吐量
2. **序列长度**: 根据数据集特点调整`--max-seq-length`，避免过长序列占用过多显存
3. **分块大小**: 根据可用内存调整`--chunk-size`，更大的chunk可以减少I/O操作
4. **Flash Attention**: 代码默认使用Flash Attention 2，如果不可用会自动降级

## 注意事项

1. 确保模型路径正确且包含所有必要文件
2. 输入数据集必须是JSON Lines格式，每行包含`text`字段
3. 采样过程可能需要较长时间，建议使用`nohup`或`screen`运行
4. 输出文件会覆盖同名文件，请谨慎操作

## 故障排除

### 内存不足

- 减小`--batch-size`
- 减小`--max-seq-length`
- 减小`--chunk-size`

### 模型加载失败

- 检查模型路径是否正确
- 添加`--trust-remote-code`参数
- 确保有足够的磁盘空间

### MoE层检测失败

- 确保模型是MoE架构
- 检查模型结构是否被修改
- 查看错误日志获取详细信息
