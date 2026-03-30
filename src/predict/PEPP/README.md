# PEPP Predictor Module

## 概述

PEPP (Pre-Attention Expert Prediction) 预测器模块实现了对MoE模型门控函数的预测功能。该模块通过在线采样获取Attention层之前的隐藏层状态，预测当前层的Gate输出，从而实现对专家路由的提前预测。

## 核心特性

- **在线采样训练**：使用OnlineSampler实时采集激活值进行训练
- **零磁盘IO**：所有数据保持在GPU HBM中，避免CPU-GPU数据传输
- **异步并行**：采样和训练可以同时进行，提高整体效率
- **自定义损失函数**：实现了论文中的Ranking-aware multi-label classification损失
- **灵活的模型结构**：支持不同维度的输入和专家数量

## 模块结构

```
src/predict/PEPP/
├── __init__.py              # 模块初始化
├── model.py                 # 预测器模型定义
├── loss.py                  # 自定义损失函数
├── trainer.py               # 训练器实现
├── train_predictor.py       # 训练脚本
└── test_predictor.py        # 推理脚本
```

## 模型架构

预测器采用每层独立的预测器架构：

```
对于每一层l (l = 0, 1, ..., L-1):
    Input (hidden_dim) → Linear(2048) → BatchNorm1d → GELU → Dropout(0.1) → Linear(num_experts) → Output

总共有L个独立的预测器，每个预测器负责预测对应层的专家路由
```

**关键设计**：
- 每层都有独立的预测器，共享相同的架构但参数不同
- 预测时需要指定层索引：`predict_model(inputs, layer_idx)`
- 训练时每层的预测器独立更新

## 损失函数

使用论文中的Ranking-aware multi-label classification损失，包含：

1. **Weighted Binary Cross Entropy (L\_WBCE)**：对不同排名的专家赋予不同权重
   - Top-10专家：权重3.0
   - Top-11到Top-30专家：权重1.5
   - 其他专家：权重0.5
2. **Pairwise Ranking Loss (L\_ranking)**：对Top-10专家进行成对排序损失

总损失：L\_total = L\_WBCE + λ \* L\_ranking

## 训练逻辑

### 核心设计原则

1. **每个样本只被训练一次**：Buffer采用FIFO队列，采样的数据被训练后即被丢弃
2. **训练batch size = 从buffer取出的样本数**：批量训练，提高效率
3. **每训练一个batch保存一次checkpoint**：确保训练安全，可以随时恢复

### 训练流程

```
┌─────────────────────────────────────────────────────┐
│  在线采样线程（后台）                              │
│  ┌───────────────────────────────────────────────────┐  │
│  │ 采样样本1 → 写入buffer                      │  │
│  │ 采样样本2 → 写入buffer                      │  │
│  │ 采样样本3 → 写入buffer                      │  │
│  │ ...                                      │  │
│  │ 采样样本N → 写入buffer                      │  │
│  └───────────────────────────────────────────────────┘  │
│                    ↓                                  │
│              ┌─────────┐                            │
│              │  Buffer  │ (FIFO队列，4GB)            │
│              └─────────┘                            │
│                    ↓                                  │
│  主训练线程（前台）                                │
│  ┌───────────────────────────────────────────────────┐  │
│  │ 从buffer读取1个样本 → 训练 → 累计50个样本后保存 │  │
│  │ 从buffer读取1个样本 → 训练 → 累计50个样本后保存 │  │
│  │ ...                                      │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Checkpoint保存策略

- **保存频率**：每训练50个样本保存一次
- **命名规则**：`predictor_sample_{global_sample_count}.pt`
- **最终checkpoint**：`predictor_final.pt`（训练结束时保存）

### 参数关系

| 参数 | 作用 |
|------|------|
| `BATCH_SIZE` | 采样批次大小（每次采样1个样本） |
| `CHECKPOINT_INTERVAL` | Checkpoint保存间隔（每训练多少个样本保存一次，默认50） |
| `MAX_SAMPLES_PER_EPOCH` | 每个epoch最大训练样本数（None=无限制） |

## 使用方法

### 训练预测器

```bash
cd /data1/gx/MoE-predict/src/predict/PEPP

python train_predictor.py \
    --model_path /path/to/Mixtral-8x7B-v0.1 \
    --dataset_path /data1/gx/MoE-predict/dataset/processed/train/mmlu.jsonl \
    --pattern attn_gate \
    --buffer_size_gb 4.0 \
    --batch_size 1 \
    --max_seq_length 2048 \
    --num_epochs 10 \
    --learning_rate 1e-3 \
    --weight_decay 1e-4 \
    --top_k_routing 8 \
    --lambda_rank 1.0 \
    --margin 0.1 \
    --save_dir ./checkpoints \
    --checkpoint_interval 50 \
    --device cuda
```

### 评估预测器

#### 使用脚本运行（推荐）

```bash
cd /data1/gx/MoE-predict/src/predict/PEPP
bash run_test.sh
```

#### 手动运行

```bash
cd /data1/gx/MoE-predict/src/predict/PEPP

python test_predictor.py \
    --model_path /path/to/Mixtral-8x7B-v0.1 \
    --checkpoint_path ./checkpoints/predictor_final.pt \
    --dataset_path /data1/gx/MoE-predict/dataset/processed/test/mmlu.jsonl \
    --pattern attn_gate \
    --buffer_size_gb 4.0 \
    --batch_size 1 \
    --max_seq_length 2048 \
    --top_k 8 \
    --max_samples 200 \
    --device cuda
```

## 参数说明

### 训练参数

| 参数                       | 类型    | 默认值           | 说明                                       |
| ------------------------ | ----- | ------------- | ---------------------------------------- |
| `--model_path`           | str   | 必需            | 基础模型路径                                   |
| `--dataset_path`         | str   | 必需            | 训练数据集路径                                  |
| `--pattern`              | str   | attn\_gate    | 数据模式（attn\_gate/gate\_input/token\_gate） |
| `--buffer_size_gb`       | float | 4.0           | 缓冲区大小（GB）                                |
| `--batch_size`           | int   | 1             | 采样批次大小                                   |
| `--max_seq_length`       | int   | 2048          | 最大序列长度                                   |
| `--num_epochs`           | int   | 10            | 训练轮数                                     |
| `--learning_rate`        | float | 1e-3          | 学习率                                      |
| `--weight_decay`         | float | 1e-4          | 权重衰减                                     |
| `--top_k_routing`        | int   | 8             | 路由选择的Top-K专家数                            |
| `--lambda_rank`          | float | 1.0           | 排序损失权重                                   |
| `--margin`               | float | 0.1           | 排序损失边界                                   |
| `--save_dir`             | str   | ./checkpoints | 检查点保存目录                                  |
| `--checkpoint_interval`  | int   | 50            | Checkpoint保存间隔（每训练多少个样本保存一次） |
| `--max_samples_per_epoch` | int   | None           | 每个epoch最大训练样本数（None=无限制，持续训练） |
| `--device`               | str   | cuda          | 设备类型                                     |

### 评估参数

| 参数                       | 类型    | 默认值        | 说明         |
| ------------------------ | ----- | ---------- | ---------- |
| `--model_path`           | str   | 必需         | 基础模型路径     |
| `--checkpoint_path`      | str   | 必需         | 预测器检查点路径   |
| `--dataset_path`         | str   | 必需         | 测试数据集路径    |
| `--pattern`              | str   | attn\_gate | 数据模式       |
| `--buffer_size_gb`       | float | 4.0        | 缓冲区大小（GB）  |
| `--batch_size`           | int   | 1          | 采样批次大小     |
| `--max_seq_length`       | int   | 2048       | 最大序列长度     |
| `--top_k`                | int   | 8          | 评估使用的Top-K |
| `--max_samples`          | int   | None       | 最大评估样本数    |
| `--device`               | str   | cuda       | 设备类型       |

## 评估指标

1. **Top-1 Accuracy**：预测排名第一的专家是否与真实排名第一的专家一致
2. **Top-K Hit Rate**：预测的Top-K专家中命中真实激活专家的比例

## 训练流程

1. 加载基础模型和tokenizer
2. 提取模型配置信息
3. 创建激活值缓冲区
4. 创建预测器模型
5. 启动在线采样器（后台运行）
6. 创建预测器训练器
7. 开始训练（从缓冲区读取数据）
8. 保存检查点
9. 停止采样器

## 推理流程

1. 加载基础模型和tokenizer
2. 提取模型配置信息
3. 创建激活值缓冲区
4. 加载预测器检查点
5. 启动在线采样器（后台运行）
6. 创建预测器评估器
7. 开始评估（从缓冲区读取数据）
8. 计算评估指标
9. 停止采样器

## 注意事项

1. **显存管理**：根据可用显存调整`buffer_size_gb`和`predictor_batch_size`
2. **数据模式**：确保训练和推理使用相同的`pattern`
3. **模型兼容性**：确保预测器模型的输入维度与基础模型的隐藏层维度一致
4. **检查点加载**：推理时需要加载训练时保存的检查点

## 示例输出

### 训练输出

```
============================================================
PEPP Predictor Training
============================================================

1. Loading model and tokenizer...
   Model loaded from: /path/to/Mixtral-8x7B-v0.1
   Model dtype: torch.bfloat16

2. Extracting model configuration...
   Model name: Mixtral-8x7B-v0.1
   Number of layers: 32
   Hidden dimension: 4096
   Number of experts: 8
   Max sequence length: 2048

...

Epoch 1/10
--------------------------------------------------
  Batch 10, Loss: 0.12345, Avg Loss: 0.12345, Buffer: 100 samples, 1.00 GB (25.0%)
  ...
Epoch 1 completed:
  Total batches: 100
  Average loss: 0.12345
  Checkpoint saved to: ./checkpoints/predictor_epoch_1.pt

...
```

### 评估输出

```
============================================================
PEPP Predictor Evaluation
============================================================

...

8. Starting evaluation...
============================================================
Starting evaluation...
  Collected 10 data chunks, 50000 tokens total

Processing 50000 tokens...
Predicting: 100%|████████████████████| 13/13 [00:05<00:00,  2.50it/s]

Computing metrics...

============================================================
Evaluation Results
============================================================
Total tokens evaluated: 50000
Total samples evaluated: 10
Top-1 Accuracy: 85.23%
Top-8 Hit Rate: 92.15%
============================================================
```

## 故障排除

### 问题1：显存不足 (OOM)

**解决方案**：

- 减小`buffer_size_gb`参数
- 减小`predictor_batch_size`参数
- 减小`batch_size`参数

### 问题2：训练速度慢

**解决方案**：

- 增大`predictor_batch_size`参数
- 增大`buffer_size_gb`参数
- 使用多个GPU进行并行训练

### 问题3：评估准确率低

**解决方案**：

- 增加训练轮数`num_epochs`
- 调整学习率`learning_rate`
- 调整损失函数参数`lambda_rank`和`margin`
- 使用更多的训练数据

## 扩展性

### 添加新的数据模式

1. 在`data_structures.py`中添加新的模式常量
2. 在`model.py`中更新模型结构
3. 在`trainer.py`中更新数据提取逻辑

### 自定义损失函数

在`loss.py`中实现自定义损失函数，并在`trainer.py`中使用。

## 版本历史

### 0.1.0 (2026-03-29)

- ✅ 初始版本
- ✅ 实现预测器模型
- ✅ 实现自定义损失函数
- ✅ 实现训练器和评估器
- ✅ 支持在线采样训练
- ✅ 支持attn\_gate模式
- ✅ 完整的文档和示例

## 许可证

本项目遵循项目的整体许可证。

## 联系方式

如有问题或建议，请通过以下方式联系：

- 提交Issue
- 发送邮件
- 参与讨论

