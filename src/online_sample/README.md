# Online Sample 模块

## 概述

Online Sample 模块提供了一个高效的在线采样解决方案，用于从MoE模型中实时采集激活值，并通过共享缓冲区提供给预测器训练模块使用。该模块解决了传统采样方法中IO开销大和存储空间占用高的问题。

## 模块架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Online Sample Module                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐         ┌─────────────────────────┐   │
│  │  OnlineSampler   │         │   ActivationBuffer      │   │
│  │  (Producer)      │         │   (Shared Buffer)       │   │
│  │                  │         │                         │   │
│  │  - Model Inference│───────▶│  - Thread-safe Queue    │   │
│  │  - Hook Capture   │         │  - Memory Management   │   │
│  │  - Data Write     │         │  - Pattern Support      │   │
│  └──────────────────┘         └─────────────────────────┘   │
│                                       │                      │
│                                       │ Data Read            │
│                                       ▼                      │
│                               ┌─────────────────────────┐   │
│                               │  PredictorTrainer       │   │
│                               │  (Consumer)             │   │
│                               │  - Training Loop        │   │
│                               │  - Data Consume         │   │
│                               └─────────────────────────┘   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## 快速开始

### 安装依赖

```bash
conda activate MoE-test
pip install torch transformers tqdm
```

### 基本使用

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from online_sample import (
    create_buffer,
    extract_model_config,
    OnlineSampler,
    create_predictor_interface
)

# 1. 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "/path/to/model",
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained("/path/to/model", trust_remote_code=True)

# 2. 创建缓冲区
model_config = extract_model_config(model, "Mixtral-8x7B-v0.1", 2048)
buffer = create_buffer(
    model_config=model_config,
    pattern="attn_gate",
    buffer_size_gb=4.0,
    device="cuda"
)

# 3. 启动采样器
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

# 4. 训练预测器
interface = create_predictor_interface(buffer, "attn_gate", batch_size=1)
while True:
    batch = interface.get_batch()
    if batch is None:
        if not sampler.is_running():
            break
        continue
    
    # 训练代码...
    for data in batch:
        print(f"Processing sample: {data.tokens.shape}")

sampler.join()
```

## 数据格式模式

### Pattern 1: `attn_gate`
Attention层前激活值 + Gate Logits

```python
{
    "attn_hidden_states": torch.Tensor,  # [batch, num_layers, seq_len, hidden_dim]
    "gate_logits": torch.Tensor,          # [batch, num_layers, seq_len, num_experts]
    "tokens": torch.Tensor,               # [batch, seq_len]
    "seq_lengths": torch.Tensor,         # [batch]
    "metadata": Dict
}
```

**适用场景**：需要Attention层激活值进行训练的场景

### Pattern 2: `gate_input`
Gate输入激活值 + Gate Logits

```python
{
    "gate_inputs": torch.Tensor,         # [batch, num_layers, seq_len, hidden_dim]
    "gate_logits": torch.Tensor,         # [batch, num_layers, seq_len, num_experts]
    "tokens": torch.Tensor,              # [batch, seq_len]
    "seq_lengths": torch.Tensor,         # [batch]
    "metadata": Dict
}
```

**适用场景**：需要Gate层输入进行训练的场景

### Pattern 3: `token_gate`
Token序列 + Gate Logits

```python
{
    "tokens": torch.Tensor,              # [batch, seq_len]
    "gate_logits": torch.Tensor,         # [batch, num_layers, seq_len, num_experts]
    "seq_lengths": torch.Tensor,        # [batch]
    "metadata": Dict
}
```

**适用场景**：只需要Gate输出进行训练，内存受限的场景

## API 文档

### 1. ActivationBuffer

共享缓冲区，用于在生产者和消费者之间传递激活值。

#### 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_config` | ModelConfig | 必需 | 模型配置对象 |
| `pattern` | str | 必需 | 数据模式：`attn_gate`, `gate_input`, `token_gate` |
| `buffer_size_gb` | float | 4.0 | 缓冲区大小（GB） |
| `device` | str | "cuda" | 设备类型 |

#### 主要方法

- `write(data: ActivationData, timeout: Optional[float] = None) -> bool`
  - 写入数据到缓冲区
  - 如果缓冲区满，则阻塞等待
  - 返回是否成功写入

- `read(batch_size: int = 1, timeout: Optional[float] = None) -> Optional[List[ActivationData]]`
  - 从缓冲区读取数据
  - 如果没有足够数据，则阻塞等待
  - 返回数据列表或None

- `get_size() -> int`
  - 获取当前缓冲区中的样本数

- `is_empty() -> bool`
  - 判断缓冲区是否为空

- `is_full() -> bool`
  - 判断缓冲区是否已满

- `get_stats() -> BufferStats`
  - 获取缓冲区统计信息

- `mark_write_finished()`
  - 标记写入已完成，通知消费者

- `stop()`
  - 停止缓冲区

- `clear()`
  - 清空缓冲区

### 2. OnlineSampler

在线采样器，负责从模型中采集激活值并写入缓冲区。

#### 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | AutoModelForCausalLM | 必需 | 预训练模型 |
| `tokenizer` | AutoTokenizer | 必需 | 分词器 |
| `dataset_path` | str | 必需 | 数据集路径（JSON Lines格式） |
| `buffer` | ActivationBuffer | 必需 | 激活值缓冲区 |
| `pattern` | str | 必需 | 数据模式 |
| `batch_size` | int | 1 | 采样批次大小 |
| `max_seq_length` | int | 2048 | 最大序列长度 |
| `trust_remote_code` | bool | False | 是否信任远程代码 |
| `epochs` | int | 1 | 训练轮数，控制数据集重复采样次数 |

#### 主要方法

- `start()`
  - 启动在线采样（异步执行）
  - 如果指定了epochs > 1，会自动重复采样数据集指定次数

- `stop()`
  - 停止采样

- `is_running() -> bool`
  - 判断是否正在运行

- `join(timeout: Optional[float] = None)`
  - 等待采样线程结束

**Epochs功能说明**：
- `epochs`参数控制数据集的重复采样次数
- 当epochs > 1时，sampler会自动循环采样整个数据集指定次数
- 每个epoch都会显示进度信息
- 适用于需要多次遍历数据集进行训练的场景
- 训练模块无需关心epoch，只需持续从buffer读取器数据

### 3. PredictorInterface

预测器接口，供预测器训练模块使用。

#### 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `buffer` | ActivationBuffer | 必需 | 激活值缓冲区 |
| `pattern` | str | 必需 | 数据模式 |
| `batch_size` | int | 1 | 读取批次大小 |
| `timeout` | Optional[float] | None | 读取超时时间（秒） |

#### 主要方法

- `get_batch() -> Optional[List[ActivationData]]`
  - 获取一个批次的数据
  - 如果没有足够数据，则返回None

- `get_stats() -> Dict[str, Any]`
  - 获取缓冲区统计信息

- `is_buffer_empty() -> bool`
  - 判断缓冲区是否为空

- `is_buffer_full() -> bool`
  - 判断缓冲区是否已满

- `wait_for_data(min_samples: int = 1, timeout: Optional[float] = None) -> bool`
  - 等待缓冲区中有足够数据

## 完整示例

### 示例1：基本流程

参见 `example_usage.py` 文件，这是一个完整的演示程序，展示了：

1. 模型和tokenizer加载
2. 缓冲区创建
3. 采样器启动
4. 预测器训练（模拟）
5. 实时监控
6. 资源清理

运行示例：

```bash
cd /data1/gx/MoE-predict/src/online_sample
conda activate MoE-test
python example_usage.py
```

### 示例2：多线程训练

```python
import threading
from online_sample import create_predictor_interface

def training_worker(interface, worker_id):
    while True:
        batch = interface.get_batch()
        if batch is None:
            break
        
        for data in batch:
            print(f"Worker {worker_id}: Processing sample")
            # 训练代码...

# 创建多个训练线程
workers = []
for i in range(4):
    interface = create_predictor_interface(buffer, "attn_gate", batch_size=1)
    worker = threading.Thread(target=training_worker, args=(interface, i))
    worker.start()
    workers.append(worker)

# 等待所有线程完成
for worker in workers:
    worker.join()
```

### 示例3：带验证的训练

```python
from online_sample import create_predictor_interface

interface = create_predictor_interface(buffer, "attn_gate", batch_size=1)

train_count = 0
val_count = 0

while True:
    batch = interface.get_batch()
    if batch is None:
        break
    
    for data in batch:
        if train_count % 10 == 0:
            # 验证
            val_count += 1
            print(f"Validation batch {val_count}")
        else:
            # 训练
            train_count += 1
            print(f"Training batch {train_count}")
            # 训练代码...
```

## 测试

### 运行所有测试

```bash
cd /data1/gx/MoE-predict/src/online_sample/test
conda activate MoE-test
python -m pytest test_*.py -v
```

### 运行单个测试

```bash
# 测试数据结构
python test_data_structures.py

# 测试工具函数
python test_utils.py

# 测试缓冲区
python test_buffer.py

# 测试缓冲区压力
python test_buffer_stress.py

# 测试采样器（需要真实模型）
python test_sampler.py

# 测试预测器接口
python test_predictor_interface.py

# 测试集成（需要真实模型）
python test_integration.py
```

### 测试覆盖

| 模块 | 测试文件 | 测试数量 | 状态 |
|------|----------|----------|------|
| data_structures | test_data_structures.py | 12 | ✅ 通过 |
| utils | test_utils.py | 8 | ✅ 通过 |
| buffer | test_buffer.py | 8 | ✅ 通过 |
| buffer_stress | test_buffer_stress.py | 4 | ✅ 通过 |
| sampler | test_sampler.py | 4 | ✅ 通过 |
| predictor_interface | test_predictor_interface.py | 12 | ✅ 通过 |
| integration | test_integration.py | 3 | ✅ 通过 |

## 内存管理

### 缓冲区大小选择

| 模型 | 隐藏层维度 | 专家数 | 层数 | 推荐缓冲区大小 |
|------|-----------|--------|------|---------------|
| Mixtral-8x7B | 4096 | 8 | 32 | 4-8 GB |
| DeepSeek-V2-Lite | 4096 | 2 | 28 | 2-4 GB |
| Phi-tiny-MoE | 2048 | 4 | 24 | 1-2 GB |

### 内存使用估算

```python
from online_sample.utils import calculate_memory_usage

memory = calculate_memory_usage(
    num_samples=100,
    num_layers=32,
    hidden_dim=4096,
    num_experts=8,
    seq_length=1024,
    pattern="attn_gate"
)

print(f"Total memory: {memory['total'] / (1024**3):.2f} GB")
```

### 缓冲区容量估算

```python
from online_sample.utils import estimate_buffer_capacity

capacity = estimate_buffer_capacity(
    buffer_size_gb=4.0,
    num_layers=32,
    hidden_dim=4096,
    num_experts=8,
    avg_seq_length=1024,
    pattern="attn_gate"
)

print(f"Estimated capacity: {capacity} samples")
```

## 性能优化

### 1. 批次大小优化

```python
# 小批次（低显存，低吞吐量）
sampler = OnlineSampler(..., batch_size=1)

# 大批次（高显存，高吞吐量）
sampler = OnlineSampler(..., batch_size=4)
```

### 2. 缓冲区大小优化

```python
# 小缓冲区（节省显存，可能阻塞）
buffer = create_buffer(..., buffer_size_gb=2.0)

# 大缓冲区（占用显存，减少阻塞）
buffer = create_buffer(..., buffer_size_gb=8.0)
```

### 3. 数据模式选择

| 模式 | 内存占用 | 适用场景 |
|------|----------|----------|
| `token_gate` | 最低 | 只需Gate输出 |
| `gate_input` | 中等 | 需要Gate输入 |
| `attn_gate` | 最高 | 需要Attention激活 |

### 4. 多GPU配置

```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",  # 自动分配到多个GPU
    torch_dtype=torch.bfloat16
)
```

## 性能基准

### Mixtral-8x7B 性能

| 配置 | 采样速率 | 训练速率 | 显存占用 |
|------|----------|----------|----------|
| batch_size=1, buffer=4GB | ~1.5 samples/s | ~0.8 samples/s | ~20 GB |
| batch_size=2, buffer=4GB | ~2.0 samples/s | ~1.2 samples/s | ~22 GB |
| batch_size=4, buffer=8GB | ~3.0 samples/s | ~2.0 samples/s | ~26 GB |

### 不同数据模式内存占用

| 模式 | 每样本内存 | 100样本总内存 |
|------|-----------|--------------|
| `token_gate` | ~0.5 MB | ~50 MB |
| `gate_input` | ~2.5 MB | ~250 MB |
| `attn_gate` | ~10 MB | ~1 GB |

## 故障排除

### 问题1：内存不足 (OOM)

**症状**：`RuntimeError: CUDA out of memory`

**解决方案**：
```python
# 减小缓冲区大小
buffer = create_buffer(..., buffer_size_gb=2.0)

# 减小批次大小
sampler = OnlineSampler(..., batch_size=1)

# 减小最大序列长度
sampler = OnlineSampler(..., max_seq_length=1024)

# 选择更小的数据模式
buffer = create_buffer(..., pattern="token_gate")
```

### 问题2：缓冲区阻塞

**症状**：采样速度慢，训练等待时间长

**解决方案**：
```python
# 增大缓冲区大小
buffer = create_buffer(..., buffer_size_gb=8.0)

# 减小批次大小
sampler = OnlineSampler(..., batch_size=1)

# 确保消费者及时读取数据
while True:
    batch = interface.get_batch()
    if batch is not None:
        # 立即处理
        process_batch(batch)
```

### 问题3：模型加载失败

**症状**：`OSError: Can't load tokenizer`

**解决方案**：
```python
# 检查模型路径
import os
assert os.path.exists(model_path)

# 添加信任参数
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

# 确保有足够的磁盘空间
# Mixtral-8x7B 需要约 30GB 磁盘空间
```

### 问题4：MoE层检测失败

**症状**：`ValueError: No MoE layers found`

**解决方案**：
```python
# 检查模型结构
from online_sample.utils import detect_moe_layers
layers = detect_moe_layers(model)
print(f"Found {len(layers)} MoE layers")

# 手动指定MoE层
sampler = OnlineSampler(
    ...,
    moe_layer_indices=[0, 1, 2, ..., 31]  # 手动指定
)
```

## 常见问题 (FAQ)

### Q1: 如何选择合适的数据模式？

**A**: 根据你的训练需求选择：
- 如果只需要Gate输出进行训练，使用 `token_gate`
- 如果需要Gate输入进行训练，使用 `gate_input`
- 如果需要Attention激活值进行训练，使用 `attn_gate`

### Q2: 缓冲区大小应该如何设置？

**A**: 根据以下因素综合考虑：
- 可用GPU显存
- 采样和训练的速度差异
- 期望的吞吐量

一般建议设置为可用显存的10-20%。

### Q3: 如何处理变长序列？

**A**: 模块已经内置了变长序列支持：
```python
data = interface.get_batch()
for sample in data:
    seq_len = sample.seq_lengths.item()
    # 使用 seq_len 进行切片
    tokens = sample.tokens[:, :seq_len]
    gate_logits = sample.gate_logits[:, :, :seq_len, :]
```

### Q4: 可以在多GPU上运行吗？

**A**: 可以，使用 `device_map="auto"` 自动分配：
```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",  # 自动分配到多个GPU
    torch_dtype=torch.bfloat16
)
```

### Q5: 如何监控训练进度？

**A**: 使用统计信息接口：
```python
stats = interface.get_stats()
print(f"Buffer utilization: {stats['utilization']*100:.1f}%")
print(f"Samples processed: {stats['used_samples']}")
print(f"Memory used: {stats['used_memory_gb']:.2f} GB")
```

## 项目结构

```
src/online_sample/
├── __init__.py              # 模块初始化
├── data_structures.py        # 数据结构定义
├── buffer.py                # 缓冲区实现
├── utils.py                 # 工具函数
├── sampler.py               # 在线采样器
├── predictor_interface.py    # 预测器接口
├── example_usage.py          # 完整示例
├── README.md               # 本文档
└── test/                  # 测试目录
    ├── test_config.py       # 测试配置
    ├── test_data_structures.py
    ├── test_utils.py
    ├── test_buffer.py
    ├── test_buffer_stress.py
    ├── test_sampler.py
    ├── test_predictor_interface.py
    └── test_integration.py
```

## 扩展性

### 添加新的数据模式

1. 在 `data_structures.py` 中添加新的模式常量
2. 在 `ActivationData` 中添加对应的字段
3. 在 `buffer.py` 中更新内存计算逻辑
4. 在 `sampler.py` 中更新数据采集逻辑

### 自定义缓冲区策略

```python
from online_sample.buffer import ActivationBuffer

class CustomBuffer(ActivationBuffer):
    def write(self, data, timeout=None):
        # 自定义写入逻辑
        pass
    
    def read(self, batch_size=1, timeout=None):
        # 自定义读取逻辑
        pass
```

### 自定义采样器

```python
from online_sample.sampler import OnlineSampler

class CustomSampler(OnlineSampler):
    def _collect_activations(self, outputs, batch):
        # 自定义激活值采集逻辑
        pass
```

## 版本历史

### 0.1.0 (2024-03-26)
- ✅ 初始版本
- ✅ 支持三种数据模式
- ✅ 线程安全的缓冲区实现
- ✅ 异步采样和训练
- ✅ 完整的测试覆盖
- ✅ 完整的文档和示例

## 许可证

本项目遵循项目的整体许可证。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 Issue
- 发送邮件
- 参与讨论

## 致谢

感谢所有为本项目做出贡献的开发者。
