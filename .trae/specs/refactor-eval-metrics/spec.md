# 重构评估指标 Spec

## Why
当前 evaluater.py 中的评估指标（CrossEntropy Loss、全局平均 top-k accuracy）不符合需求。需要移除不需要的指标，并实现三个新的评估任务：逐层逐批次 top-k 命中率、Batch Accuracy (B_acc)、Error Rate。

## What Changes
- **移除** CrossEntropy Loss 相关计算与记录（`criterion`、`layer_losses`、`avg_loss` 等）
- **移除** 全局累计式的 top-k accuracy（当前实现是跨批次累加 correct count 再除以总 token 数，不满足"每批次每层"粒度要求）
- **修改** top-k accuracy 的保存粒度：保留原有的 `_compute_top_k_match` 算法（严格 top-k 集合完全匹配，无关顺序），将保存方式从全局累计改为每批次每层的命中率、每批次所有层平均命中率、全局平均随批次变化
- **新增** B_acc 指标：基于 batch 内所有 token 路由路径按位或合并后，衡量预测激活专家集合与实际激活专家集合的交集准确率
  - 保存每个批次的实际值和平均值随批次的变化
  - 模拟 batch_size=1 情况下的值（将 batch 中每个 token 当作独立 batch 计算）
  - 需要参数：实际激活的专家数（num_active_experts）
- **新增** Error Rate 指标：|p̂ - p| / (1/#experts)，其中 p̂ 是模型预测的 expert 被选中概率，p 是测试集上统计的经验概率
  - 保存每个批次的实际值和平均值随批次的变化
- **所有评估结果保存到 wandb**

## Impact
- Affected code: `src/predict/attn-gate/evaluater.py`, `src/predict/attn-gate/test_predictor.py`

## ADDED Requirements

### Requirement: Task 1 - 逐批次逐层严格 Top-K 命中率

对于给定的 top_k_values 列表，逐个进行严格的 top-k 是否相同的分析（预测 top-k 集合 == 真实 top-k 集合）。

需要保存的维度：
1. 每个批次的每层的预测 top-k 命中率比例（非全局平均，而是每批次每层的正确率）
2. 每个批次所有层的平均正确率
3. 全局平均 top-k 正确率随批次的变化（累计平均）

#### Scenario: 正常评估批次
- **WHEN** 一个评估批次被处理
- **THEN** 对每个 layer_idx 和每个 k 值，计算该批次该层的 top-k 命中率 = 正确匹配的 token 数 / 有效 token 数
- **THEN** 记录 `topk_accuracy/batch_{batch}/layer_{layer}/k_{k}` 到 wandb
- **THEN** 记录 `topk_accuracy/batch_{batch}/avg_over_layers/k_{k}` (该批次所有层平均) 到 wandb
- **THEN** 记录 `topk_accuracy/cumulative_avg/k_{k}` (全局累计平均) 到 wandb

### Requirement: Task 2 - B_acc 指标

$$
\mathbb{B}_{\text{acc}} = \frac{1}{L} \sum_{i=0}^{L-1}
\frac{\sum_{j=0}^{E-1} \mathbb{I}(R_{\text{batch}}[i,j] = 1 \land \hat{R}_{\text{batch}}[i,j] = 1)}
{\sum_{j=0}^{E-1} \mathbb{I}(R_{\text{batch}}[i,j] = 1)}
$$

其中 $R_{\text{batch}} = r_1 \lor r_2 \lor \cdots \lor r_n$，将 batch 中所有 token 的路由路径按位或合并。

需要保存：
1. 每个批次的实际值（B_acc 值）和平均值随批次的变化
2. 模拟 batch_size=1 的情况：如果 batch_size=3，将每个 token 当作独立 batch 计算 B_acc，然后取三个值的平均值

参数：`num_active_experts`（实际激活的专家数，如 top-k 路由中的 k）

#### Scenario: 计算 B_acc
- **WHEN** 一个评估批次被处理
- **THEN** 对每层，计算 $R_{\text{batch}}$（真实路由按位或）和 $\hat{R}_{\text{batch}}$（预测路由按位或）
- **THEN** 真实路由：对每个 token，取 gate_logits 的 top-num_active_experts 的索引作为该 token 的路由
- **THEN** 预测路由：对每个 token，取模型预测的 top-num_active_experts 的索引作为该 token 的路由
- **THEN** 计算 B_acc 并记录到 wandb
- **THEN** 模拟 batch_size=1：对每个 token 单独计算 B_acc，再取平均

#### Scenario: batch_size=1 模拟
- **WHEN** batch 中有 N 个 token
- **THEN** 对每个 token t，计算 R_t（该 token 的真实路由，即 top-num_active_experts）和 R̂_t（该 token 的预测路由）
- **THEN** 对每个 token t，B_acc_t = |R_t ∩ R̂_t| / |R_t|
- **THEN** 模拟结果 = mean(B_acc_t for t in batch)
- **THEN** 同样保存每个批次的模拟值和累计平均值到 wandb

### Requirement: Task 3 - Error Rate 指标

$$
\text{error rate} = \frac{|\hat{p} - p|}{1 / \#\text{experts}}
$$

- $\hat{p}$：在训练集上估计出的 expert 被选中的概率（模型预测值）
- $p$：在测试集上统计得到的经验概率（真实观测值）
- $\#\text{experts}$：该层的 Expert 总数

需要保存：每个批次的实际值和平均值随批次的变化。

#### Scenario: 计算 Error Rate
- **WHEN** 一个评估批次被处理
- **THEN** 对每层，$\hat{p}$ = softmax(model_predictions) 得到的每个 expert 的被选中概率分布
- **THEN** 对每层，$p$ = 基于真实 gate_logits 的 top-1（或实际路由）统计的经验概率分布
- **THEN** error_rate_per_layer = |p̂ - p| / (1/num_experts)，对每层计算
- **THEN** 整体 error_rate = 各层 error_rate 的平均
- **THEN** 记录每个批次的实际值和累计平均值到 wandb

## MODIFIED Requirements

### Requirement: evaluater.py 构造函数
构造函数需要新增参数 `num_active_experts: int`，用于 B_acc 指标计算。移除与 loss 相关的状态变量。

### Requirement: test_predictor.py
需要传递 `num_active_experts` 参数给 evaluater。移除与 loss 相关的输出。

## REMOVED Requirements

### Requirement: CrossEntropy Loss 计算
**Reason**: 用户明确要求只保留指定的三个评估任务，不需要 loss 计算。
**Migration**: 移除 `self.criterion`、`self.layer_losses`、`batch_layer_losses` 等所有 loss 相关代码。

### Requirement: 全局累计式 top-k accuracy 保存方式
**Reason**: 当前实现是跨批次累加再平均，用户要求的是每批次每层的粒度。算法本身（`_compute_top_k_match`，严格 top-k 集合完全匹配）是正确的，保留不变。
**Migration**: 仅修改保存粒度，替换为 Task 1 中定义的逐批次逐层 top-k 命中率。
