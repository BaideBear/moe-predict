# Tasks

- [x] Task 1: 重构 evaluater.py - 移除不需要的指标和状态
  - [x] SubTask 1.1: 移除 CrossEntropy Loss 相关代码（`self.criterion`、`self.layer_losses`、`batch_layer_losses`、`avg_loss` 等所有 loss 相关计算和日志）
  - [x] SubTask 1.2: 移除全局累计式 top-k accuracy 相关状态变量（`self.layer_top_k_correct`、`self.layer_total_tokens`），替换为 Task 1 要求的逐批次逐层结构
  - [x] SubTask 1.3: 在构造函数中新增 `num_active_experts: int` 参数
  - [x] SubTask 1.4: 清理 `_log_metrics` 和 `finish` 中的 loss 相关输出

- [x] Task 2: 实现 Task 1 - 逐批次逐层严格 Top-K 命中率
  - [x] SubTask 2.1: 在 `_eval_batch` 中，保留原有 `_compute_top_k_match` 算法（严格集合匹配），改为每批次每层粒度计算命中率
  - [x] SubTask 2.2: 维护累计平均变量，计算全局累计平均 top-k 正确率
  - [x] SubTask 2.3: 在 `_log_metrics` 中记录：每批次每层命中率、每批次所有层平均命中率、全局累计平均

- [x] Task 3: 实现 Task 2 - B_acc 指标
  - [x] SubTask 3.1: 在 `_eval_batch` 中，计算真实路由 $R_{\text{batch}}$ 和预测路由 $\hat{R}_{\text{batch}}$（基于 `num_active_experts` 的 top-k 按位或）
  - [x] SubTask 3.2: 计算 B_acc 值
  - [x] SubTask 3.3: 模拟 batch_size=1 的情况：对每个 token 单独计算 B_acc，取平均
  - [x] SubTask 3.4: 维护累计平均变量
  - [x] SubTask 3.5: 在 `_log_metrics` 中记录 B_acc 每批次值、累计平均值、batch_size=1 模拟值和累计平均值

- [x] Task 4: 实现 Task 3 - Error Rate 指标
  - [x] SubTask 4.1: 在 `_eval_batch` 中，对每层计算模型预测概率分布 $\hat{p}$ 和真实经验概率分布 $p$
  - [x] SubTask 4.2: 计算 error_rate = |p̂ - p| / (1/num_experts)
  - [x] SubTask 4.3: 维护累计平均变量
  - [x] SubTask 4.4: 在 `_log_metrics` 中记录 error rate 每批次值和累计平均值

- [x] Task 5: 修改 test_predictor.py
  - [x] SubTask 5.1: 新增命令行参数 `--num_active_experts`
  - [x] SubTask 5.2: 传递 `num_active_experts` 参数给 GatePredictorEvaluater
  - [x] SubTask 5.3: 移除与 loss 相关的输出

# Task Dependencies
- [Task 2] depends on [Task 1]
- [Task 3] depends on [Task 1]
- [Task 4] depends on [Task 1]
- [Task 5] depends on [Task 2, Task 3, Task 4]
