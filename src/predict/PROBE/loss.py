import torch
import torch.nn.functional as F


def compute_ce_loss(pred_logits: torch.Tensor, target_logits: torch.Tensor) -> torch.Tensor:
    """
    PROBE 论文 §4.2 Scale-Driven Online Distillation:
    用 KL-divergence 将 predictor 的分布蒸馏到 gate 的真实分布。

    参数:
        pred_logits:   [N, num_experts], 预测器输出的原始 logits
        target_logits: [N, num_experts], gate 输出的原始 logits（从 buffer 中采得的值）

    本函数内部对两者都先做 softmax，再算 KL-div(predictor || gate)。
    """
    pred_probs = F.log_softmax(pred_logits, dim=-1)
    target_probs = F.softmax(target_logits, dim=-1)
    return F.kl_div(pred_probs, target_probs, reduction='batchmean')
