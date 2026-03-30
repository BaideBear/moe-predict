import torch
import torch.nn.functional as F


def compute_custom_loss(pred_logits, target_scores, top_k_routing=8, lambda_rank=1.0, margin=0.1):
    """
    论文中的 Ranking-aware multi-label classification 损失函数。

    参数:
        pred_logits: [N, E], 预测器输出的原始 logits (未经过 sigmoid)
        target_scores: [N, E], 专家真实的路由概率或打分
        top_k_routing: int, 模型真实选取的专家数量 k，用于判断 Equation (7) 中的 "if expert j is top-k"
        lambda_rank: float, 公式 (9) 中的权重系数 λ
        margin: float, 公式 (11) 中的 m
    """
    N, E = pred_logits.shape
    device = pred_logits.device

    sorted_scores, indices = torch.sort(target_scores, dim=-1, descending=True)

    ranks = torch.empty_like(indices)
    ranks.scatter_(1, indices, torch.arange(E, device=device).unsqueeze(0).expand(N, E))

    W = torch.full_like(target_scores, 0.5)

    W = torch.where(ranks < 10, torch.tensor(3.0, device=device), W)

    W = torch.where((ranks >= 10) & (ranks < 30), torch.tensor(1.5, device=device), W)

    labels_bce = (ranks < top_k_routing).float()

    L_WBCE = F.binary_cross_entropy_with_logits(
        pred_logits,
        labels_bce,
        weight=W,
        reduction='mean'
    )

    top10_idx = indices[:, :10]

    s_raw_top10 = torch.gather(pred_logits, 1, top10_idx)
    s_real_top10 = torch.gather(target_scores, 1, top10_idx)

    s_real_j = s_real_top10.unsqueeze(2)
    s_real_k = s_real_top10.unsqueeze(1)

    s_raw_j = s_raw_top10.unsqueeze(2)
    s_raw_k = s_raw_top10.unsqueeze(1)

    valid_mask = (s_real_j > s_real_k).float()

    diff_raw = s_raw_j - s_raw_k
    pair_loss_matrix = F.relu(margin - diff_raw)

    masked_pair_loss = pair_loss_matrix * valid_mask

    L_ranking = masked_pair_loss.sum(dim=(1, 2)).mean()

    L_total = L_WBCE + lambda_rank * L_ranking

    return L_total
