import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class WeightedBCELoss(nn.Module):
    def __init__(
        self,
        top_k: int = 2,
        weight_top10: float = 3.0,
        weight_top11_30: float = 1.5,
        weight_others: float = 0.5,
        top_n_for_ranking: int = 10
    ):
        super().__init__()
        self.top_k = top_k
        self.weight_top10 = weight_top10
        self.weight_top11_30 = weight_top11_30
        self.weight_others = weight_others
        self.top_n_for_ranking = top_n_for_ranking
        
        self.register_buffer('weight_top10_tensor', torch.tensor(weight_top10))
        self.register_buffer('weight_top11_30_tensor', torch.tensor(weight_top11_30))
        self.register_buffer('weight_others_tensor', torch.tensor(weight_others))
    
    def forward(
        self,
        predictions: torch.Tensor,
        gate_logits: torch.Tensor
    ) -> torch.Tensor:
        num_experts = gate_logits.shape[-1]
        
        gate_ranks = gate_logits.argsort(dim=-1, descending=True).argsort(dim=-1)
        
        weight_top10 = self.weight_top10_tensor.to(device=predictions.device, dtype=predictions.dtype)
        weight_top11_30 = self.weight_top11_30_tensor.to(device=predictions.device, dtype=predictions.dtype)
        weight_others = self.weight_others_tensor.to(device=predictions.device, dtype=predictions.dtype)
        
        weights = torch.where(
            gate_ranks < self.top_n_for_ranking,
            weight_top10,
            torch.where(
                gate_ranks < 30,
                weight_top11_30,
                weight_others
            )
        )
        
        actual_top_k = min(self.top_k, num_experts)
        gate_top_k_indices = gate_logits.topk(actual_top_k, dim=-1).indices
        
        targets = torch.zeros_like(predictions)
        targets.scatter_(-1, gate_top_k_indices, 1.0)
        
        bce_loss = F.binary_cross_entropy_with_logits(
            predictions, 
            targets, 
            reduction='none'
        )
        
        weighted_loss = weights * bce_loss
        
        loss = weighted_loss.mean()
        
        return loss


class RankingAwareBCELoss(nn.Module):
    def __init__(
        self,
        top_k: int = 2,
        lambda_ranking: float = 0.3,
        margin: float = 0.1,
        weight_top10: float = 3.0,
        weight_top11_30: float = 1.5,
        weight_others: float = 0.5,
        top_n_for_ranking: int = 10
    ):
        super().__init__()
        self.top_k = top_k
        self.lambda_ranking = lambda_ranking
        self.margin = margin
        self.weight_top10 = weight_top10
        self.weight_top11_30 = weight_top11_30
        self.weight_others = weight_others
        self.top_n_for_ranking = top_n_for_ranking
        
        self.register_buffer('weight_top10_tensor', torch.tensor(weight_top10))
        self.register_buffer('weight_top11_30_tensor', torch.tensor(weight_top11_30))
        self.register_buffer('weight_others_tensor', torch.tensor(weight_others))
    
    def forward(
        self,
        predictions: torch.Tensor,
        gate_logits: torch.Tensor
    ) -> torch.Tensor:
        wbce_loss = self._compute_weighted_bce(predictions, gate_logits)
        
        ranking_loss = self._compute_ranking_loss(predictions, gate_logits)
        
        total_loss = wbce_loss + self.lambda_ranking * ranking_loss
        
        return total_loss
    
    def _compute_weighted_bce(
        self,
        predictions: torch.Tensor,
        gate_logits: torch.Tensor
    ) -> torch.Tensor:
        num_experts = gate_logits.shape[-1]
        
        gate_ranks = gate_logits.argsort(dim=-1, descending=True).argsort(dim=-1)
        
        weight_top10 = self.weight_top10_tensor.to(device=predictions.device, dtype=predictions.dtype)
        weight_top11_30 = self.weight_top11_30_tensor.to(device=predictions.device, dtype=predictions.dtype)
        weight_others = self.weight_others_tensor.to(device=predictions.device, dtype=predictions.dtype)
        
        weights = torch.where(
            gate_ranks < self.top_n_for_ranking,
            weight_top10,
            torch.where(
                gate_ranks < 30,
                weight_top11_30,
                weight_others
            )
        )
        
        actual_top_k = min(self.top_k, num_experts)
        gate_top_k_indices = gate_logits.topk(actual_top_k, dim=-1).indices
        
        targets = torch.zeros_like(predictions)
        targets.scatter_(-1, gate_top_k_indices, 1.0)
        
        bce_loss = F.binary_cross_entropy_with_logits(
            predictions, 
            targets, 
            reduction='none'
        )
        
        weighted_loss = weights * bce_loss
        
        loss = weighted_loss.mean()
        
        return loss
    
    def _compute_ranking_loss(
        self,
        predictions: torch.Tensor,
        gate_logits: torch.Tensor
    ) -> torch.Tensor:
        batch_size = predictions.shape[0]
        num_experts = predictions.shape[-1]
        
        if num_experts < 2:
            return predictions.new_zeros(1, requires_grad=False).squeeze()
        
        top_k_for_ranking = min(self.top_n_for_ranking, num_experts)
        
        gate_top_k = gate_logits.topk(top_k_for_ranking, dim=-1)
        gate_top_k_indices = gate_top_k.indices
        gate_top_k_values = gate_top_k.values
        
        batch_indices = torch.arange(batch_size, device=predictions.device).unsqueeze(1).expand(-1, top_k_for_ranking)
        pred_top_k = predictions[batch_indices, gate_top_k_indices]
        
        pred_diff = pred_top_k.unsqueeze(2) - pred_top_k.unsqueeze(1)
        score_diff = gate_top_k_values.unsqueeze(2) - gate_top_k_values.unsqueeze(1)
        
        valid_pairs = (score_diff > 0).float()
        
        pair_losses = F.relu(self.margin - pred_diff) * valid_pairs
        
        num_valid_pairs = valid_pairs.sum()
        
        if num_valid_pairs > 0:
            ranking_loss = pair_losses.sum() / num_valid_pairs
        else:
            ranking_loss = predictions.new_zeros(1, requires_grad=False).squeeze()
        
        return ranking_loss


class CrossEntropyLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce_loss(predictions, targets)


def create_loss_function(
    loss_type: str,
    top_k: int = 2,
    lambda_ranking: float = 0.3,
    margin: float = 0.1,
    weight_top10: float = 3.0,
    weight_top11_30: float = 1.5,
    weight_others: float = 0.5,
    top_n_for_ranking: int = 10
) -> nn.Module:
    loss_type_lower = loss_type.lower()
    
    if loss_type_lower in ["ce", "cross_entropy"]:
        return CrossEntropyLoss()
    elif loss_type_lower in ["weighted_bce", "wbce"]:
        return WeightedBCELoss(
            top_k=top_k,
            weight_top10=weight_top10,
            weight_top11_30=weight_top11_30,
            weight_others=weight_others,
            top_n_for_ranking=top_n_for_ranking
        )
    elif loss_type_lower in ["ranking_aware_bce", "rabce"]:
        return RankingAwareBCELoss(
            top_k=top_k,
            lambda_ranking=lambda_ranking,
            margin=margin,
            weight_top10=weight_top10,
            weight_top11_30=weight_top11_30,
            weight_others=weight_others,
            top_n_for_ranking=top_n_for_ranking
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Available options: ce, weighted_bce, ranking_aware_bce")


def list_available_losses() -> list:
    return ["ce", "weighted_bce", "ranking_aware_bce"]
