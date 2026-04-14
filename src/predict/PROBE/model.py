import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from online_sample.utils import get_moe_layer_info


class LayerPredictor(nn.Module):
    """
    PROBE: Gate-initialized Lookahead Predictor
    ŷ = W_L h_{L-1} + b_L  +  \hat{W}^2_L \sigma(\hat{W}^1_L h_{L-1})
    """

    def __init__(self, gate_weight: torch.Tensor, gate_bias: torch.Tensor,
                 input_dim: int, num_experts: int, residual_hidden_dim: int = 2048):
        super().__init__()
        self.num_experts = num_experts
        self.input_dim = input_dim

        # 1. 验证并处理 Gate 权重
        # 预期形状: [num_experts, input_dim]
        gw = gate_weight.clone().float()
        if gw.shape != (num_experts, input_dim):
            if gw.shape == (input_dim, num_experts):
                gw = gw.t()
            else:
                raise RuntimeError(f"Gate weight shape {gw.shape} incompatible with "
                                   f"input_dim {input_dim} and num_experts {num_experts}")

        gb = gate_bias.clone().float() if gate_bias is not None else torch.zeros(num_experts).float()

        self.register_buffer('gate_w', gw)
        self.register_buffer('gate_b', gb)

        # 2. 可训练残差 MLP
        # 最后一层必须输出 num_experts (所有专家的 logits)
        self.residual_mlp = nn.Sequential(
            nn.Linear(input_dim, residual_hidden_dim),
            nn.SiLU(),
            nn.Linear(residual_hidden_dim, num_experts)
        )

        # 零初始化
        nn.init.zeros_(self.residual_mlp[-1].weight)
        nn.init.zeros_(self.residual_mlp[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, input_dim]  ->  logit [N, num_experts]"""
        # 维度检查 (Debug)
        if x.shape[-1] != self.input_dim:
            raise RuntimeError(f"Input dim mismatch: expected {self.input_dim}, got {x.shape[-1]}")

        # prior: [N, input_dim] @ [input_dim, num_experts] + [num_experts] -> [N, num_experts]
        prior = torch.matmul(x, self.gate_w.t()) + self.gate_b
        residual = self.residual_mlp(x)
        return prior + residual


class PredictorModel(nn.Module):
    def __init__(self, num_layers: int, input_dim: int, num_experts: int,
                 gate_params: list, residual_hidden_dim: int = 2048):
        super().__init__()
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.layer_predictors = nn.ModuleList([
            LayerPredictor(gate_w, gate_b, input_dim, num_experts, residual_hidden_dim)
            for gate_w, gate_b in gate_params
        ])

    def forward(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        return self.layer_predictors[layer_idx](x)


def create_predictor_model(model, num_layers: int, input_dim: int, num_experts: int,
                           residual_hidden_dim: int = 2048) -> PredictorModel:
    """
    从 base MoE 模型中提取每层 gate 的 weight/bias。
    强制以第一层 gate 的权重形状为准，忽略可能错误的 num_experts_config。
    """
    gate_params = []

    # 1. 首先通过第一层确定真实的专家数量
    first_info = get_moe_layer_info(model, 0)
    if first_info is None:
        # 如果第0层没找到，就找第一个能找到的
        for i in range(num_layers):
            info = get_moe_layer_info(model, i)
            if info:
                first_info = info
                break

    if first_info is None:
        raise RuntimeError("No MoE gates found in the model")

    actual_num_experts = first_info['gate_module'].weight.shape[0]
    print(f"PROBE: Detected actual num_experts = {actual_num_experts} (Config was {num_experts_config})")

    # 2. 提取所有层的参数
    for layer_idx in range(num_layers):
        info = get_moe_layer_info(model, layer_idx)
        if info is None:
            continue
        gate_module = info['gate_module']

        gate_w = gate_module.weight.data
        gate_b = gate_module.bias.data if hasattr(gate_module, 'bias') and gate_module.bias is not None \
            else torch.zeros(actual_num_experts, dtype=gate_w.dtype)

        gate_params.append((gate_w, gate_b))

    return PredictorModel(
        num_layers=num_layers,
        input_dim=input_dim,
        num_experts=actual_num_experts,
        gate_params=gate_params,
        residual_hidden_dim=residual_hidden_dim,
    )
