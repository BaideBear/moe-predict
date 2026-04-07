import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from online_sample.utils import get_moe_layer_info


class LayerPredictor(nn.Module):
    """
    PROBE 公式(7): Gate-initialized Lookahead Predictor

    \hat{l}_L = W_L h_{L-1} + b_L  +  \hat{W}^2_L \sigma(\hat{W}^1_L h_{L-1})
                冻结先验                零初始化残差 MLP
    """

    def __init__(self, gate_weight: torch.Tensor, gate_bias: torch.Tensor,
                 input_dim: int, num_experts: int, residual_hidden_dim: int = 2048):
        super().__init__()
        self.num_experts = num_experts
        self.input_dim = input_dim

        # 冻结先验：克隆目标层 gate 的参数，训练时不更新
        self.register_buffer('gate_w', gate_weight.clone())
        self.register_buffer('gate_b', gate_bias.clone())

        # 可训练残差 MLP: Linear -> SiLU -> Linear
        self.residual_mlp = nn.Sequential(
            nn.Linear(input_dim, residual_hidden_dim),   # \hat{W}^1_L
            nn.SiLU(),                                    # \sigma
            nn.Linear(residual_hidden_dim, num_experts)  # \hat{W}^2_L
        )

        # 零初始化确保冷启动时 predictor \equiv gate
        nn.init.zeros_(self.residual_mlp[-1].weight)
        nn.init.zeros_(self.residual_mlp[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, input_dim]  ->  logit [N, num_experts]"""
        prior = x @ self.gate_w.t() + self.gate_b
        residual = self.residual_mlp(x)
        return prior + residual


class PredictorModel(nn.Module):
    """每层一个独立的 LayerPredictor，结构与 PEPP 的 PredictorModel 一致"""

    def __init__(self, num_layers: int, input_dim: int, num_experts: int,
                 gate_params: list, residual_hidden_dim: int = 2048):
        """
        gate_params: list of tuples [(gate_w_l0, gate_b_l0), (gate_w_l1, gate_b_l1), ...]
        """
        super().__init__()
        self.num_layers = num_layers
        self.layer_predictors = nn.ModuleList([
            LayerPredictor(gate_w, gate_b, input_dim, num_experts, residual_hidden_dim)
            for gate_w, gate_b in gate_params
        ])

    def forward(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        return self.layer_predictors[layer_idx](x)


def create_predictor_model(model, num_layers: int, input_dim: int, num_experts: int,
                           residual_hidden_dim: int = 2048) -> PredictorModel:
    """从 base MoE 模型中提取每层 gate 的 weight/bias，构建 PROBE 预测器"""
    gate_params = []
    for layer_idx in range(num_layers):
        info = get_moe_layer_info(model, layer_idx)
        if info is None:
            raise ValueError(f"Layer {layer_idx}: no MoE gate found")
        gate_module = info['gate_module']
        gate_w = gate_module.weight.data
        gate_b = gate_module.bias.data if hasattr(gate_module, 'bias') and gate_module.bias is not None \
            else torch.zeros(gate_module.out_features, dtype=gate_w.dtype)
        gate_params.append((gate_w, gate_b))

    return PredictorModel(
        num_layers=num_layers,
        input_dim=input_dim,
        num_experts=num_experts,
        gate_params=gate_params,
        residual_hidden_dim=residual_hidden_dim,
    )
