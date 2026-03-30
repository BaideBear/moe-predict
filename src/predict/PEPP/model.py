import torch
import torch.nn as nn


class LayerPredictor(nn.Module):
    def __init__(self, input_dim: int, num_experts: int, hidden_dim: int = 2048, dropout: float = 0.1):
        super(LayerPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_experts)
        )

    def forward(self, x):
        return self.net(x)


class PredictorModel(nn.Module):
    def __init__(self, num_layers: int, input_dim: int, num_experts: int, hidden_dim: int = 2048, dropout: float = 0.1):
        super(PredictorModel, self).__init__()
        self.num_layers = num_layers
        self.layer_predictors = nn.ModuleList([
            LayerPredictor(input_dim, num_experts, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, layer_idx):
        return self.layer_predictors[layer_idx](x)


def create_predictor_model(num_layers: int, input_dim: int, num_experts: int, hidden_dim: int = 2048, dropout: float = 0.1) -> PredictorModel:
    return PredictorModel(num_layers, input_dim, num_experts, hidden_dim, dropout)
