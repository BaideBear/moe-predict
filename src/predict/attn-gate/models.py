import torch
import torch.nn as nn


class SimpleMLPPredictor(nn.Module):
    def __init__(self, input_dim: int, num_experts: int, hidden_dim: int = 2048, dropout: float = 0.1):
        super(SimpleMLPPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_experts)
        )

    def forward(self, x):
        return self.net(x)


class SimpleMLPPredictorModel(nn.Module):
    def __init__(self, num_layers: int, input_dim: int, num_experts: int, hidden_dim: int = 2048, dropout: float = 0.1):
        super(SimpleMLPPredictorModel, self).__init__()
        self.num_layers = num_layers
        self.layer_predictors = nn.ModuleList([
            SimpleMLPPredictor(input_dim, num_experts, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, layer_idx):
        return self.layer_predictors[layer_idx](x)
    
    def forward_all_layers(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x = x.view(-1, hidden_dim)
        
        outputs = []
        for layer_idx in range(self.num_layers):
            layer_output = self.layer_predictors[layer_idx](x)
            outputs.append(layer_output)
        
        stacked_outputs = torch.stack(outputs, dim=1)
        stacked_outputs = stacked_outputs.view(batch_size, seq_len, self.num_layers, -1)
        return stacked_outputs


def create_simple_mlp_predictor(num_layers: int, input_dim: int, num_experts: int, hidden_dim: int = 2048, dropout: float = 0.1) -> SimpleMLPPredictorModel:
    return SimpleMLPPredictorModel(num_layers, input_dim, num_experts, hidden_dim, dropout)
