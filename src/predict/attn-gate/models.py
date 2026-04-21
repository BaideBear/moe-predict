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


class SimpleMLPPredictorWithoutDropout(nn.Module):
    def __init__(self, input_dim: int, num_experts: int, hidden_dim: int = 2048, dropout: float = 0.1):
        super(SimpleMLPPredictorWithoutDropout, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_experts)
        )

    def forward(self, x):
        return self.net(x)


class SimpleMLPPredictorModelWithoutDropout(nn.Module):
    def __init__(self, num_layers: int, input_dim: int, num_experts: int, hidden_dim: int = 2048, dropout: float = 0.1):
        super(SimpleMLPPredictorModelWithoutDropout, self).__init__()
        self.num_layers = num_layers
        self.layer_predictors = nn.ModuleList([
            SimpleMLPPredictorWithoutDropout(input_dim, num_experts, hidden_dim, dropout)
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


def create_mlp_without_dropout_predictor(num_layers: int, input_dim: int, num_experts: int, hidden_dim: int = 2048, dropout: float = 0.1) -> SimpleMLPPredictorModelWithoutDropout:
    return SimpleMLPPredictorModelWithoutDropout(num_layers, input_dim, num_experts, hidden_dim, dropout)


class LSTMPredictor(nn.Module):
    def __init__(self, input_dim: int, num_experts: int, lstm_hidden_dim: int = 64, 
                 projection_dim: int = 128, num_lstm_layers: int = 2, dropout: float = 0.1):
        super(LSTMPredictor, self).__init__()
        self.projection_dim = projection_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.ReLU()
        )
        
        self.lstm = nn.LSTM(
            input_size=projection_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(projection_dim, lstm_hidden_dim),
            nn.ReLU(),
            nn.Linear(lstm_hidden_dim, lstm_hidden_dim)
        )
        
        self.classifier = nn.Linear(lstm_hidden_dim, num_experts)
        
        self.layer_norm = nn.LayerNorm(lstm_hidden_dim)

    def forward(self, x):
        x_proj = self.input_projection(x)
        
        x_proj_reshaped = x_proj.unsqueeze(1)
        
        lstm_out, _ = self.lstm(x_proj_reshaped)
        
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        ffn_out = self.ffn(x_proj)
        
        combined = self.layer_norm(attn_out.squeeze(1) + ffn_out)
        
        logits = self.classifier(combined)
        
        return logits


class LSTMPredictorModel(nn.Module):
    def __init__(self, num_layers: int, input_dim: int, num_experts: int, 
                 lstm_hidden_dim: int = 64, projection_dim: int = 128, 
                 num_lstm_layers: int = 2, dropout: float = 0.1):
        super(LSTMPredictorModel, self).__init__()
        self.num_layers = num_layers
        self.layer_predictors = nn.ModuleList([
            LSTMPredictor(
                input_dim=input_dim,
                num_experts=num_experts,
                lstm_hidden_dim=lstm_hidden_dim,
                projection_dim=projection_dim,
                num_lstm_layers=num_lstm_layers,
                dropout=dropout
            )
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


def create_lstm_predictor(num_layers: int, input_dim: int, num_experts: int, hidden_dim: int = 2048,
                        lstm_hidden_dim: int = 64, projection_dim: int = 128, 
                        num_lstm_layers: int = 2, dropout: float = 0.1) -> LSTMPredictorModel:
    return LSTMPredictorModel(
        num_layers=num_layers,
        input_dim=input_dim,
        num_experts=num_experts,
        lstm_hidden_dim=lstm_hidden_dim,
        projection_dim=projection_dim,
        num_lstm_layers=num_lstm_layers,
        dropout=dropout
    )

