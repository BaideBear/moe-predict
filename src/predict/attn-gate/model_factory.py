import torch.nn as nn
from typing import Dict, Any
try:
    from .models import SimpleMLPPredictorModel, create_simple_mlp_predictor, create_mlp_without_dropout_predictor, create_lstm_predictor
except ImportError:
    from models import SimpleMLPPredictorModel, create_simple_mlp_predictor, create_mlp_without_dropout_predictor, create_lstm_predictor


class ModelFactory:
    _registry: Dict[str, Any] = {}
    
    @classmethod
    def register(cls, name: str, model_class):
        cls._registry[name] = model_class
    
    @classmethod
    def create(cls, name: str, **kwargs):
        if name not in cls._registry:
            raise ValueError(f"Unknown model type: {name}. Available models: {list(cls._registry.keys())}")
        return cls._registry[name](**kwargs)
    
    @classmethod
    def list_models(cls) -> list:
        return list(cls._registry.keys())


ModelFactory.register("simple_mlp", create_simple_mlp_predictor)
ModelFactory.register("mlp_without_dropout", create_mlp_without_dropout_predictor)
ModelFactory.register("lstm", create_lstm_predictor)


def get_predictor_model(model_type: str, num_layers: int, input_dim: int, num_experts: int, 
                       hidden_dim: int = 2048, dropout: float = 0.1) -> nn.Module:
    return ModelFactory.create(
        model_type,
        num_layers=num_layers,
        input_dim=input_dim,
        num_experts=num_experts,
        hidden_dim=hidden_dim,
        dropout=dropout
    )


def list_available_models() -> list:
    return ModelFactory.list_models()
