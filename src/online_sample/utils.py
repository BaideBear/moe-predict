import torch
from typing import Dict, List, Tuple, Optional, Any
from transformers import AutoModelForCausalLM
from .data_structures import ModelConfig


def extract_model_config(
    model: AutoModelForCausalLM,
    model_name: str,
    max_seq_length: int = 2048
) -> ModelConfig:
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
        num_layers = len(layers)
        
        first_layer = layers[0]
        hidden_dim = None
        num_experts = None
        
        if hasattr(first_layer, 'block_sparse_moe'):
            block_sparse_moe = first_layer.block_sparse_moe
            if hasattr(block_sparse_moe, 'gate'):
                gate = block_sparse_moe.gate
                if hasattr(gate, 'in_features'):
                    hidden_dim = gate.in_features
                if hasattr(gate, 'out_features'):
                    num_experts = gate.out_features
        
        if hidden_dim is None and hasattr(first_layer, 'mlp'):
            mlp = first_layer.mlp
            if hasattr(mlp, 'gate'):
                gate = mlp.gate
                if hasattr(gate, 'in_features'):
                    hidden_dim = gate.in_features
                if hasattr(gate, 'out_features'):
                    num_experts = gate.out_features
            elif hasattr(mlp, 'block_sparse_moe'):
                block_sparse_moe = mlp.block_sparse_moe
                if hasattr(block_sparse_moe, 'gate'):
                    gate = block_sparse_moe.gate
                    if hasattr(gate, 'in_features'):
                        hidden_dim = gate.in_features
                    if hasattr(gate, 'out_features'):
                        num_experts = gate.out_features
        
        if hidden_dim is None:
            hidden_dim = model.config.hidden_size
        
        if num_experts is None:
            if hasattr(model.config, 'num_local_experts'):
                num_experts = model.config.num_local_experts
            elif hasattr(model.config, 'n_routed_experts'):
                num_experts = model.config.n_routed_experts
            elif hasattr(model.config, 'num_experts'):
                num_experts = model.config.num_experts
            else:
                num_experts = 8
        
        return ModelConfig(
            model_name=model_name,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            max_seq_length=max_seq_length,
            dtype=torch.bfloat16
        )
    
    raise ValueError("Unable to extract model config: unsupported model architecture")


def detect_moe_layers(model: AutoModelForCausalLM) -> List[int]:
    moe_layers = []
    
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        for i, layer in enumerate(model.model.layers):
            has_moe = False
            
            if hasattr(layer, 'block_sparse_moe'):
                if hasattr(layer.block_sparse_moe, 'gate'):
                    has_moe = True
            
            if not has_moe and hasattr(layer, 'mlp'):
                if hasattr(layer.mlp, 'gate'):
                    has_moe = True
                elif hasattr(layer.mlp, 'block_sparse_moe'):
                    if hasattr(layer.mlp.block_sparse_moe, 'gate'):
                        has_moe = True
            
            if has_moe:
                moe_layers.append(i)
    
    return moe_layers


def get_moe_layer_info(model: AutoModelForCausalLM, layer_idx: int) -> Optional[Dict[str, Any]]:
    if not hasattr(model, 'model') or not hasattr(model.model, 'layers'):
        return None
    
    layer = model.model.layers[layer_idx]
    moe_info = None
    
    if hasattr(layer, 'block_sparse_moe'):
        block_sparse_moe = layer.block_sparse_moe
        if hasattr(block_sparse_moe, 'gate'):
            moe_info = {
                'type': 'direct_block_sparse_moe',
                'gate_path': f'layer.block_sparse_moe.gate',
                'gate_module': block_sparse_moe.gate,
                'input_layernorm_path': 'layer.input_layernorm'
            }
    
    if moe_info is None and hasattr(layer, 'mlp'):
        mlp = layer.mlp
        if hasattr(mlp, 'gate'):
            moe_info = {
                'type': 'mlp_gate',
                'gate_path': f'layer.mlp.gate',
                'gate_module': mlp.gate,
                'input_layernorm_path': 'layer.input_layernorm'
            }
        elif hasattr(mlp, 'block_sparse_moe'):
            block_sparse_moe = mlp.block_sparse_moe
            if hasattr(block_sparse_moe, 'gate'):
                moe_info = {
                    'type': 'mlp_block_sparse_moe',
                    'gate_path': f'layer.mlp.block_sparse_moe.gate',
                    'gate_module': block_sparse_moe.gate,
                    'input_layernorm_path': 'layer.input_layernorm'
                }
    
    return moe_info


def calculate_memory_usage(
    num_samples: int,
    num_layers: int,
    hidden_dim: int,
    num_experts: int,
    seq_length: int,
    pattern: str,
    dtype: torch.dtype = torch.bfloat16
) -> Dict[str, int]:
    bytes_per_element = torch.finfo(dtype).bits // 8
    
    memory = {}
    
    if pattern in ['attn_gate', 'gate_input']:
        if pattern == 'attn_gate':
            memory['attn_hidden_states'] = num_samples * num_layers * seq_length * hidden_dim * bytes_per_element
        if pattern == 'gate_input':
            memory['gate_inputs'] = num_samples * num_layers * seq_length * hidden_dim * bytes_per_element
        
        memory['gate_logits'] = num_samples * num_layers * seq_length * num_experts * bytes_per_element
        memory['tokens'] = num_samples * seq_length * 4
    
    elif pattern == 'token_gate':
        memory['gate_logits'] = num_samples * num_layers * seq_length * num_experts * bytes_per_element
        memory['tokens'] = num_samples * seq_length * 4
    
    memory['total'] = sum(memory.values())
    
    return memory


def estimate_buffer_capacity(
    buffer_size_gb: float,
    num_layers: int,
    hidden_dim: int,
    num_experts: int,
    avg_seq_length: int,
    pattern: str,
    dtype: torch.dtype = torch.bfloat16
) -> int:
    buffer_size_bytes = int(buffer_size_gb * 1024**3)
    
    memory_per_sample = calculate_memory_usage(
        num_samples=1,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        seq_length=avg_seq_length,
        pattern=pattern,
        dtype=dtype
    )['total']
    
    capacity = buffer_size_bytes // memory_per_sample
    
    return max(1, capacity)
