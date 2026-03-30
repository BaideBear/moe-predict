from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import torch


@dataclass
class ModelConfig:
    model_name: str
    num_layers: int
    hidden_dim: int
    num_experts: int
    max_seq_length: int
    dtype: torch.dtype = torch.bfloat16


@dataclass
class BufferStats:
    total_samples: int
    used_samples: int
    free_samples: int
    buffer_size_gb: float
    used_memory_gb: float


class ActivationPattern:
    ATTN_GATE = "attn_gate"
    GATE_INPUT = "gate_input"
    TOKEN_GATE = "token_gate"
    
    ALL_PATTERNS = [ATTN_GATE, GATE_INPUT, TOKEN_GATE]
    
    @classmethod
    def validate(cls, pattern: str) -> bool:
        return pattern in cls.ALL_PATTERNS
    
    @classmethod
    def get_required_fields(cls, pattern: str) -> List[str]:
        if pattern == cls.ATTN_GATE:
            return ["attn_hidden_states", "gate_logits", "tokens"]
        elif pattern == cls.GATE_INPUT:
            return ["gate_inputs", "gate_logits", "tokens"]
        elif pattern == cls.TOKEN_GATE:
            return ["tokens", "gate_logits"]
        else:
            raise ValueError(f"Unknown pattern: {pattern}")


@dataclass
class ActivationData:
    tokens: torch.Tensor
    gate_logits: torch.Tensor
    attn_hidden_states: Optional[torch.Tensor] = None
    gate_inputs: Optional[torch.Tensor] = None
    seq_lengths: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to(self, device: torch.device) -> 'ActivationData':
        return ActivationData(
            tokens=self.tokens.to(device),
            gate_logits=self.gate_logits.to(device),
            attn_hidden_states=self.attn_hidden_states.to(device) if self.attn_hidden_states is not None else None,
            gate_inputs=self.gate_inputs.to(device) if self.gate_inputs is not None else None,
            seq_lengths=self.seq_lengths.to(device) if self.seq_lengths is not None else None,
            metadata=self.metadata
        )
    
    def get_memory_size(self) -> int:
        size = 0
        size += self.tokens.element_size() * self.tokens.numel()
        size += self.gate_logits.element_size() * self.gate_logits.numel()
        if self.attn_hidden_states is not None:
            size += self.attn_hidden_states.element_size() * self.attn_hidden_states.numel()
        if self.gate_inputs is not None:
            size += self.gate_inputs.element_size() * self.gate_inputs.numel()
        if self.seq_lengths is not None:
            size += self.seq_lengths.element_size() * self.seq_lengths.numel()
        return size
    
    def validate(self, pattern: str) -> bool:
        required_fields = ActivationPattern.get_required_fields(pattern)
        for field in required_fields:
            if getattr(self, field) is None:
                return False
        return True
