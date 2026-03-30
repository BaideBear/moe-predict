import unittest
import torch
from test_config import DEVICE, DTYPE
from data_structures import ModelConfig, BufferStats, ActivationPattern, ActivationData


class TestModelConfig(unittest.TestCase):
    
    def test_model_config_creation(self):
        config = ModelConfig(
            model_name="test_model",
            num_layers=32,
            hidden_dim=4096,
            num_experts=8,
            max_seq_length=2048,
            dtype=DTYPE
        )
        
        self.assertEqual(config.model_name, "test_model")
        self.assertEqual(config.num_layers, 32)
        self.assertEqual(config.hidden_dim, 4096)
        self.assertEqual(config.num_experts, 8)
        self.assertEqual(config.max_seq_length, 2048)
        self.assertEqual(config.dtype, DTYPE)
    
    def test_model_config_default_dtype(self):
        config = ModelConfig(
            model_name="test_model",
            num_layers=32,
            hidden_dim=4096,
            num_experts=8,
            max_seq_length=2048
        )
        
        self.assertEqual(config.dtype, torch.bfloat16)


class TestBufferStats(unittest.TestCase):
    
    def test_buffer_stats_creation(self):
        stats = BufferStats(
            total_samples=100,
            used_samples=50,
            free_samples=50,
            buffer_size_gb=4.0,
            used_memory_gb=2.0
        )
        
        self.assertEqual(stats.total_samples, 100)
        self.assertEqual(stats.used_samples, 50)
        self.assertEqual(stats.free_samples, 50)
        self.assertEqual(stats.buffer_size_gb, 4.0)
        self.assertEqual(stats.used_memory_gb, 2.0)


class TestActivationPattern(unittest.TestCase):
    
    def test_pattern_constants(self):
        self.assertEqual(ActivationPattern.ATTN_GATE, "attn_gate")
        self.assertEqual(ActivationPattern.GATE_INPUT, "gate_input")
        self.assertEqual(ActivationPattern.TOKEN_GATE, "token_gate")
    
    def test_all_patterns(self):
        self.assertIn("attn_gate", ActivationPattern.ALL_PATTERNS)
        self.assertIn("gate_input", ActivationPattern.ALL_PATTERNS)
        self.assertIn("token_gate", ActivationPattern.ALL_PATTERNS)
    
    def test_validate_valid_pattern(self):
        self.assertTrue(ActivationPattern.validate("attn_gate"))
        self.assertTrue(ActivationPattern.validate("gate_input"))
        self.assertTrue(ActivationPattern.validate("token_gate"))
    
    def test_validate_invalid_pattern(self):
        self.assertFalse(ActivationPattern.validate("invalid_pattern"))
    
    def test_get_required_fields_attn_gate(self):
        fields = ActivationPattern.get_required_fields("attn_gate")
        self.assertIn("attn_hidden_states", fields)
        self.assertIn("gate_logits", fields)
        self.assertIn("tokens", fields)
        self.assertEqual(len(fields), 3)
    
    def test_get_required_fields_gate_input(self):
        fields = ActivationPattern.get_required_fields("gate_input")
        self.assertIn("gate_inputs", fields)
        self.assertIn("gate_logits", fields)
        self.assertIn("tokens", fields)
        self.assertEqual(len(fields), 3)
    
    def test_get_required_fields_token_gate(self):
        fields = ActivationPattern.get_required_fields("token_gate")
        self.assertIn("tokens", fields)
        self.assertIn("gate_logits", fields)
        self.assertEqual(len(fields), 2)
    
    def test_get_required_fields_invalid_pattern(self):
        with self.assertRaises(ValueError):
            ActivationPattern.get_required_fields("invalid_pattern")


class TestActivationData(unittest.TestCase):
    
    def test_activation_data_creation(self):
        tokens = torch.randint(0, 1000, (10, 100), dtype=torch.int32).to(DEVICE)
        gate_logits = torch.randn(10, 32, 100, 8, dtype=DTYPE).to(DEVICE)
        attn_hidden_states = torch.randn(10, 32, 100, 4096, dtype=DTYPE).to(DEVICE)
        seq_lengths = torch.randint(50, 100, (10,), dtype=torch.int32).to(DEVICE)
        
        data = ActivationData(
            tokens=tokens,
            gate_logits=gate_logits,
            attn_hidden_states=attn_hidden_states,
            seq_lengths=seq_lengths,
            metadata={"pattern": "attn_gate"}
        )
        
        self.assertEqual(data.tokens.shape, (10, 100))
        self.assertEqual(data.gate_logits.shape, (10, 32, 100, 8))
        self.assertEqual(data.attn_hidden_states.shape, (10, 32, 100, 4096))
        self.assertEqual(data.seq_lengths.shape, (10,))
        self.assertEqual(data.metadata["pattern"], "attn_gate")
    
    def test_activation_data_to_device(self):
        tokens = torch.randint(0, 1000, (10, 100), dtype=torch.int32)
        gate_logits = torch.randn(10, 32, 100, 8, dtype=DTYPE)
        
        data = ActivationData(
            tokens=tokens,
            gate_logits=gate_logits,
            metadata={"pattern": "attn_gate"}
        )
        
        self.assertEqual(data.tokens.device.type, "cpu")
        self.assertEqual(data.gate_logits.device.type, "cpu")
        
        data_gpu = data.to(DEVICE)
        
        self.assertEqual(data_gpu.tokens.device.type, DEVICE)
        self.assertEqual(data_gpu.gate_logits.device.type, DEVICE)
    
    def test_get_memory_size(self):
        tokens = torch.randint(0, 1000, (10, 100), dtype=torch.int32).to(DEVICE)
        gate_logits = torch.randn(10, 32, 100, 8, dtype=DTYPE).to(DEVICE)
        attn_hidden_states = torch.randn(10, 32, 100, 4096, dtype=DTYPE).to(DEVICE)
        
        data = ActivationData(
            tokens=tokens,
            gate_logits=gate_logits,
            attn_hidden_states=attn_hidden_states,
            metadata={"pattern": "attn_gate"}
        )
        
        memory_size = data.get_memory_size()
        
        expected_tokens = tokens.element_size() * tokens.numel()
        expected_gate = gate_logits.element_size() * gate_logits.numel()
        expected_attn = attn_hidden_states.element_size() * attn_hidden_states.numel()
        expected_total = expected_tokens + expected_gate + expected_attn
        
        self.assertEqual(memory_size, expected_total)
    
    def test_validate_attn_gate_pattern(self):
        tokens = torch.randint(0, 1000, (10, 100), dtype=torch.int32).to(DEVICE)
        gate_logits = torch.randn(10, 32, 100, 8, dtype=DTYPE).to(DEVICE)
        attn_hidden_states = torch.randn(10, 32, 100, 4096, dtype=DTYPE).to(DEVICE)
        
        data = ActivationData(
            tokens=tokens,
            gate_logits=gate_logits,
            attn_hidden_states=attn_hidden_states,
            metadata={"pattern": "attn_gate"}
        )
        
        self.assertTrue(data.validate("attn_gate"))
        self.assertIsNone(data.gate_inputs)
    
    def test_validate_gate_input_pattern(self):
        tokens = torch.randint(0, 1000, (10, 100), dtype=torch.int32).to(DEVICE)
        gate_logits = torch.randn(10, 32, 100, 8, dtype=DTYPE).to(DEVICE)
        gate_inputs = torch.randn(10, 32, 100, 4096, dtype=DTYPE).to(DEVICE)
        
        data = ActivationData(
            tokens=tokens,
            gate_logits=gate_logits,
            gate_inputs=gate_inputs,
            metadata={"pattern": "gate_input"}
        )
        
        self.assertTrue(data.validate("gate_input"))
        self.assertIsNone(data.attn_hidden_states)
    
    def test_validate_token_gate_pattern(self):
        tokens = torch.randint(0, 1000, (10, 100), dtype=torch.int32).to(DEVICE)
        gate_logits = torch.randn(10, 32, 100, 8, dtype=DTYPE).to(DEVICE)
        
        data = ActivationData(
            tokens=tokens,
            gate_logits=gate_logits,
            metadata={"pattern": "token_gate"}
        )
        
        self.assertTrue(data.validate("token_gate"))
        self.assertIsNone(data.attn_hidden_states)
        self.assertIsNone(data.gate_inputs)


if __name__ == "__main__":
    unittest.main()
