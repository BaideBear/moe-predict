import unittest
import torch
from unittest.mock import Mock, MagicMock
from test_config import DEVICE, DTYPE
import online_sample.utils as utils_module
from online_sample.utils import (
    extract_model_config,
    detect_moe_layers,
    get_moe_layer_info,
    calculate_memory_usage,
    estimate_buffer_capacity
)


class TestExtractModelConfig(unittest.TestCase):
    
    def setUp(self):
        self.mock_model = Mock()
        self.mock_model.model = Mock()
        self.mock_model.model.layers = []
        
        for i in range(32):
            layer = Mock()
            layer.input_layernorm = Mock()
            
            block_sparse_moe = Mock()
            block_sparse_moe.gate = Mock()
            block_sparse_moe.gate.in_features = 4096
            block_sparse_moe.gate.out_features = 8
            
            layer.block_sparse_moe = block_sparse_moe
            self.mock_model.model.layers.append(layer)
    
    def test_extract_model_config_basic(self):
        config = extract_model_config(self.mock_model, "test_model", 2048)
        
        self.assertEqual(config.model_name, "test_model")
        self.assertEqual(config.num_layers, 32)
        self.assertEqual(config.hidden_dim, 4096)
        self.assertEqual(config.num_experts, 8)
        self.assertEqual(config.max_seq_length, 2048)
        self.assertEqual(config.dtype, torch.bfloat16)
    
    def test_extract_model_config_default_max_seq_length(self):
        config = extract_model_config(self.mock_model, "test_model")
        
        self.assertEqual(config.max_seq_length, 2048)


class TestDetectMoELayers(unittest.TestCase):
    
    def test_detect_moe_layers_with_block_sparse_moe(self):
        mock_model = Mock()
        mock_model.model = Mock()
        mock_model.model.layers = []
        
        for i in range(5):
            layer = Mock()
            layer.block_sparse_moe = Mock()
            layer.block_sparse_moe.gate = Mock()
            mock_model.model.layers.append(layer)
        
        moe_layers = detect_moe_layers(mock_model)
        
        self.assertEqual(len(moe_layers), 5)
        self.assertEqual(moe_layers, [0, 1, 2, 3, 4])
    
    def test_detect_moe_layers_with_mlp_gate(self):
        mock_model = Mock()
        mock_model.model = Mock()
        mock_model.model.layers = []
        
        for i in range(3):
            layer = Mock()
            layer.mlp = Mock()
            layer.mlp.gate = Mock()
            mock_model.model.layers.append(layer)
        
        moe_layers = detect_moe_layers(mock_model)
        
        self.assertEqual(len(moe_layers), 3)
        self.assertEqual(moe_layers, [0, 1, 2])
    
    def test_detect_moe_layers_mixed(self):
        mock_model = Mock()
        mock_model.model = Mock()
        mock_model.model.layers = []
        
        for i in range(5):
            layer = Mock()
            if i % 2 == 0:
                layer.block_sparse_moe = Mock()
                layer.block_sparse_moe.gate = Mock()
            else:
                layer.mlp = Mock()
                layer.mlp.gate = Mock()
            mock_model.model.layers.append(layer)
        
        moe_layers = detect_moe_layers(mock_model)
        
        self.assertEqual(len(moe_layers), 5)
        self.assertEqual(moe_layers, [0, 1, 2, 3, 4])
    
    def test_detect_moe_layers_no_moe(self):
        mock_model = Mock()
        mock_model.model = Mock()
        mock_model.model.layers = []
        
        for i in range(3):
            layer = Mock(spec=[])
            mock_model.model.layers.append(layer)
        
        moe_layers = detect_moe_layers(mock_model)
        
        self.assertEqual(len(moe_layers), 0)


class TestGetMoELayerInfo(unittest.TestCase):
    
    def test_get_moe_layer_info_block_sparse_moe(self):
        mock_model = Mock()
        mock_model.model = Mock()
        mock_model.model.layers = []
        
        layer = Mock()
        layer.input_layernorm = Mock()
        layer.block_sparse_moe = Mock()
        layer.block_sparse_moe.gate = Mock()
        mock_model.model.layers.append(layer)
        
        info = get_moe_layer_info(mock_model, 0)
        
        self.assertIsNotNone(info)
        self.assertEqual(info['type'], 'direct_block_sparse_moe')
        self.assertEqual(info['gate_path'], 'layer.block_sparse_moe.gate')
        self.assertEqual(info['input_layernorm_path'], 'layer.input_layernorm')
    
    def test_get_moe_layer_info_mlp_gate(self):
        mock_model = Mock()
        mock_model.model = Mock()
        mock_model.model.layers = []
        
        layer = Mock(spec=[])
        layer.input_layernorm = Mock()
        layer.mlp = Mock()
        layer.mlp.gate = Mock()
        mock_model.model.layers.append(layer)
        
        info = get_moe_layer_info(mock_model, 0)
        
        self.assertIsNotNone(info)
        self.assertEqual(info['type'], 'mlp_gate')
        self.assertEqual(info['gate_path'], 'layer.mlp.gate')
        self.assertEqual(info['input_layernorm_path'], 'layer.input_layernorm')
    
    def test_get_moe_layer_info_no_moe(self):
        mock_model = Mock()
        mock_model.model = Mock()
        mock_model.model.layers = []
        
        layer = Mock(spec=[])
        mock_model.model.layers.append(layer)
        
        info = get_moe_layer_info(mock_model, 0)
        
        self.assertIsNone(info)


class TestCalculateMemoryUsage(unittest.TestCase):
    
    def test_calculate_memory_usage_attn_gate(self):
        memory = calculate_memory_usage(
            num_samples=10,
            num_layers=32,
            hidden_dim=4096,
            num_experts=8,
            seq_length=100,
            pattern="attn_gate",
            dtype=DTYPE
        )
        
        self.assertIn('attn_hidden_states', memory)
        self.assertIn('gate_logits', memory)
        self.assertIn('tokens', memory)
        self.assertIn('total', memory)
        
        expected_attn = 10 * 32 * 100 * 4096 * 2
        expected_gate = 10 * 32 * 100 * 8 * 2
        expected_tokens = 10 * 100 * 4
        expected_total = expected_attn + expected_gate + expected_tokens
        
        self.assertEqual(memory['attn_hidden_states'], expected_attn)
        self.assertEqual(memory['gate_logits'], expected_gate)
        self.assertEqual(memory['tokens'], expected_tokens)
        self.assertEqual(memory['total'], expected_total)
    
    def test_calculate_memory_usage_gate_input(self):
        memory = calculate_memory_usage(
            num_samples=10,
            num_layers=32,
            hidden_dim=4096,
            num_experts=8,
            seq_length=100,
            pattern="gate_input",
            dtype=DTYPE
        )
        
        self.assertIn('gate_inputs', memory)
        self.assertIn('gate_logits', memory)
        self.assertIn('tokens', memory)
        self.assertIn('total', memory)
        
        expected_gate_input = 10 * 32 * 100 * 4096 * 2
        expected_gate = 10 * 32 * 100 * 8 * 2
        expected_tokens = 10 * 100 * 4
        expected_total = expected_gate_input + expected_gate + expected_tokens
        
        self.assertEqual(memory['gate_inputs'], expected_gate_input)
        self.assertEqual(memory['gate_logits'], expected_gate)
        self.assertEqual(memory['tokens'], expected_tokens)
        self.assertEqual(memory['total'], expected_total)
    
    def test_calculate_memory_usage_token_gate(self):
        memory = calculate_memory_usage(
            num_samples=10,
            num_layers=32,
            hidden_dim=4096,
            num_experts=8,
            seq_length=100,
            pattern="token_gate",
            dtype=DTYPE
        )
        
        self.assertIn('gate_logits', memory)
        self.assertIn('tokens', memory)
        self.assertIn('total', memory)
        
        expected_gate = 10 * 32 * 100 * 8 * 2
        expected_tokens = 10 * 100 * 4
        expected_total = expected_gate + expected_tokens
        
        self.assertEqual(memory['gate_logits'], expected_gate)
        self.assertEqual(memory['tokens'], expected_tokens)
        self.assertEqual(memory['total'], expected_total)
    
    def test_calculate_memory_usage_float16(self):
        memory = calculate_memory_usage(
            num_samples=10,
            num_layers=32,
            hidden_dim=4096,
            num_experts=8,
            seq_length=100,
            pattern="token_gate",
            dtype=torch.float16
        )
        
        self.assertEqual(memory['total'], 10 * 32 * 100 * 8 * 2 + 10 * 100 * 4)


class TestEstimateBufferCapacity(unittest.TestCase):
    
    def test_estimate_buffer_capacity_attn_gate(self):
        capacity = estimate_buffer_capacity(
            buffer_size_gb=4.0,
            num_layers=32,
            hidden_dim=4096,
            num_experts=8,
            avg_seq_length=100,
            pattern="attn_gate",
            dtype=DTYPE
        )
        
        self.assertGreater(capacity, 0)
        self.assertIsInstance(capacity, int)
    
    def test_estimate_buffer_capacity_token_gate(self):
        capacity = estimate_buffer_capacity(
            buffer_size_gb=4.0,
            num_layers=32,
            hidden_dim=4096,
            num_experts=8,
            avg_seq_length=100,
            pattern="token_gate",
            dtype=DTYPE
        )
        
        self.assertGreater(capacity, 0)
        self.assertIsInstance(capacity, int)
    
    def test_estimate_buffer_capacity_small_buffer(self):
        capacity_small = estimate_buffer_capacity(
            buffer_size_gb=1.0,
            num_layers=32,
            hidden_dim=4096,
            num_experts=8,
            avg_seq_length=100,
            pattern="attn_gate",
            dtype=DTYPE
        )
        
        capacity_large = estimate_buffer_capacity(
            buffer_size_gb=4.0,
            num_layers=32,
            hidden_dim=4096,
            num_experts=8,
            avg_seq_length=100,
            pattern="attn_gate",
            dtype=DTYPE
        )
        
        self.assertLess(capacity_small, capacity_large)


if __name__ == "__main__":
    unittest.main()
