import unittest
import torch
import threading
import time
from test_config import DEVICE, DTYPE
from online_sample.data_structures import ModelConfig, ActivationPattern, ActivationData
from online_sample.buffer import ActivationBuffer, create_buffer


class TestActivationBuffer(unittest.TestCase):
    
    def setUp(self):
        self.model_config = ModelConfig(
            model_name="test_model",
            num_layers=32,
            hidden_dim=4096,
            num_experts=8,
            max_seq_length=2048,
            dtype=DTYPE
        )
        self.buffer_size_gb = 1.0
        self.buffer = ActivationBuffer(
            model_config=self.model_config,
            pattern="attn_gate",
            buffer_size_gb=self.buffer_size_gb,
            device=DEVICE
        )
    
    def tearDown(self):
        self.buffer.stop()
    
    def test_buffer_initialization(self):
        self.assertEqual(self.buffer.model_config, self.model_config)
        self.assertEqual(self.buffer.pattern, "attn_gate")
        self.assertEqual(self.buffer.buffer_size_gb, self.buffer_size_gb)
        self.assertEqual(self.buffer.device.type, DEVICE)
        self.assertEqual(self.buffer.dtype, DTYPE)
    
    def test_buffer_initially_empty(self):
        self.assertTrue(self.buffer.is_empty())
        self.assertFalse(self.buffer.is_full())
        self.assertEqual(self.buffer.get_size(), 0)
    
    def test_write_single_data(self):
        tokens = torch.randint(0, 1000, (1, 100), dtype=torch.int32).to(DEVICE)
        gate_logits = torch.randn(1, 32, 100, 8, dtype=DTYPE).to(DEVICE)
        attn_hidden_states = torch.randn(1, 32, 100, 4096, dtype=DTYPE).to(DEVICE)
        
        data = ActivationData(
            tokens=tokens,
            gate_logits=gate_logits,
            attn_hidden_states=attn_hidden_states,
            metadata={"pattern": "attn_gate"}
        )
        
        success = self.buffer.write(data)
        
        self.assertTrue(success)
        self.assertEqual(self.buffer.get_size(), 1)
        self.assertFalse(self.buffer.is_empty())
    
    def test_write_multiple_data(self):
        for i in range(5):
            tokens = torch.randint(0, 1000, (1, 100), dtype=torch.int32).to(DEVICE)
            gate_logits = torch.randn(1, 32, 100, 8, dtype=DTYPE).to(DEVICE)
            attn_hidden_states = torch.randn(1, 32, 100, 4096, dtype=DTYPE).to(DEVICE)
            
            data = ActivationData(
                tokens=tokens,
                gate_logits=gate_logits,
                attn_hidden_states=attn_hidden_states,
                metadata={"pattern": "attn_gate"}
            )
            
            success = self.buffer.write(data)
            self.assertTrue(success)
        
        self.assertEqual(self.buffer.get_size(), 5)
    
    def test_read_single_data(self):
        tokens = torch.randint(0, 1000, (1, 100), dtype=torch.int32).to(DEVICE)
        gate_logits = torch.randn(1, 32, 100, 8, dtype=DTYPE).to(DEVICE)
        attn_hidden_states = torch.randn(1, 32, 100, 4096, dtype=DTYPE).to(DEVICE)
        
        data = ActivationData(
            tokens=tokens,
            gate_logits=gate_logits,
            attn_hidden_states=attn_hidden_states,
            metadata={"pattern": "attn_gate"}
        )
        
        self.buffer.write(data)
        
        batch = self.buffer.read(batch_size=1)
        
        self.assertIsNotNone(batch)
        self.assertEqual(len(batch), 1)
        self.assertEqual(batch[0].tokens.shape, tokens.shape)
        self.assertEqual(batch[0].gate_logits.shape, gate_logits.shape)
        self.assertEqual(batch[0].attn_hidden_states.shape, attn_hidden_states.shape)
        self.assertTrue(self.buffer.is_empty())
    
    def test_read_multiple_data(self):
        for i in range(5):
            tokens = torch.randint(0, 1000, (1, 100), dtype=torch.int32).to(DEVICE)
            gate_logits = torch.randn(1, 32, 100, 8, dtype=DTYPE).to(DEVICE)
            attn_hidden_states = torch.randn(1, 32, 100, 4096, dtype=DTYPE).to(DEVICE)
            
            data = ActivationData(
                tokens=tokens,
                gate_logits=gate_logits,
                attn_hidden_states=attn_hidden_states,
                metadata={"pattern": "attn_gate"}
            )
            
            self.buffer.write(data)
        
        batch = self.buffer.read(batch_size=3)
        
        self.assertIsNotNone(batch)
        self.assertEqual(len(batch), 3)
        self.assertEqual(self.buffer.get_size(), 2)
    
    def test_read_from_empty_buffer(self):
        batch = self.buffer.read(batch_size=1, timeout=0.1)
        
        self.assertIsNone(batch)
    
    def test_write_invalid_pattern(self):
        tokens = torch.randint(0, 1000, (1, 100), dtype=torch.int32).to(DEVICE)
        gate_logits = torch.randn(1, 32, 100, 8, dtype=DTYPE).to(DEVICE)
        
        data = ActivationData(
            tokens=tokens,
            gate_logits=gate_logits,
            metadata={"pattern": "attn_gate"}
        )
        
        with self.assertRaises(ValueError):
            self.buffer.write(data)
    
    def test_buffer_stats(self):
        for i in range(3):
            tokens = torch.randint(0, 1000, (1, 100), dtype=torch.int32).to(DEVICE)
            gate_logits = torch.randn(1, 32, 100, 8, dtype=DTYPE).to(DEVICE)
            attn_hidden_states = torch.randn(1, 32, 100, 4096, dtype=DTYPE).to(DEVICE)
            
            data = ActivationData(
                tokens=tokens,
                gate_logits=gate_logits,
                attn_hidden_states=attn_hidden_states,
                metadata={"pattern": "attn_gate"}
            )
            
            self.buffer.write(data)
        
        stats = self.buffer.get_stats()
        
        self.assertEqual(stats.total_samples, 3)
        self.assertEqual(stats.used_samples, 3)
        self.assertEqual(stats.buffer_size_gb, self.buffer_size_gb)
        self.assertGreater(stats.used_memory_gb, 0)
        self.assertLess(stats.used_memory_gb, self.buffer_size_gb)
    
    def test_mark_write_finished(self):
        self.buffer.mark_write_finished()
        
        batch = self.buffer.read(batch_size=1, timeout=0.1)
        
        self.assertIsNone(batch)
    
    def test_clear_buffer(self):
        for i in range(3):
            tokens = torch.randint(0, 1000, (1, 100), dtype=torch.int32).to(DEVICE)
            gate_logits = torch.randn(1, 32, 100, 8, dtype=DTYPE).to(DEVICE)
            attn_hidden_states = torch.randn(1, 32, 100, 4096, dtype=DTYPE).to(DEVICE)
            
            data = ActivationData(
                tokens=tokens,
                gate_logits=gate_logits,
                attn_hidden_states=attn_hidden_states,
                metadata={"pattern": "attn_gate"}
            )
            
            self.buffer.write(data)
        
        self.assertEqual(self.buffer.get_size(), 3)
        
        self.buffer.clear()
        
        self.assertTrue(self.buffer.is_empty())
        self.assertEqual(self.buffer.get_size(), 0)
    
    def test_concurrent_write_read(self):
        write_count = 0
        read_count = 0
        errors = []
        
        def writer():
            nonlocal write_count
            for i in range(10):
                try:
                    tokens = torch.randint(0, 1000, (1, 100), dtype=torch.int32).to(DEVICE)
                    gate_logits = torch.randn(1, 32, 100, 8, dtype=DTYPE).to(DEVICE)
                    attn_hidden_states = torch.randn(1, 32, 100, 4096, dtype=DTYPE).to(DEVICE)
                    
                    data = ActivationData(
                        tokens=tokens,
                        gate_logits=gate_logits,
                        attn_hidden_states=attn_hidden_states,
                        metadata={"pattern": "attn_gate"}
                    )
                    
                    self.buffer.write(data)
                    write_count += 1
                    time.sleep(0.01)
                except Exception as e:
                    errors.append(f"Write error: {e}")
        
        def reader():
            nonlocal read_count
            for i in range(10):
                try:
                    batch = self.buffer.read(batch_size=1, timeout=1.0)
                    if batch is not None:
                        read_count += len(batch)
                    time.sleep(0.01)
                except Exception as e:
                    errors.append(f"Read error: {e}")
        
        writer_thread = threading.Thread(target=writer)
        reader_thread = threading.Thread(target=reader)
        
        writer_thread.start()
        reader_thread.start()
        
        writer_thread.join(timeout=10.0)
        reader_thread.join(timeout=10.0)
        
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")
        self.assertGreater(write_count, 0)
        self.assertGreater(read_count, 0)


class TestCreateBuffer(unittest.TestCase):
    
    def test_create_buffer_factory(self):
        model_config = ModelConfig(
            model_name="test_model",
            num_layers=32,
            hidden_dim=4096,
            num_experts=8,
            max_seq_length=2048,
            dtype=DTYPE
        )
        
        buffer = create_buffer(
            model_config=model_config,
            pattern="attn_gate",
            buffer_size_gb=2.0,
            device=DEVICE
        )
        
        self.assertIsNotNone(buffer)
        self.assertEqual(buffer.buffer_size_gb, 2.0)
        self.assertEqual(buffer.pattern, "attn_gate")
        
        buffer.stop()


if __name__ == "__main__":
    unittest.main()
