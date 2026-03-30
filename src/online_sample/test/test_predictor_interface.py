import unittest
import torch
import threading
import time
from test_config import DEVICE, DTYPE
from online_sample.data_structures import ModelConfig, ActivationData
from online_sample.buffer import ActivationBuffer, create_buffer
from online_sample.predictor_interface import PredictorInterface, create_predictor_interface


class TestPredictorInterface(unittest.TestCase):
    
    def setUp(self):
        self.model_config = ModelConfig(
            model_name="test_model",
            num_layers=32,
            hidden_dim=4096,
            num_experts=8,
            max_seq_length=2048,
            dtype=DTYPE
        )
        self.buffer = ActivationBuffer(
            model_config=self.model_config,
            pattern="attn_gate",
            buffer_size_gb=1.0,
            device=DEVICE
        )
    
    def tearDown(self):
        self.buffer.stop()
    
    def test_interface_initialization(self):
        interface = PredictorInterface(
            buffer=self.buffer,
            pattern="attn_gate",
            batch_size=1,
            timeout=None
        )
        
        self.assertEqual(interface.pattern, "attn_gate")
        self.assertEqual(interface.batch_size, 1)
        self.assertIsNone(interface.timeout)
    
    def test_interface_with_timeout(self):
        interface = PredictorInterface(
            buffer=self.buffer,
            pattern="attn_gate",
            batch_size=1,
            timeout=1.0
        )
        
        self.assertEqual(interface.timeout, 1.0)
    
    def test_get_batch_from_empty_buffer(self):
        interface = PredictorInterface(
            buffer=self.buffer,
            pattern="attn_gate",
            batch_size=1,
            timeout=0.1
        )
        
        batch = interface.get_batch()
        
        self.assertIsNone(batch)
    
    def test_get_batch_with_data(self):
        interface = PredictorInterface(
            buffer=self.buffer,
            pattern="attn_gate",
            batch_size=1,
            timeout=None
        )
        
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
        
        batch = interface.get_batch()
        
        self.assertIsNotNone(batch)
        self.assertEqual(len(batch), 1)
        self.assertEqual(batch[0].tokens.shape, tokens.shape)
        self.assertEqual(batch[0].gate_logits.shape, gate_logits.shape)
        self.assertEqual(batch[0].attn_hidden_states.shape, attn_hidden_states.shape)
    
    def test_get_batch_multiple(self):
        interface = PredictorInterface(
            buffer=self.buffer,
            pattern="attn_gate",
            batch_size=3,
            timeout=None
        )
        
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
        
        batch = interface.get_batch()
        
        self.assertIsNotNone(batch)
        self.assertEqual(len(batch), 3)
        self.assertEqual(self.buffer.get_size(), 2)
    
    def test_get_batch_timeout(self):
        interface = PredictorInterface(
            buffer=self.buffer,
            pattern="attn_gate",
            batch_size=1,
            timeout=0.1
        )
        
        batch = interface.get_batch()
        
        self.assertIsNone(batch)
    
    def test_get_stats(self):
        interface = PredictorInterface(
            buffer=self.buffer,
            pattern="attn_gate",
            batch_size=1,
            timeout=None
        )
        
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
        
        stats = interface.get_stats()
        
        self.assertEqual(stats['total_samples'], 3)
        self.assertEqual(stats['used_samples'], 3)
        self.assertEqual(stats['buffer_size_gb'], 1.0)
        self.assertGreater(stats['used_memory_gb'], 0)
        self.assertLess(stats['used_memory_gb'], 1.0)
        self.assertGreater(stats['utilization'], 0)
        self.assertLess(stats['utilization'], 1.0)
    
    def test_is_buffer_empty(self):
        interface = PredictorInterface(
            buffer=self.buffer,
            pattern="attn_gate",
            batch_size=1,
            timeout=None
        )
        
        self.assertTrue(interface.is_buffer_empty())
        
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
        
        self.assertFalse(interface.is_buffer_empty())
    
    def test_is_buffer_full(self):
        small_buffer = ActivationBuffer(
            model_config=self.model_config,
            pattern="attn_gate",
            buffer_size_gb=0.1,
            device=DEVICE
        )
        
        interface = PredictorInterface(
            buffer=small_buffer,
            pattern="attn_gate",
            batch_size=1,
            timeout=None
        )
        
        self.assertFalse(interface.is_buffer_full())
        
        for i in range(100):
            tokens = torch.randint(0, 1000, (1, 100), dtype=torch.int32).to(DEVICE)
            gate_logits = torch.randn(1, 32, 100, 8, dtype=DTYPE).to(DEVICE)
            attn_hidden_states = torch.randn(1, 32, 100, 4096, dtype=DTYPE).to(DEVICE)
            
            data = ActivationData(
                tokens=tokens,
                gate_logits=gate_logits,
                attn_hidden_states=attn_hidden_states,
                metadata={"pattern": "attn_gate"}
            )
            
            success = small_buffer.write(data, timeout=0.1)
            if not success:
                print(f"Buffer became full at sample {i}")
                break
        
        if small_buffer.get_size() > 0:
            print(f"Buffer has {small_buffer.get_size()} samples")
            print(f"Buffer is full: {small_buffer.is_full()}")
        
        self.assertTrue(small_buffer.get_size() > 0, "Buffer should have some samples")
        
        small_buffer.stop()
    
    def test_wait_for_data(self):
        interface = PredictorInterface(
            buffer=self.buffer,
            pattern="attn_gate",
            batch_size=1,
            timeout=None
        )
        
        result = interface.wait_for_data(min_samples=3, timeout=5.0)
        
        self.assertFalse(result, "Should timeout waiting for data")
        
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
        
        result = interface.wait_for_data(min_samples=3, timeout=1.0)
        
        self.assertTrue(result, "Should succeed waiting for data")
    
    def test_concurrent_read_write(self):
        interface = PredictorInterface(
            buffer=self.buffer,
            pattern="attn_gate",
            batch_size=1,
            timeout=None
        )
        
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
                    batch = interface.get_batch()
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


class TestCreatePredictorInterface(unittest.TestCase):
    
    def test_create_predictor_interface_factory(self):
        model_config = ModelConfig(
            model_name="test_model",
            num_layers=32,
            hidden_dim=4096,
            num_experts=8,
            max_seq_length=2048,
            dtype=DTYPE
        )
        
        buffer = ActivationBuffer(
            model_config=model_config,
            pattern="attn_gate",
            buffer_size_gb=1.0,
            device=DEVICE
        )
        
        interface = create_predictor_interface(
            buffer=buffer,
            pattern="attn_gate",
            batch_size=1,
            timeout=None
        )
        
        self.assertIsNotNone(interface)
        self.assertEqual(interface.pattern, "attn_gate")
        self.assertEqual(interface.batch_size, 1)
        
        buffer.stop()


if __name__ == "__main__":
    unittest.main()
