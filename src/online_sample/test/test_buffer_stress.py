import unittest
import torch
import threading
import time
from test_config import DEVICE, DTYPE
from online_sample.data_structures import ModelConfig, ActivationData
from online_sample.buffer import ActivationBuffer


class TestBufferHighLoad(unittest.TestCase):
    
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
    
    def test_large_volume_write_read(self):
        num_samples = 50
        
        for i in range(num_samples):
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
            self.assertTrue(success, f"Failed to write sample {i}")
        
        self.assertEqual(self.buffer.get_size(), num_samples)
        
        read_count = 0
        while read_count < num_samples:
            batch = self.buffer.read(batch_size=10)
            if batch is None:
                break
            read_count += len(batch)
        
        self.assertEqual(read_count, num_samples)
        self.assertTrue(self.buffer.is_empty())
    
    def test_buffer_full_blocking(self):
        small_buffer = ActivationBuffer(
            model_config=self.model_config,
            pattern="attn_gate",
            buffer_size_gb=0.1,
            device=DEVICE
        )
        
        try:
            for i in range(1000):
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
            
            self.assertGreater(i, 0, "Should have written at least some samples")
            self.assertTrue(small_buffer.is_full())
        
        finally:
            small_buffer.stop()
    
    def test_concurrent_multiple_writers_readers(self):
        num_writers = 2
        num_readers = 1
        samples_per_writer = 10
        errors = []
        write_counts = [0] * num_writers
        read_counts = [0] * num_readers
        
        def writer(writer_id):
            nonlocal write_counts
            for i in range(samples_per_writer):
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
                    write_counts[writer_id] += 1
                    time.sleep(0.01)
                except Exception as e:
                    errors.append(f"Writer {writer_id} error: {e}")
        
        def reader(reader_id):
            nonlocal read_counts
            for i in range(samples_per_writer * num_writers // num_readers + 10):
                try:
                    batch = self.buffer.read(batch_size=2, timeout=1.0)
                    if batch is not None:
                        read_counts[reader_id] += len(batch)
                    time.sleep(0.01)
                except Exception as e:
                    errors.append(f"Reader {reader_id} error: {e}")
        
        threads = []
        for i in range(num_writers):
            t = threading.Thread(target=writer, args=(i,))
            threads.append(t)
        
        for i in range(num_readers):
            t = threading.Thread(target=reader, args=(i,))
            threads.append(t)
        
        start_time = time.time()
        for t in threads:
            t.start()
        
        for t in threads:
            t.join(timeout=30.0)
        
        end_time = time.time()
        
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")
        total_written = sum(write_counts)
        total_read = sum(read_counts)
        
        print(f"Total written: {total_written}, Total read: {total_read}")
        print(f"Write counts: {write_counts}")
        print(f"Read counts: {read_counts}")
        print(f"Time elapsed: {end_time - start_time:.2f}s")
        
        self.assertGreater(total_written, 0)
        self.assertGreater(total_read, 0)
    
    def test_rapid_write_read_cycles(self):
        num_cycles = 20
        
        for cycle in range(num_cycles):
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
            
            if cycle % 10 == 0:
                print(f"Completed {cycle} cycles")
    
    def test_memory_efficiency(self):
        initial_stats = self.buffer.get_stats()
        
        for i in range(20):
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
        
        final_stats = self.buffer.get_stats()
        
        self.assertGreater(final_stats.used_memory_gb, initial_stats.used_memory_gb)
        self.assertLess(final_stats.used_memory_gb, self.buffer_size_gb)
        
        utilization = final_stats.used_memory_gb / self.buffer_size_gb
        print(f"Buffer utilization: {utilization*100:.2f}%")
        
        self.assertGreater(utilization, 0)
        self.assertLess(utilization, 1.0)
    
    def test_stress_test(self):
        duration_seconds = 5
        start_time = time.time()
        write_count = 0
        read_count = 0
        errors = []
        
        def stress_writer():
            nonlocal write_count
            while time.time() - start_time < duration_seconds:
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
                    
                    if self.buffer.write(data, timeout=0.1):
                        write_count += 1
                except Exception as e:
                    errors.append(f"Writer error: {e}")
        
        def stress_reader():
            nonlocal read_count
            while time.time() - start_time < duration_seconds:
                try:
                    batch = self.buffer.read(batch_size=1, timeout=0.1)
                    if batch is not None:
                        read_count += len(batch)
                except Exception as e:
                    errors.append(f"Reader error: {e}")
        
        writer_thread = threading.Thread(target=stress_writer)
        reader_thread = threading.Thread(target=stress_reader)
        
        writer_thread.start()
        reader_thread.start()
        
        writer_thread.join(timeout=duration_seconds + 5)
        reader_thread.join(timeout=duration_seconds + 5)
        
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")
        
        print(f"Stress test results:")
        print(f"  Duration: {duration_seconds}s")
        print(f"  Writes: {write_count}")
        print(f"  Reads: {read_count}")
        print(f"  Write rate: {write_count/duration_seconds:.2f} samples/s")
        print(f"  Read rate: {read_count/duration_seconds:.2f} samples/s")
        
        self.assertGreater(write_count, 0)
        self.assertGreater(read_count, 0)


if __name__ == "__main__":
    unittest.main()
