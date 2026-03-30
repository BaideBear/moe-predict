import unittest
import torch
import threading
import time
from test_config import DEVICE, DTYPE
from transformers import AutoModelForCausalLM, AutoTokenizer

from online_sample.data_structures import ModelConfig
from online_sample.buffer import create_buffer
from online_sample.sampler import OnlineSampler
from online_sample.predictor_interface import create_predictor_interface
from online_sample.utils import extract_model_config


class TestIntegration(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.model_path = "/data1/gx/MoE-predict/models/Mixtral-8x7B-v0.1"
        cls.dataset_path = "/data1/gx/MoE-predict/dataset/processed/test/mmlu.jsonl"
        
        print(f"\nModel path: {cls.model_path}")
        print(f"Dataset path: {cls.dataset_path}")
    
    def test_full_pipeline_basic(self):
        print("\n=== Testing Full Pipeline (Basic) ===")
        
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="eager"
        )
        model.eval()
        
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("Extracting model config...")
        model_config = extract_model_config(model, "Mixtral-8x7B-v0.1", 2048)
        
        print(f"Model config: {model_config.num_layers} layers, {model_config.hidden_dim} dim, {model_config.num_experts} experts")
        
        print("Creating buffer...")
        buffer = create_buffer(
            model_config=model_config,
            pattern="attn_gate",
            buffer_size_gb=2.0,
            device=DEVICE
        )
        
        print("Creating sampler...")
        sampler = OnlineSampler(
            model=model,
            tokenizer=tokenizer,
            dataset_path=self.dataset_path,
            buffer=buffer,
            pattern="attn_gate",
            batch_size=1,
            max_seq_length=2048,
            trust_remote_code=True
        )
        
        print("Creating predictor interface...")
        interface = create_predictor_interface(
            buffer=buffer,
            pattern="attn_gate",
            batch_size=1,
            timeout=None
        )
        
        print("Starting sampler...")
        sampler.start()
        
        print("Waiting for samples...")
        start_time = time.time()
        timeout = 60
        
        while time.time() - start_time < timeout:
            if buffer.get_size() >= 3:
                print(f"Collected {buffer.get_size()} samples")
                break
            time.sleep(1)
        
        self.assertGreater(buffer.get_size(), 0, "No samples were written to buffer")
        
        print("Reading samples from interface...")
        batch = interface.get_batch()
        
        self.assertIsNotNone(batch, "Failed to read samples from interface")
        self.assertEqual(len(batch), 1, f"Expected 1 sample, got {len(batch)}")
        
        data = batch[0]
        print(f"Sample shapes:")
        print(f"  Tokens: {data.tokens.shape}")
        print(f"  Gate logits: {data.gate_logits.shape}")
        print(f"  Attn hidden states: {data.attn_hidden_states.shape}")
        
        self.assertIsNotNone(data.tokens)
        self.assertIsNotNone(data.gate_logits)
        self.assertIsNotNone(data.attn_hidden_states)
        
        self.assertTrue(data.validate("attn_gate"))
        
        print("Stopping sampler...")
        sampler.stop()
        
        buffer.stop()
        
        del model
        del tokenizer
        torch.cuda.empty_cache()
        
        print("=== Full Pipeline Test Passed ===\n")
    
    def test_full_pipeline_concurrent(self):
        print("\n=== Testing Full Pipeline (Concurrent) ===")
        
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="eager"
        )
        model.eval()
        
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("Extracting model config...")
        model_config = extract_model_config(model, "Mixtral-8x7B-v0.1", 2048)
        
        print(f"Model config: {model_config.num_layers} layers, {model_config.hidden_dim} dim, {model_config.num_experts} experts")
        
        print("Creating buffer...")
        buffer = create_buffer(
            model_config=model_config,
            pattern="attn_gate",
            buffer_size_gb=2.0,
            device=DEVICE
        )
        
        print("Creating sampler...")
        sampler = OnlineSampler(
            model=model,
            tokenizer=tokenizer,
            dataset_path=self.dataset_path,
            buffer=buffer,
            pattern="attn_gate",
            batch_size=1,
            max_seq_length=2048,
            trust_remote_code=True
        )
        
        print("Creating predictor interface...")
        interface = create_predictor_interface(
            buffer=buffer,
            pattern="attn_gate",
            batch_size=1,
            timeout=None
        )
        
        print("Starting sampler...")
        sampler.start()
        
        read_count = 0
        errors = []
        read_samples = []
        
        def reader():
            nonlocal read_count
            try:
                while sampler.is_running() or buffer.get_size() > 0:
                    batch = interface.get_batch()
                    if batch is not None:
                        read_count += len(batch)
                        read_samples.extend(batch)
                        print(f"Read {read_count} samples total")
            except Exception as e:
                errors.append(f"Reader error: {e}")
        
        reader_thread = threading.Thread(target=reader)
        reader_thread.start()
        
        print("Waiting for concurrent operations...")
        start_time = time.time()
        timeout = 60
        
        while time.time() - start_time < timeout:
            if read_count >= 5:
                print(f"Successfully read {read_count} samples concurrently")
                break
            time.sleep(1)
        
        print("Stopping sampler...")
        sampler.stop()
        
        reader_thread.join(timeout=10.0)
        
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")
        self.assertGreater(read_count, 0, "No samples were read")
        
        print(f"Validating {len(read_samples)} read samples...")
        for i, data in enumerate(read_samples[:3]):
            self.assertTrue(data.validate("attn_gate"), f"Sample {i} validation failed")
        
        buffer.stop()
        
        del model
        del tokenizer
        torch.cuda.empty_cache()
        
        print("=== Concurrent Pipeline Test Passed ===\n")
    
    def test_full_pipeline_buffer_management(self):
        print("\n=== Testing Full Pipeline (Buffer Management) ===")
        
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="eager"
        )
        model.eval()
        
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("Extracting model config...")
        model_config = extract_model_config(model, "Mixtral-8x7B-v0.1", 2048)
        
        print("Creating small buffer...")
        buffer = create_buffer(
            model_config=model_config,
            pattern="attn_gate",
            buffer_size_gb=1.0,
            device=DEVICE
        )
        
        print("Creating sampler...")
        sampler = OnlineSampler(
            model=model,
            tokenizer=tokenizer,
            dataset_path=self.dataset_path,
            buffer=buffer,
            pattern="attn_gate",
            batch_size=1,
            max_seq_length=2048,
            trust_remote_code=True
        )
        
        print("Creating predictor interface...")
        interface = create_predictor_interface(
            buffer=buffer,
            pattern="attn_gate",
            batch_size=1,
            timeout=None
        )
        
        print("Starting sampler...")
        sampler.start()
        
        print("Monitoring buffer management...")
        max_buffer_size = 0
        read_count = 0
        
        start_time = time.time()
        timeout = 60
        
        while time.time() - start_time < timeout:
            current_size = buffer.get_size()
            if current_size > max_buffer_size:
                max_buffer_size = current_size
            
            stats = interface.get_stats()
            print(f"Buffer: {current_size} samples, {stats['used_memory_gb']:.2f} GB / {stats['buffer_size_gb']:.2f} GB ({stats['utilization']*100:.1f}%)")
            
            batch = interface.get_batch()
            if batch is not None:
                read_count += len(batch)
                print(f"Read {read_count} samples total")
            
            if read_count >= 5:
                print(f"Successfully read {read_count} samples")
                break
            
            time.sleep(1)
        
        self.assertGreater(read_count, 0, "No samples were read")
        self.assertGreater(max_buffer_size, 0, "Buffer never filled")
        
        print("Stopping sampler...")
        sampler.stop()
        
        buffer.stop()
        
        del model
        del tokenizer
        torch.cuda.empty_cache()
        
        print("=== Buffer Management Test Passed ===\n")


if __name__ == "__main__":
    unittest.main()
