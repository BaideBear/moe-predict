import unittest
import torch
import json
import threading
import time
from test_config import DEVICE, DTYPE
from online_sample.data_structures import ModelConfig, ActivationPattern, ActivationData
from online_sample.buffer import ActivationBuffer, create_buffer
from online_sample.sampler import OnlineSampler
from online_sample.utils import extract_model_config


class TestOnlineSampler(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.model_path = "/data1/gx/MoE-predict/models/Mixtral-8x7B-v0.1"
        cls.dataset_path = "/data1/gx/MoE-predict/dataset/processed/test/mmlu.jsonl"
        
        print(f"\nModel path: {cls.model_path}")
        print(f"Dataset path: {cls.dataset_path}")
    
    def setUp(self):
        pass
    
    def test_sampler_initialization(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print("\nLoading model...")
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
        
        model_config = extract_model_config(model, "Mixtral-8x7B-v0.1", 2048)
        
        print(f"Model config: {model_config.num_layers} layers, {model_config.hidden_dim} dim, {model_config.num_experts} experts")
        
        buffer = create_buffer(
            model_config=model_config,
            pattern="attn_gate",
            buffer_size_gb=1.0,
            device=DEVICE
        )
        
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
        
        self.assertIsNotNone(sampler)
        self.assertEqual(sampler.pattern, "attn_gate")
        self.assertEqual(sampler.batch_size, 1)
        self.assertEqual(sampler.max_seq_length, 2048)
        
        buffer.stop()
        del model
        del tokenizer
        torch.cuda.empty_cache()
    
    def test_sampler_with_real_model_small_dataset(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print("\nLoading model...")
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
        
        model_config = extract_model_config(model, "Mixtral-8x7B-v0.1", 2048)
        
        print(f"Model config: {model_config.num_layers} layers, {model_config.hidden_dim} dim, {model_config.num_experts} experts")
        
        buffer = create_buffer(
            model_config=model_config,
            pattern="attn_gate",
            buffer_size_gb=2.0,
            device=DEVICE
        )
        
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
        
        print("Starting sampler...")
        sampler.start()
        
        print("Waiting for samples to be written to buffer...")
        start_time = time.time()
        timeout = 60
        
        while time.time() - start_time < timeout:
            if buffer.get_size() >= 5:
                print(f"Collected {buffer.get_size()} samples")
                break
            time.sleep(1)
        
        self.assertGreater(buffer.get_size(), 0, "No samples were written to buffer")
        
        print(f"Stopping sampler...")
        sampler.stop()
        
        print(f"Buffer size: {buffer.get_size()} samples")
        stats = buffer.get_stats()
        print(f"Buffer stats: {stats.used_memory_gb:.2f} GB / {stats.buffer_size_gb:.2f} GB")
        
        buffer.stop()
        del model
        del tokenizer
        torch.cuda.empty_cache()
    
    def test_sampler_data_validation(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print("\nLoading model...")
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
        
        model_config = extract_model_config(model, "Mixtral-8x7B-v0.1", 2048)
        
        buffer = create_buffer(
            model_config=model_config,
            pattern="attn_gate",
            buffer_size_gb=2.0,
            device=DEVICE
        )
        
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
        
        print("Starting sampler...")
        sampler.start()
        
        print("Waiting for samples...")
        start_time = time.time()
        timeout = 60
        
        while time.time() - start_time < timeout:
            if buffer.get_size() >= 3:
                break
            time.sleep(1)
        
        self.assertGreater(buffer.get_size(), 0, "No samples were written to buffer")
        
        print(f"Reading {buffer.get_size()} samples from buffer...")
        batch = buffer.read(batch_size=3, timeout=10.0)
        
        self.assertIsNotNone(batch, "Failed to read samples from buffer")
        self.assertEqual(len(batch), 3, f"Expected 3 samples, got {len(batch)}")
        
        for i, data in enumerate(batch):
            print(f"\nSample {i}:")
            print(f"  Tokens shape: {data.tokens.shape}")
            print(f"  Gate logits shape: {data.gate_logits.shape}")
            print(f"  Attn hidden states shape: {data.attn_hidden_states.shape}")
            print(f"  Seq lengths shape: {data.seq_lengths.shape}")
            
            self.assertIsNotNone(data.tokens)
            self.assertIsNotNone(data.gate_logits)
            self.assertIsNotNone(data.attn_hidden_states)
            self.assertIsNotNone(data.seq_lengths)
            
            self.assertTrue(data.validate("attn_gate"))
            
            self.assertEqual(data.tokens.device.type, DEVICE)
            self.assertEqual(data.gate_logits.device.type, DEVICE)
            self.assertEqual(data.attn_hidden_states.device.type, DEVICE)
        
        print("\nStopping sampler...")
        sampler.stop()
        
        buffer.stop()
        del model
        del tokenizer
        torch.cuda.empty_cache()
    
    def test_sampler_concurrent_read_write(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print("\nLoading model...")
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
        
        model_config = extract_model_config(model, "Mixtral-8x7B-v0.1", 2048)
        
        buffer = create_buffer(
            model_config=model_config,
            pattern="attn_gate",
            buffer_size_gb=2.0,
            device=DEVICE
        )
        
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
        
        print("Starting sampler...")
        sampler.start()
        
        read_count = 0
        errors = []
        
        def reader():
            nonlocal read_count
            try:
                while sampler.is_running() or buffer.get_size() > 0:
                    batch = buffer.read(batch_size=1, timeout=2.0)
                    if batch is not None:
                        read_count += len(batch)
                        print(f"Read {read_count} samples")
            except Exception as e:
                errors.append(f"Reader error: {e}")
        
        reader_thread = threading.Thread(target=reader)
        reader_thread.start()
        
        start_time = time.time()
        timeout = 60
        
        while time.time() - start_time < timeout:
            if read_count >= 5:
                print(f"Successfully read {read_count} samples")
                break
            time.sleep(1)
        
        print("Stopping sampler...")
        sampler.stop()
        
        reader_thread.join(timeout=10.0)
        
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")
        self.assertGreater(read_count, 0, "No samples were read")
        
        buffer.stop()
        del model
        del tokenizer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
