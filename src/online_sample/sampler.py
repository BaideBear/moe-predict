import torch
import json
import threading
import os
from typing import Dict, List, Optional, Any
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .buffer import ActivationBuffer
from .data_structures import ModelConfig, ActivationPattern, ActivationData
from .utils import extract_model_config, detect_moe_layers, get_moe_layer_info


class OnlineSampler:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        dataset_path: str,
        buffer: ActivationBuffer,
        pattern: str,
        batch_size: int = 1,
        max_seq_length: int = 2048,
        trust_remote_code: bool = True,
        epochs: int = 1,
        start_sample: int = 0
    ):
        self.model = model
        self.tokenizer = tokenizer

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.dataset_path = dataset_path
        self.buffer = buffer
        self.pattern = pattern
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.trust_remote_code = trust_remote_code
        self.epochs = epochs
        self.start_sample = start_sample
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_config = extract_model_config(model, buffer.model_config.model_name, max_seq_length)
        self.moe_layers = detect_moe_layers(model)
        self.moe_layer_info = {idx: get_moe_layer_info(model, idx) for idx in self.moe_layers}

        self._thread: Optional[threading.Thread] = None
        self._is_running = False
        self._stop_event = threading.Event()

        self._init_buffers()
        self._register_hooks()

        print(f"OnlineSampler initialized:")
        print(f"  Pattern: {pattern}")
        print(f"  Batch size: {batch_size}")
        print(f"  Max seq length: {max_seq_length}")
        print(f"  Epochs: {epochs}")
        print(f"  Start sample: {start_sample}")
        print(f"  MoE layers: {self.moe_layers}")

    def _init_buffers(self):
        self.buffer_gate_outputs = {layer_idx: [] for layer_idx in self.moe_layers}
        self.buffer_gate_inputs = {layer_idx: [] for layer_idx in self.moe_layers}
        self.buffer_attn_hidden_states = {layer_idx: [] for layer_idx in self.moe_layers}
        self.buffer_tokens = []
        self.buffer_seq_lengths = []

    def _register_hooks(self):
        self.handles = []

        for layer_idx in self.moe_layers:
            layer = self.model.model.layers[layer_idx]
            moe_info = self.moe_layer_info[layer_idx]

            if hasattr(layer, 'input_layernorm'):
                handle = layer.input_layernorm.register_forward_hook(
                    self._create_attn_hidden_state_hook(layer_idx)
                )
                self.handles.append(handle)

            gate_module = moe_info['gate_module']
            handle = gate_module.register_forward_hook(
                self._create_gate_hook(layer_idx)
            )
            self.handles.append(handle)

        print(f"Registered {len(self.handles)} hooks")

    def _create_attn_hidden_state_hook(self, layer_idx: int):
        def hook(module, input, output):
            hidden_state = output[0].detach()
            self.buffer_attn_hidden_states[layer_idx].append(hidden_state)
        return hook

    def _create_gate_hook(self, layer_idx: int):
        def hook(module, input, output):
            # 针对deepseek的special case
            if hasattr(module, '_intercepted_scores') and module._intercepted_scores is not None:
                gate_output = module._intercepted_scores.detach()
            else:
                if isinstance(output, tuple):
                    router_output = output[0]
                else:
                    router_output = output
                
                gate_output = router_output.detach()
            
            self.buffer_gate_outputs[layer_idx].append(gate_output)

            if len(input) > 0:
                gate_input = input[0].detach()
                self.buffer_gate_inputs[layer_idx].append(gate_input)
        return hook

    def _clear_buffers(self):
        for layer_idx in self.moe_layers:
            self.buffer_gate_outputs[layer_idx].clear()
            self.buffer_gate_inputs[layer_idx].clear()
            self.buffer_attn_hidden_states[layer_idx].clear()
        self.buffer_tokens.clear()
        self.buffer_seq_lengths.clear()

    def _load_dataset(self) -> List[Dict]:
        print(f"Loading dataset from: {self.dataset_path}...")
        data = []

        # 检查是文件还是目录
        if os.path.isdir(self.dataset_path):
            print(f"  Detected directory: {self.dataset_path}. Loading all .jsonl files...")
            files = sorted([f for f in os.listdir(self.dataset_path) if f.endswith('.jsonl')])
            if not files:
                print(f"  ERROR: No .jsonl files found in {self.dataset_path}")
                return []

            for filename in files:
                file_path = os.path.join(self.dataset_path, filename)
                print(f"    Loading {filename}...")
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data.append(json.loads(line))
        elif os.path.isfile(self.dataset_path):
            print(f"  Detected file: {self.dataset_path}. Loading...")
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))
        else:
            print(f"  ERROR: dataset_path {self.dataset_path} is neither a file nor a directory")
            return []

        print(f"Loaded {len(data)} samples in total")
        return data

    def _tokenize_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt"
        )

        attention_mask = encodings['attention_mask']
        input_ids = encodings['input_ids']

        seq_lengths = attention_mask.sum(dim=1).tolist()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'seq_lengths': seq_lengths
        }

    def _process_batch(self, batch_data: List[Dict], batch_idx: int):
        texts = [item['text'] for item in batch_data]

        tokenized = self._tokenize_batch(texts)
        input_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device)
        seq_lengths = tokenized['seq_lengths']

        with torch.no_grad():
            _ = self.model(input_ids=input_ids, attention_mask=attention_mask)

        for i, seq_len in enumerate(seq_lengths):
            self.buffer_tokens.append(input_ids[i, :seq_len].clone())
            self.buffer_seq_lengths.append(seq_len)

    def _create_activation_data(self) -> ActivationData:
        num_samples = len(self.buffer_tokens)
        num_layers = len(self.moe_layers)

        if num_samples == 0:
            raise ValueError("No samples in buffer")

        first_layer_idx = self.moe_layers[0]
        first_gate_output = self.buffer_gate_outputs[first_layer_idx][0]

        if first_gate_output.dim() == 3:
            num_experts = first_gate_output.shape[-1]
        elif first_gate_output.dim() == 2:
            num_experts = first_gate_output.shape[-1]
        else:
            raise ValueError(f"Unexpected gate output shape: {first_gate_output.shape}")

        max_seq_len = max(self.buffer_seq_lengths)

        gate_logits = torch.zeros((num_samples, num_layers, max_seq_len, num_experts),
                                  dtype=torch.bfloat16, device=self.device)
        tokens = torch.zeros((num_samples, max_seq_len), dtype=torch.int32, device=self.device)

        attn_hidden_states = None
        gate_inputs = None

        if self.pattern == ActivationPattern.ATTN_GATE:
            first_attn_hidden_state = self.buffer_attn_hidden_states[first_layer_idx][0]
            hidden_dim = first_attn_hidden_state.shape[-1]
            attn_hidden_states = torch.zeros((num_samples, num_layers, max_seq_len, hidden_dim),
                                            dtype=torch.bfloat16, device=self.device)
        elif self.pattern == ActivationPattern.GATE_INPUT:
            first_gate_input = self.buffer_gate_inputs[first_layer_idx][0]
            hidden_dim = first_gate_input.shape[-1]
            gate_inputs = torch.zeros((num_samples, num_layers, max_seq_len, hidden_dim),
                                     dtype=torch.bfloat16, device=self.device)

        for sample_idx in range(num_samples):
            seq_len = self.buffer_seq_lengths[sample_idx]
            tokens[sample_idx, :seq_len] = self.buffer_tokens[sample_idx]

            batch_idx = sample_idx // self.batch_size
            pos_in_batch = sample_idx % self.batch_size

            for layer_idx_pos, layer_idx in enumerate(self.moe_layers):
                gate_output_batch = self.buffer_gate_outputs[layer_idx][batch_idx]

                if gate_output_batch.dim() == 2:
                    start_seq = 0
                    for i in range(sample_idx - pos_in_batch, sample_idx + 1):
                        if i == sample_idx:
                            break
                        start_seq += self.buffer_seq_lengths[i]

                    gate_output = gate_output_batch[start_seq:start_seq + seq_len]
                else:
                    gate_output = gate_output_batch[pos_in_batch]
                    if gate_output.dim() == 3:
                        gate_output = gate_output[0]

                if gate_output.shape[0] >= seq_len:
                    gate_logits[sample_idx, layer_idx_pos, :seq_len] = gate_output[:seq_len]
                else:
                    gate_logits[sample_idx, layer_idx_pos, :gate_output.shape[0]] = gate_output

                if self.pattern == ActivationPattern.ATTN_GATE:
                    attn_hidden_state_batch = self.buffer_attn_hidden_states[layer_idx][batch_idx]
                    if attn_hidden_state_batch.dim() == 2:
                        start_seq = 0
                        for i in range(sample_idx - pos_in_batch, sample_idx + 1):
                            if i == sample_idx:
                                break
                            start_seq += self.buffer_seq_lengths[i]
                        attn_hidden_state = attn_hidden_state_batch[start_seq:start_seq + seq_len]
                    else:
                        attn_hidden_state = attn_hidden_state_batch[pos_in_batch]
                        if attn_hidden_state.dim() == 3:
                            attn_hidden_state = attn_hidden_state[0]

                    if attn_hidden_state.dim() == 2:
                        attn_hidden_states[sample_idx, layer_idx_pos, :seq_len] = attn_hidden_state[:seq_len]
                    elif attn_hidden_state.dim() == 1:
                        attn_hidden_states[sample_idx, layer_idx_pos, :seq_len] = attn_hidden_state.unsqueeze(0)[:seq_len]

                elif self.pattern == ActivationPattern.GATE_INPUT:
                    gate_input_batch = self.buffer_gate_inputs[layer_idx][batch_idx]
                    if gate_input_batch.dim() == 2:
                        start_seq = 0
                        for i in range(sample_idx - pos_in_batch, sample_idx + 1):
                            if i == sample_idx:
                                break
                            start_seq += self.buffer_seq_lengths[i]
                        gate_input = gate_input_batch[start_seq:start_seq + seq_len]
                    else:
                        gate_input = gate_input_batch[pos_in_//batch_size]
                        if gate_input.dim() == 3:
                            gate_input = gate_input[0]

                    if gate_input.shape[0] >= seq_len:
                        gate_inputs[sample_idx, layer_idx_pos, :seq_len] = gate_input[:seq_len]
                    else:
                        gate_inputs[sample_idx, layer_idx_pos, :gate_input.shape[0]] = gate_input

        seq_lengths_tensor = torch.tensor(self.buffer_seq_lengths, dtype=torch.int32, device=self.device)

        return ActivationData(
            tokens=tokens,
            gate_logits=gate_logits,
            attn_hidden_states=attn_hidden_states,
            gate_inputs=gate_inputs,
            seq_lengths=seq_lengths_tensor,
            metadata={
                'pattern': self.pattern,
                'num_samples': num_samples,
                'num_layers': num_layers,
                'max_seq_len': max_seq_len
            }
        )

    def _sample_loop(self):
        dataset = self._load_dataset()
        if not dataset:
            print("CRITICAL ERROR: No dataset loaded. Sampler will stop.")
            self.buffer.mark_write_finished()
            return

        num_batches = (len(dataset) + self.batch_size - 1) // self.batch_size

        print(f"\nStarting online sampling...")
        print(f"Total samples: {len(dataset)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Number of batches: {num_batches}")
        print(f"Epochs: {self.epochs}")
        print(f"Start sample (first epoch only): {self.start_sample}")
        print(f"Total samples to process: {len(dataset) * self.epochs}\n")
        
        try:
            for epoch in range(self.epochs):
                if self.epochs > 1:
                    print(f"\n{'=' * 80}")
                    print(f"Epoch {epoch + 1}/{self.epochs}")
                    print(f"{'=' * 80}")
                
                start_offset = self.start_sample if epoch == 0 else 0
                if start_offset > 0:
                    print(f"  Skipping first {start_offset} samples for epoch {epoch + 1}")
                
                for batch_idx in tqdm(range(num_batches), desc=f"Sampling Epoch {epoch + 1}/{self.epochs}" if self.epochs > 1 else "Sampling"):
                    if self._stop_event.is_set():
                        print("Stop event received, stopping sampling...")
                        break
                    
                    start_idx = batch_idx * self.batch_size + start_offset
                    if start_idx >= len(dataset):
                        break
                    end_idx = min(start_idx + self.batch_size, len(dataset))
                    batch_data = dataset[start_idx:end_idx]
                    
                    self._process_batch(batch_data, batch_idx)
                    
                    activation_data = self._create_activation_data()
                    
                    success = self.buffer.write(activation_data)
                    if not success:
                        print("Failed to write to buffer, stopping sampling...")
                        break
                    
                    self._clear_buffers()
                    
                    torch.cuda.empty_cache()
        
        except Exception as e:
            print(f"\nError during sampling: {e}")
            import traceback
            traceback.print_exc()
            raise

        finally:
            for handle in self.handles:
                handle.remove()

            self.buffer.mark_write_finished()

        print(f"\nOnline sampling completed!")

    def start(self):
        if self._is_running:
            print("Sampler is already running")
            return

        self._is_running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        print("Online sampler started")

    def stop(self):
        if not self._is_running:
            print("Sampler is not running")
            return

        print("Stopping online sampler...")
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=10.0)
            if self._thread.is_alive():
                print("Warning: Sampler thread did not stop gracefully")

        self._is_running = False
        print("Online sampler stopped")

    def is_running(self) -> bool:
        return self._is_running and (self._thread is not None and self._thread.is_alive())

    def join(self, timeout: Optional[float] = None):
        if self._thread is not None:
            self._thread.join(timeout=timeout)
