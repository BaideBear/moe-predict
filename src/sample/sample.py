import torch
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class MoESampler:
    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        model_path: str,
        input_path: str,
        output_dir: str,
        batch_size: int = 1,
        max_seq_length: int = 2048,
        chunk_size: int = 100,
        trust_remote_code: bool = True
    ):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.model_path = model_path
        self.input_path = input_path
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.chunk_size = chunk_size
        self.trust_remote_code = trust_remote_code
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._setup_output_dir()
        self._load_model_and_tokenizer()
        self._detect_moe_layers()
        self._init_buffers()
        self._register_hooks()
    
    def _setup_output_dir(self):
        output_path = Path(self.output_dir) / self.model_name
        output_path.mkdir(parents=True, exist_ok=True)
        self.output_path = output_path
    
    def _load_model_and_tokenizer(self):
        print(f"Loading model from {self.model_path}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=self.trust_remote_code
            )
        except Exception as e:
            print(f"Fast tokenizer loading failed: {e}")
            print("Falling back to slow tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=self.trust_remote_code,
                use_fast=False
            )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        attn_impl = "flash_attention_2"
        try:
            import flash_attn
            print("Flash Attention 2 is available, using it for better performance")
        except ImportError:
            print("Flash Attention 2 is not installed, falling back to default attention implementation")
            attn_impl = "eager"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl
        )
        self.model.eval()
        print(f"Model loaded successfully. Using device_map: {self.model.hf_device_map}")
    
    def _detect_moe_layers(self):
        self.moe_layers = []
        self.moe_layer_info = {}
        
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            for i, layer in enumerate(self.model.model.layers):
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
                        print(f"Layer {i}: Found direct block_sparse_moe.gate (Mixtral style)")
                    else:
                        print(f"Layer {i}: Has block_sparse_moe but no gate")
                
                if moe_info is None and hasattr(layer, 'mlp'):
                    mlp = layer.mlp
                    
                    if hasattr(mlp, 'gate'):
                        moe_info = {
                            'type': 'mlp_gate',
                            'gate_path': f'layer.mlp.gate',
                            'gate_module': mlp.gate,
                            'input_layernorm_path': 'layer.input_layernorm'
                        }
                        print(f"Layer {i}: Found mlp.gate (DeepSeek style)")
                    elif hasattr(mlp, 'block_sparse_moe'):
                        block_sparse_moe = mlp.block_sparse_moe
                        if hasattr(block_sparse_moe, 'gate'):
                            moe_info = {
                                'type': 'mlp_block_sparse_moe',
                                'gate_path': f'layer.mlp.block_sparse_moe.gate',
                                'gate_module': block_sparse_moe.gate,
                                'input_layernorm_path': 'layer.input_layernorm'
                            }
                            print(f"Layer {i}: Found mlp.block_sparse_moe.gate")
                        else:
                            print(f"Layer {i}: Has mlp.block_sparse_moe but no gate")
                
                if moe_info is None:
                    print(f"Layer {i}: No MoE structure found")
                
                if moe_info is not None:
                    self.moe_layers.append(i)
                    self.moe_layer_info[i] = moe_info
        
        print(f"Detected {len(self.moe_layers)} MoE layers: {self.moe_layers}")
        
        if len(self.moe_layers) == 0:
            print("\n=== Detailed Model Structure Debug ===")
            print(f"Model type: {type(self.model)}")
            print(f"Model class: {self.model.__class__.__name__}")
            
            if hasattr(self.model, 'model'):
                if hasattr(self.model.model, 'layers'):
                    print(f"Number of layers: {len(self.model.model.layers)}")
                    
                    print("\n=== First layer structure ===")
                    first_layer = self.model.model.layers[0]
                    print(f"Layer 0 type: {type(first_layer)}")
                    print(f"Layer 0 attributes: {[attr for attr in dir(first_layer) if not attr.startswith('_')]}")
                    
                    if hasattr(first_layer, 'mlp'):
                        print(f"\nMLP type: {type(first_layer.mlp)}")
                        print(f"MLP attributes: {[attr for attr in dir(first_layer.mlp) if not attr.startswith('_')]}")
                    
                    if hasattr(first_layer, 'block_sparse_moe'):
                        print(f"\nblock_sparse_moe type: {type(first_layer.block_sparse_moe)}")
                        print(f"block_sparse_moe attributes: {[attr for attr in dir(first_layer.block_sparse_moe) if not attr.startswith('_')]}")
            
            raise ValueError("No MoE layers detected in the model!")
    
    def _init_buffers(self):
        self.buffer_gate_outputs = {layer_idx: [] for layer_idx in self.moe_layers}
        self.buffer_gate_inputs = {layer_idx: [] for layer_idx in self.moe_layers}
        self.buffer_attn_hidden_states = {layer_idx: [] for layer_idx in self.moe_layers}
        self.buffer_tokens = []
        self.buffer_sample_indices = []
    
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
            
            print(f"Registered hooks for layer {layer_idx} ({moe_info['type']})")
        
        print(f"Total registered {len(self.handles)} hooks")
    
    def _create_attn_hidden_state_hook(self, layer_idx: int):
        def hook(module, input, output):
            hidden_state = output[0].detach().cpu()
            self.buffer_attn_hidden_states[layer_idx].append(hidden_state)
        return hook
    
    def _create_gate_hook(self, layer_idx: int):
        def hook(module, input, output):
            if isinstance(output, tuple):
                router_output = output[0]
            else:
                router_output = output
            
            gate_output = router_output.detach().cpu()
            self.buffer_gate_outputs[layer_idx].append(gate_output)
            
            if len(input) > 0:
                gate_input = input[0].detach().cpu()
                self.buffer_gate_inputs[layer_idx].append(gate_input)
        return hook
    
    def _clear_buffers(self):
        for layer_idx in self.moe_layers:
            self.buffer_gate_outputs[layer_idx].clear()
            self.buffer_gate_inputs[layer_idx].clear()
            self.buffer_attn_hidden_states[layer_idx].clear()
        self.buffer_tokens.clear()
        self.buffer_sample_indices.clear()
    
    def _load_dataset(self) -> List[Dict]:
        print(f"Loading dataset from {self.input_path}...")
        data = []
        
        with open(self.input_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        
        print(f"Loaded {len(data)} samples")
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
        sample_indices = [batch_idx * self.batch_size + i for i in range(len(batch_data))]
        
        tokenized = self._tokenize_batch(texts)
        input_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device)
        seq_lengths = tokenized['seq_lengths']
        
        with torch.no_grad():
            _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        for i, seq_len in enumerate(seq_lengths):
            self.buffer_tokens.append(input_ids[i, :seq_len].cpu())
        
        self.buffer_sample_indices.extend(sample_indices)
    
    def _save_chunk(self, chunk_idx: int):
        print(f"Saving chunk {chunk_idx}...")
        
        num_samples = len(self.buffer_tokens)
        num_layers = len(self.moe_layers)
        
        if num_samples == 0:
            print("Warning: No samples in this chunk, skipping save")
            return
        
        first_layer_idx = self.moe_layers[0]
        first_gate_output = self.buffer_gate_outputs[first_layer_idx][0]
        
        if first_gate_output.dim() == 3:
            num_experts = first_gate_output.shape[-1]
        elif first_gate_output.dim() == 2:
            num_experts = first_gate_output.shape[-1]
        else:
            raise ValueError(f"Unexpected gate output shape: {first_gate_output.shape}")
        
        first_attn_hidden_state = self.buffer_attn_hidden_states[first_layer_idx][0]
        hidden_dim = first_attn_hidden_state.shape[-1]
        
        max_seq_len = max(len(tokens) for tokens in self.buffer_tokens)
        
        gate_outputs = torch.zeros((num_samples, num_layers, num_experts), dtype=torch.bfloat16)
        gate_inputs = torch.zeros((num_samples, num_layers, hidden_dim), dtype=torch.bfloat16)
        attn_hidden_states = torch.zeros((num_samples, num_layers, max_seq_len, hidden_dim), dtype=torch.bfloat16)
        tokens = torch.zeros((num_samples, max_seq_len), dtype=torch.int32)
        
        for sample_idx in range(num_samples):
            seq_len = len(self.buffer_tokens[sample_idx])
            
            if seq_len > tokens.shape[1]:
                new_tokens = torch.zeros((num_samples, seq_len), dtype=torch.int32)
                new_tokens[:, :tokens.shape[1]] = tokens
                tokens = new_tokens
            
            if seq_len > attn_hidden_states.shape[2]:
                new_attn_hidden_states = torch.zeros((num_samples, num_layers, seq_len, hidden_dim), dtype=torch.bfloat16)
                new_attn_hidden_states[:, :, :attn_hidden_states.shape[2]] = attn_hidden_states
                attn_hidden_states = new_attn_hidden_states
            
            tokens[sample_idx, :seq_len] = self.buffer_tokens[sample_idx]
            
            batch_idx = sample_idx // self.batch_size
            pos_in_batch = sample_idx % self.batch_size
            
            for layer_idx_pos, layer_idx in enumerate(self.moe_layers):
                gate_output_batch = self.buffer_gate_outputs[layer_idx][batch_idx]
                gate_input_batch = self.buffer_gate_inputs[layer_idx][batch_idx]
                attn_hidden_state_batch = self.buffer_attn_hidden_states[layer_idx][batch_idx]
                
                gate_output = gate_output_batch[pos_in_batch]
                gate_input = gate_input_batch[pos_in_batch]
                attn_hidden_state = attn_hidden_state_batch[pos_in_batch]
                
                if gate_output.dim() == 3:
                    gate_output = gate_output[0]
                if gate_input.dim() == 3:
                    gate_input = gate_input[0]
                if attn_hidden_state.dim() == 3:
                    attn_hidden_state = attn_hidden_state[0]
                
                if gate_output.shape[0] >= seq_len:
                    gate_outputs[sample_idx, layer_idx_pos] = gate_output[seq_len - 1]
                else:
                    gate_outputs[sample_idx, layer_idx_pos] = gate_output[-1]
                
                if gate_input.shape[0] >= seq_len:
                    gate_inputs[sample_idx, layer_idx_pos] = gate_input[seq_len - 1]
                else:
                    gate_inputs[sample_idx, layer_idx_pos] = gate_input[-1]
                
                if attn_hidden_state.dim() == 2:
                    attn_hidden_states[sample_idx, layer_idx_pos, :seq_len] = attn_hidden_state[:seq_len]
                elif attn_hidden_state.dim() == 1:
                    attn_hidden_states[sample_idx, layer_idx_pos, :seq_len] = attn_hidden_state.unsqueeze(0)[:seq_len]
                else:
                    attn_hidden_states[sample_idx, layer_idx_pos, :seq_len] = attn_hidden_state[:seq_len]
        
        sample_indices = torch.tensor(self.buffer_sample_indices, dtype=torch.int32)
        
        output_file = self.output_path / f"{self.dataset_name}_sample.pt"
        
        if chunk_idx == 0:
            data = {
                "gate_outputs": gate_outputs,
                "gate_inputs": gate_inputs,
                "attn_hidden_states": attn_hidden_states,
                "tokens": tokens,
                "sample_indices": sample_indices
            }
            torch.save(data, output_file)
        else:
            existing_data = torch.load(output_file)
            
            existing_max_seq_len = existing_data["attn_hidden_states"].shape[2]
            current_max_seq_len = attn_hidden_states.shape[2]
            
            if current_max_seq_len > existing_max_seq_len:
                new_existing_attn = torch.zeros(
                    (existing_data["attn_hidden_states"].shape[0], 
                     existing_data["attn_hidden_states"].shape[1], 
                     current_max_seq_len, 
                     existing_data["attn_hidden_states"].shape[3]), 
                    dtype=torch.bfloat16
                )
                new_existing_attn[:, :, :existing_max_seq_len] = existing_data["attn_hidden_states"]
                existing_data["attn_hidden_states"] = new_existing_attn
                
                new_existing_tokens = torch.zeros(
                    (existing_data["tokens"].shape[0], current_max_seq_len), 
                    dtype=torch.int32
                )
                new_existing_tokens[:, :existing_max_seq_len] = existing_data["tokens"]
                existing_data["tokens"] = new_existing_tokens
            
            if existing_max_seq_len > current_max_seq_len:
                new_attn_hidden_states = torch.zeros(
                    (attn_hidden_states.shape[0], 
                     attn_hidden_states.shape[1], 
                     existing_max_seq_len, 
                     attn_hidden_states.shape[3]), 
                    dtype=torch.bfloat16
                )
                new_attn_hidden_states[:, :, :current_max_seq_len] = attn_hidden_states
                attn_hidden_states = new_attn_hidden_states
                
                new_tokens = torch.zeros(
                    (tokens.shape[0], existing_max_seq_len), 
                    dtype=torch.int32
                )
                new_tokens[:, :current_max_seq_len] = tokens
                tokens = new_tokens
            
            gate_outputs_all = torch.cat([existing_data["gate_outputs"], gate_outputs], dim=0)
            gate_inputs_all = torch.cat([existing_data["gate_inputs"], gate_inputs], dim=0)
            attn_hidden_states_all = torch.cat([existing_data["attn_hidden_states"], attn_hidden_states], dim=0)
            tokens_all = torch.cat([existing_data["tokens"], tokens], dim=0)
            sample_indices_all = torch.cat([existing_data["sample_indices"], sample_indices], dim=0)
            
            data = {
                "gate_outputs": gate_outputs_all,
                "gate_inputs": gate_inputs_all,
                "attn_hidden_states": attn_hidden_states_all,
                "tokens": tokens_all,
                "sample_indices": sample_indices_all
            }
            torch.save(data, output_file)
        
        print(f"Saved chunk {chunk_idx} to {output_file}")
        print(f"  Total samples: {gate_outputs.shape[0]}")
        print(f"  Shape: gate_outputs={gate_outputs.shape}, gate_inputs={gate_inputs.shape}")
        print(f"  Shape: attn_hidden_states={attn_hidden_states.shape}, tokens={tokens.shape}")
    
    def sample(self):
        dataset = self._load_dataset()
        num_batches = (len(dataset) + self.batch_size - 1) // self.batch_size
        chunk_idx = 0
        samples_in_chunk = 0
        
        print(f"\nStarting sampling...")
        print(f"Total samples: {len(dataset)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Chunk size: {self.chunk_size}")
        print(f"Number of batches: {num_batches}\n")
        
        try:
            for batch_idx in tqdm(range(num_batches), desc="Sampling"):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(dataset))
                batch_data = dataset[start_idx:end_idx]
                
                self._process_batch(batch_data, batch_idx)
                samples_in_chunk += len(batch_data)
                
                if samples_in_chunk >= self.chunk_size or batch_idx == num_batches - 1:
                    self._save_chunk(chunk_idx)
                    self._clear_buffers()
                    chunk_idx += 1
                    samples_in_chunk = 0
                    
                    torch.cuda.empty_cache()
        
        except Exception as e:
            print(f"\nError during sampling: {e}")
            raise
        
        finally:
            for handle in self.handles:
                handle.remove()
        
        print(f"\nSampling completed!")
        print(f"Output saved to: {self.output_path / f'{self.dataset_name}_sample.npz'}")


def main():
    parser = argparse.ArgumentParser(description='Sample hidden states from MoE models')
    
    parser.add_argument(
        '--model-name',
        type=str,
        required=True,
        help='Name of the model (e.g., Mixtral-8x7B-v0.1)'
    )
    parser.add_argument(
        '--dataset-name',
        type=str,
        required=True,
        help='Name of the dataset (e.g., mmlu)'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to the model directory'
    )
    parser.add_argument(
        '--input-path',
        type=str,
        default=None,
        help='Path to input dataset (default: {project_dir}/dataset/processed/train/{dataset_name}.jsonl)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/data1/gx/MoE-predict/samples',
        help='Path to output directory (default: /data1/gx/MoE-predict/samples)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for inference (default: 1)'
    )
    parser.add_argument(
        '--max-seq-length',
        type=int,
        default=2048,
        help='Maximum sequence length (default: 2048)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=100,
        help='Number of samples per chunk (default: 100)'
    )
    parser.add_argument(
        '--trust-remote-code',
        action='store_true',
        default=True,
        help='Trust remote code when loading model'
    )
    
    args = parser.parse_args()
    
    if args.input_path is None:
        project_dir = Path(__file__).parent.parent.parent
        args.input_path = str(project_dir / "dataset" / "processed" / "train" / f"{args.dataset_name}.jsonl")
    
    sampler = MoESampler(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        model_path=args.model_path,
        input_path=args.input_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        chunk_size=args.chunk_size,
        trust_remote_code=args.trust_remote_code
    )
    
    sampler.sample()


if __name__ == "__main__":
    main()
