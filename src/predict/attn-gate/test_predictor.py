import torch
import sys
import os
import argparse
import importlib.util
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from online_sample import (
    create_buffer,
    extract_model_config,
    OnlineSampler,
    create_predictor_interface,
    detect_moe_layers,
    get_moe_layer_info
)

current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from model_factory import get_predictor_model, list_available_models
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location("model_factory", current_dir / "model_factory.py")
    model_factory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_factory)
    get_predictor_model = model_factory.get_predictor_model
    list_available_models = model_factory.list_available_models

try:
    from evaluater import GatePredictorEvaluater
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location("evaluater", current_dir / "evaluater.py")
    evaluater_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluater_module)
    GatePredictorEvaluater = evaluater_module.GatePredictorEvaluater


def parse_args():
    parser = argparse.ArgumentParser(description='Test MoE gate predictor')
    parser.add_argument('--model_path', type=str, required=True, help='Path to MoE model')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset (JSONL format)')
    parser.add_argument('--pattern', type=str, default='attn_gate', help='Data pattern: attn_gate, gate_input, token_gate')
    parser.add_argument('--batch_size', type=int, default=1, help='Sampling batch size')
    parser.add_argument('--max_seq_length', type=int, default=2048, help='Maximum sequence length')
    parser.add_argument('--buffer_size_gb', type=float, default=4.0, help='Buffer size in GB')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--max_samples', type=int, default=100, help='Maximum number of samples to process')
    parser.add_argument('--epochs', type=int, default=1, help='Number of evaluation epochs')
    parser.add_argument('--eval_batch_size', type=int, default=15, help='Evaluation batch size')
    parser.add_argument('--top_k_values', type=str, default='1,2,4', help='Comma-separated top-k values for accuracy evaluation')
    parser.add_argument('--use_wandb', action='store_true', help='Use wandb for logging')
    parser.add_argument('--wandb_project', type=str, default='moe-gate-predictor-eval', help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Wandb run name')
    parser.add_argument('--load_checkpoint', type=str, required=True, help='Path to checkpoint file to load')
    parser.add_argument('--model_type', type=str, default='simple_mlp', help='Predictor model type (use --list_models to see available models)')
    parser.add_argument('--list_models', action='store_true', help='List available model types and exit')
    return parser.parse_args()


def main():
    args = parse_args()
    
    available_models = list_available_models()
    
    if args.list_models:
        print("Available predictor model types:")
        for model_type in available_models:
            print(f"  - {model_type}")
        return
    
    if args.model_type not in available_models:
        print(f"Error: Unknown model type '{args.model_type}'")
        print(f"Available model types: {available_models}")
        print("Use --list_models to see all available models")
        sys.exit(1)
    
    top_k_values = [int(k.strip()) for k in args.top_k_values.split(',')]
    
    print("=" * 80)
    print("MoE Gate Predictor Evaluation")
    print("=" * 80)
    print(f"Model path: {args.model_path}")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Pattern: {args.pattern}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max sequence length: {args.max_seq_length}")
    print(f"Buffer size: {args.buffer_size_gb} GB")
    print(f"Device: {args.device}")
    print(f"Max samples per epoch: {args.max_samples}")
    print(f"Epochs: {args.epochs}")
    print(f"Eval batch size: {args.eval_batch_size}")
    print(f"Top-k values: {top_k_values}")
    print(f"Use wandb: {args.use_wandb}")
    print(f"Checkpoint to load: {args.load_checkpoint}")
    print("=" * 80)
    
    print("\n[Step 1] Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✓ Model and tokenizer loaded successfully")
    
    print("\n[Step 2] Extracting model configuration...")
    model_config = extract_model_config(
        model=model,
        model_name=os.path.basename(args.model_path),
        max_seq_length=args.max_seq_length
    )
    
    print(f"  Model name: {model_config.model_name}")
    print(f"  Number of layers: {model_config.num_layers}")
    print(f"  Hidden dimension: {model_config.hidden_dim}")
    print(f"  Number of experts (from config): {model_config.num_experts}")
    print(f"  Max sequence length: {model_config.max_seq_length}")
    
    if hasattr(model.config, 'n_routed_experts'):
        print(f"  [DEBUG] n_routed_experts from config: {model.config.n_routed_experts}")
    if hasattr(model.config, 'num_local_experts'):
        print(f"  [DEBUG] num_local_experts from config: {model.config.num_local_experts}")
    if hasattr(model.config, 'num_experts'):
        print(f"  [DEBUG] num_experts from config: {model.config.num_experts}")
    
    print("\n[Step 3] Detecting MoE layers...")
    moe_layer_indices = detect_moe_layers(model)
    print(f"  Found {len(moe_layer_indices)} MoE layers: {moe_layer_indices}")
    
    print("\n[Step 4] Getting MoE layer information...")
    moe_layer_info = {}
    for layer_idx in moe_layer_indices:
        info = get_moe_layer_info(model, layer_idx)
        moe_layer_info[layer_idx] = info
        print(f"  Layer {layer_idx}: {info['type']}")
    
    print("\n[Step 5] Creating activation buffer...")
    buffer = create_buffer(
        model_config=model_config,
        pattern=args.pattern,
        buffer_size_gb=args.buffer_size_gb,
        device=args.device
    )
    print("✓ Buffer created successfully")
    
    print("\n[Step 6] Creating online sampler...")
    sampler = OnlineSampler(
        model=model,
        tokenizer=tokenizer,
        dataset_path=args.dataset_path,
        buffer=buffer,
        pattern=args.pattern,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        trust_remote_code=True,
        epochs=args.epochs
    )
    print("✓ Sampler created successfully")
    
    print("\n[Step 7] Creating predictor interface...")
    predictor_interface = create_predictor_interface(
        buffer=buffer,
        pattern=args.pattern,
        batch_size=1
    )
    print("✓ Predictor interface created successfully")
    
    print("\n[Step 8] Predictor model will be created after first sample...")
    print(f"  Available model types: {list_available_models()}")
    predictor_model = None
    evaluater = None
    
    print("\n[Step 9] Starting online sampling...")
    sampler.start()
    print("✓ Sampling started")
    
    print("\n[Step 10] Evaluation loop...")
    print(f"  Eval batch size: {args.eval_batch_size}")
    print(f"  Top-k values: {top_k_values}")
    print(f"  Use wandb: {args.use_wandb}")
    print("(Press Ctrl+C to stop)...")
    
    samples_processed = 0
    total_samples_processed = 0
    
    try:
        while True:
            batch = predictor_interface.get_batch()
            
            if batch is None:
                if not sampler.is_running():
                    print("Sampler has stopped, no more data available")
                    break
                continue
            
            for data in batch:
                samples_processed += 1
                total_samples_processed += 1
                
                if samples_processed == 1:
                    print(f"\n[First Sample Analysis]")
                    print(f"  Tokens shape: {data.tokens.shape}")
                    print(f"  Gate logits shape: {data.gate_logits.shape}")
                    
                    num_samples_batch, num_layers, seq_len, num_experts_from_gate = data.gate_logits.shape
                    print(f"\n  [DEBUG] Gate shape analysis:")
                    print(f"    num_samples: {num_samples_batch}")
                    print(f"    num_layers: {num_layers}")
                    print(f"    seq_len: {seq_len}")
                    print(f"    num_experts (from gate): {num_experts_from_gate}")
                    print(f"    num_experts (from config): {model_config.num_experts}")
                    if num_experts_from_gate != model_config.num_experts:
                        print(f"    ⚠️  MISMATCH! Gate has {num_experts_from_gate} experts, but config says {model_config.num_experts}")
                    
                    if data.attn_hidden_states is not None:
                        print(f"\n  Attention hidden states shape: {data.attn_hidden_states.shape}")
                    
                    if data.gate_inputs is not None:
                        print(f"  Gate inputs shape: {data.gate_inputs.shape}")
                    
                    if data.seq_lengths is not None:
                        print(f"  Sequence lengths: {data.seq_lengths}")
                    
                    print(f"\n  Model structure from sampled data:")
                    print(f"    Number of MoE layers: {num_layers}")
                    print(f"    Number of experts to predict: {num_experts_from_gate}")
                    print(f"    Sequence length: {seq_len}")
                    
                    if data.attn_hidden_states is not None:
                        hidden_dim = data.attn_hidden_states.shape[-1]
                        print(f"    Hidden dimension: {hidden_dim}")
                    
                    print(f"\n  Creating predictor model...")
                    print(f"    Model type: {args.model_type}")
                    print(f"    Number of layers: {num_layers}")
                    print(f"    Input dimension: {hidden_dim}")
                    print(f"    Number of experts: {num_experts_from_gate}")
                    
                    predictor_model = get_predictor_model(
                        model_type=args.model_type,
                        num_layers=num_layers,
                        input_dim=hidden_dim,
                        num_experts=num_experts_from_gate,
                        hidden_dim=2048,
                        dropout=0.1
                    )
                    
                    print(f"\n  Creating evaluater...")
                    evaluater = GatePredictorEvaluater(
                        model=predictor_model,
                        eval_batch_size=args.eval_batch_size,
                        device=args.device,
                        top_k_values=top_k_values,
                        use_wandb=args.use_wandb,
                        wandb_project=args.wandb_project,
                        wandb_run_name=args.wandb_run_name
                    )
                    
                    print(f"  Loading checkpoint from: {args.load_checkpoint}")
                    evaluater.load_checkpoint(args.load_checkpoint)
                    
                    print(f"  ✓ Predictor model and evaluater created")
                    
                    print(f"\n  Model architecture:")
                    print(f"    Total parameters: {sum(p.numel() for p in predictor_model.parameters()):,}")
                    print(f"    Trainable parameters: {sum(p.numel() for p in predictor_model.parameters() if p.requires_grad):,}")
                
                if data.attn_hidden_states is not None and data.gate_logits is not None:
                    evaluater.add_sample(
                        attn_hidden_states=data.attn_hidden_states,
                        gate_logits=data.gate_logits,
                        seq_lengths=data.seq_lengths
                    )
    
    except KeyboardInterrupt:
        print("\n\nReceived stop signal...")
    
    print("\n[Step 11] Stopping sampling and cleanup...")
    sampler.stop()
    buffer.stop()
    
    if evaluater is not None:
        evaluater.finish()
    
    del model
    del tokenizer
    torch.cuda.empty_cache()
    
    print("\n" + "=" * 80)
    print(f"Evaluation completed! Processed {total_samples_processed} samples")
    print("=" * 80)


if __name__ == "__main__":
    main()
