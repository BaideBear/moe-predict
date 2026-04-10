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
    from trainer import GatePredictorTrainer
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location("trainer", current_dir / "trainer.py")
    trainer_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer_module)
    GatePredictorTrainer = trainer_module.GatePredictorTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train MoE gate predictor')
    parser.add_argument('--model_path', type=str, required=True, help='Path to MoE model')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset (JSONL format)')
    parser.add_argument('--pattern', type=str, default='attn_gate', help='Data pattern: attn_gate, gate_input, token_gate')
    parser.add_argument('--batch_size', type=int, default=1, help='Sampling batch size')
    parser.add_argument('--max_seq_length', type=int, default=2048, help='Maximum sequence length')
    parser.add_argument('--buffer_size_gb', type=float, default=4.0, help='Buffer size in GB')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--max_samples', type=int, default=100, help='Maximum number of samples to process')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--train_batch_size', type=int, default=15, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--use_wandb', action='store_true', help='Use wandb for logging')
    parser.add_argument('--wandb_project', type=str, default='moe-gate-predictor', help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Wandb run name')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='Directory to save checkpoints')
    parser.add_argument('--checkpoint_interval', type=int, default=1000, help='Save checkpoint every N samples')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 80)
    print("MoE Gate Predictor Training")
    print("=" * 80)
    print(f"Model path: {args.model_path}")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Pattern: {args.pattern}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max sequence length: {args.max_seq_length}")
    print(f"Buffer size: {args.buffer_size_gb} GB")
    print(f"Device: {args.device}")
    print(f"Max samples: {args.max_samples}")
    print("=" * 80)
    
    # Step 1: Load model and tokenizer
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
    
    # Step 2: Extract model configuration
    print("\n[Step 2] Extracting model configuration...")
    model_config = extract_model_config(
        model=model,
        model_name=os.path.basename(args.model_path),
        max_seq_length=args.max_seq_length
    )
    
    print(f"  Model name: {model_config.model_name}")
    print(f"  Number of layers: {model_config.num_layers}")
    print(f"  Hidden dimension: {model_config.hidden_dim}")
    print(f"  Number of experts: {model_config.num_experts}")
    print(f"  Max sequence length: {model_config.max_seq_length}")
    
    # Step 3: Detect MoE layers
    print("\n[Step 3] Detecting MoE layers...")
    moe_layer_indices = detect_moe_layers(model)
    print(f"  Found {len(moe_layer_indices)} MoE layers: {moe_layer_indices}")
    
    # Step 4: Get detailed MoE layer information
    print("\n[Step 4] Getting MoE layer information...")
    moe_layer_info = {}
    for layer_idx in moe_layer_indices:
        info = get_moe_layer_info(model, layer_idx)
        moe_layer_info[layer_idx] = info
        print(f"  Layer {layer_idx}: {info['type']}")
    
    # Step 5: Create buffer
    print("\n[Step 5] Creating activation buffer...")
    buffer = create_buffer(
        model_config=model_config,
        pattern=args.pattern,
        buffer_size_gb=args.buffer_size_gb,
        device=args.device
    )
    print("✓ Buffer created successfully")
    
    # Step 6: Create online sampler
    print("\n[Step 6] Creating online sampler...")
    sampler = OnlineSampler(
        model=model,
        tokenizer=tokenizer,
        dataset_path=args.dataset_path,
        buffer=buffer,
        pattern=args.pattern,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        trust_remote_code=True
    )
    print("✓ Sampler created successfully")
    
    # Step 7: Create predictor interface
    print("\n[Step 7] Creating predictor interface...")
    predictor_interface = create_predictor_interface(
        buffer=buffer,
        pattern=args.pattern,
        batch_size=1
    )
    print("✓ Predictor interface created successfully")
    
    # Step 8: Create predictor model (will be initialized after first sample)
    print("\n[Step 8] Predictor model will be created after first sample...")
    print(f"  Available model types: {list_available_models()}")
    predictor_model = None
    trainer = None
    
    # Step 9: Start sampling
    print("\n[Step 9] Starting online sampling...")
    sampler.start()
    print("✓ Sampling started")
    
    # Step 10: Training loop with epochs
    print("\n[Step 10] Training loop...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Train batch size: {args.train_batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Use wandb: {args.use_wandb}")
    print("(Press Ctrl+C to stop)")
    
    samples_processed = 0
    total_samples_processed = 0
    
    try:
        for epoch in range(args.epochs):
            print(f"\n{'=' * 80}")
            print(f"Epoch {epoch + 1}/{args.epochs}")
            print(f"{'=' * 80}")
            
            epoch_samples = 0
            epoch_start_time = time.time()
            
            while epoch_samples < args.max_samples:
                batch = predictor_interface.get_batch()
                
                if batch is None:
                    if not sampler.is_running():
                        print("Sampler has stopped, no more data available")
                        break
                    continue
                
                for data in batch:
                    epoch_samples += 1
                    samples_processed += 1
                    total_samples_processed += 1
                    
                    # Extract model structure information from sampled data
                    if samples_processed == 1:
                        print(f"\n[First Sample Analysis]")
                        print(f"  Tokens shape: {data.tokens.shape}")
                        print(f"  Gate logits shape: {data.gate_logits.shape}")
                        
                        if data.attn_hidden_states is not None:
                            print(f"  Attention hidden states shape: {data.attn_hidden_states.shape}")
                        
                        if data.gate_inputs is not None:
                            print(f"  Gate inputs shape: {data.gate_inputs.shape}")
                        
                        if data.seq_lengths is not None:
                            print(f"  Sequence lengths: {data.seq_lengths}")
                        
                        # Parse model structure from sampled data
                        num_samples, num_layers, seq_len, num_experts = data.gate_logits.shape
                        print(f"\n  Model structure from sampled data:")
                        print(f"    Number of MoE layers: {num_layers}")
                        print(f"    Number of experts to predict: {num_experts}")
                        print(f"    Sequence length: {seq_len}")
                        
                        if data.attn_hidden_states is not None:
                            hidden_dim = data.attn_hidden_states.shape[-1]
                            print(f"    Hidden dimension: {hidden_dim}")
                        
                        # Create predictor model on first sample
                        print(f"\n  Creating predictor model...")
                        print(f"    Model type: simple_mlp")
                        print(f"    Number of layers: {num_layers}")
                        print(f"    Input dimension: {hidden_dim}")
                        print(f"    Number of experts: {num_experts}")
                        
                        predictor_model = get_predictor_model(
                            model_type="simple_mlp",
                            num_layers=num_layers,
                            input_dim=hidden_dim,
                            num_experts=num_experts,
                            hidden_dim=2048,
                            dropout=0.1
                        )
                        
                        # Create trainer
                        trainer = GatePredictorTrainer(
                            model=predictor_model,
                            learning_rate=args.learning_rate,
                            weight_decay=args.weight_decay,
                            train_batch_size=args.train_batch_size,
                            device=args.device,
                            use_wandb=args.use_wandb,
                            wandb_project=args.wandb_project,
                            wandb_run_name=args.wandb_run_name
                        )
                        
                        print(f"  ✓ Predictor model and trainer created")
                        
                        # Print model architecture
                        print(f"\n  Model architecture:")
                        print(f"    Total parameters: {sum(p.numel() for p in predictor_model.parameters()):,}")
                        print(f"    Trainable parameters: {sum(p.numel() for p in predictor_model.parameters() if p.requires_grad):,}")
                    
                    # Add sample to trainer
                    if data.attn_hidden_states is not None and data.gate_logits is not None:
                        trainer.add_sample(
                            attn_hidden_states=data.attn_hidden_states,
                            gate_logits=data.gate_logits,
                            seq_lengths=data.seq_lengths
                        )
                    
                    # Save checkpoint
                    if args.checkpoint_dir is not None and total_samples_processed % args.checkpoint_interval == 0:
                        checkpoint_path = os.path.join(args.checkpoint_dir, f"predictor_sample_{total_samples_processed}.pt")
                        trainer.save_checkpoint(checkpoint_path, epoch)
                
                if epoch_samples >= args.max_samples:
                    break
            
            # Flush remaining samples in buffer
            if trainer is not None:
                trainer.flush_remaining()
            
            epoch_time = time.time() - epoch_start_time
            print(f"\nEpoch {epoch + 1} completed in {epoch_time:.2f}s")
            print(f"  Samples processed: {epoch_samples}")
            
            # Restart sampler for next epoch
            if epoch < args.epochs - 1:
                print("\nRestarting sampler for next epoch...")
                sampler.stop()
                sampler.join()
                
                # Clear buffer
                buffer.clear()
                
                # Restart sampler
                sampler.start()
                print("✓ Sampler restarted")
    
    except KeyboardInterrupt:
        print("\n\nReceived stop signal...")
    
    # Step 11: Stop sampling and cleanup
    print("\n[Step 11] Stopping sampling and cleanup...")
    sampler.stop()
    buffer.stop()
    
    # Finish trainer
    if trainer is not None:
        trainer.finish()
    
    del model
    del tokenizer
    torch.cuda.empty_cache()
    
    print("\n" + "=" * 80)
    print(f"Training completed! Processed {total_samples_processed} samples")
    print("=" * 80)


if __name__ == "__main__":
    main()
