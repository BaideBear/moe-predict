import os
import sys
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from online_sample import (
    create_buffer,
    extract_model_config,
    OnlineSampler
)
from predict.PEPP.model import create_predictor_model
from predict.PEPP.trainer import PredictorTrainer


def main():
    parser = argparse.ArgumentParser(description='Train PEPP Predictor')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the base model')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the training dataset')
    parser.add_argument('--pattern', type=str, default='attn_gate', choices=['attn_gate', 'gate_input', 'token_gate'], help='Data pattern')
    parser.add_argument('--buffer_size_gb', type=float, default=4.0, help='Buffer size in GB')
    parser.add_argument('--batch_size', type=int, default=1, help='Sampler batch size')
    parser.add_argument('--max_seq_length', type=int, default=2048, help='Maximum sequence length')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--top_k_routing', type=int, default=8, help='Top-k routing for loss calculation')
    parser.add_argument('--lambda_rank', type=float, default=1.0, help='Lambda for ranking loss')
    parser.add_argument('--margin', type=float, default=0.1, help='Margin for ranking loss')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--checkpoint_interval', type=int, default=50, help='Checkpoint save interval (number of samples)')
    parser.add_argument('--max_samples_per_epoch', type=int, default=None, help='Maximum samples per epoch (None = no limit, train all data)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PEPP Predictor Training")
    print("=" * 60)
    
    print("\n1. Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    print(f"   Model loaded from: {args.model_path}")
    print(f"   Model dtype: {torch.bfloat16}")
    
    print("\n2. Extracting model configuration...")
    model_config = extract_model_config(model, args.model_path, args.max_seq_length)
    print(f"   Model name: {model_config.model_name}")
    print(f"   Number of layers: {model_config.num_layers}")
    print(f"   Hidden dimension: {model_config.hidden_dim}")
    print(f"   Number of experts: {model_config.num_experts}")
    print(f"   Max sequence length: {model_config.max_seq_length}")
    
    print("\n3. Creating buffer...")
    buffer = create_buffer(
        model_config=model_config,
        pattern=args.pattern,
        buffer_size_gb=args.buffer_size_gb,
        device=args.device
    )
    print(f"   Buffer created with pattern: {args.pattern}")
    print(f"   Buffer size: {args.buffer_size_gb} GB")
    
    print("\n4. Creating predictor model...")
    predict_model = create_predictor_model(
        num_layers=model_config.num_layers,
        input_dim=model_config.hidden_dim,
        num_experts=model_config.num_experts,
        hidden_dim=2048,
        dropout=0.1
    )
    print(f"   Predictor model created:")
    print(f"     Number of layers: {model_config.num_layers}")
    print(f"     Input dimension: {model_config.hidden_dim}")
    print(f"     Number of experts: {model_config.num_experts}")
    print(f"     Hidden dimension: 2048")
    
    print("\n5. Creating online sampler...")
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
    print(f"   Sampler created:")
    print(f"     Dataset path: {args.dataset_path}")
    print(f"     Batch size: {args.batch_size}")
    print(f"     Max sequence length: {args.max_seq_length}")
    
    print("\n6. Starting online sampler...")
    sampler.start()
    print("   Online sampler started (running in background)")
    
    print("\n7. Creating predictor trainer...")
    trainer = PredictorTrainer(
        buffer=buffer,
        predict_model=predict_model,
        pattern=args.pattern,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        top_k_routing=args.top_k_routing,
        lambda_rank=args.lambda_rank,
        margin=args.margin,
        checkpoint_interval=args.checkpoint_interval,
        device=args.device
    )
    print(f"   Trainer created:")
    print(f"     Checkpoint interval: {args.checkpoint_interval} samples")
    print(f"     Learning rate: {args.learning_rate}")
    print(f"     Weight decay: {args.weight_decay}")
    print(f"     Top-k routing: {args.top_k_routing}")
    print(f"     Lambda rank: {args.lambda_rank}")
    print(f"     Margin: {args.margin}")
    
    print("\n8. Starting training...")
    print("=" * 60)
    
    try:
        trainer.train(
            num_epochs=args.num_epochs,
            save_dir=args.save_dir,
            max_samples_per_epoch=args.max_samples_per_epoch
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {e}")
        raise
    finally:
        print("\n9. Stopping online sampler...")
        sampler.stop()
        sampler.join()
        print("   Online sampler stopped")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
