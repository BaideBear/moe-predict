import os
import sys
import torch
import argparse
import json
from typing import Optional
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from transformers import AutoModelForCausalLM, AutoTokenizer
from online_sample import (
    create_buffer,
    extract_model_config,
    OnlineSampler,
    create_predictor_interface
)
from predict.PEPP.model import create_predictor_model


class PredictorEvaluator:
    def __init__(
        self,
        predict_model,
        buffer,
        pattern: str = "attn_gate",
        device: str = "cuda"
    ):
        self.predict_model = predict_model
        self.buffer = buffer
        self.pattern = pattern
        self.device = torch.device(device)
        
        self.predict_model.to(self.device)
        self.predict_model.eval()
        
        self.interface = create_predictor_interface(
            buffer=buffer,
            pattern=pattern,
            batch_size=1
        )
        
        print(f"PredictorEvaluator initialized:")
        print(f"  Pattern: {pattern}")
        print(f"  Device: {device}")
    
    def extract_data(self, data):
        """
        从ActivationData中提取输入输出对
        返回：每层的独立数据列表 [(layer_idx, inputs, outputs), ...]
        """
        if self.pattern == "attn_gate":
            attn_hidden_states = data.attn_hidden_states
            gate_logits = data.gate_logits
            seq_lengths = data.seq_lengths
            
            num_samples, num_layers, max_seq_len, hidden_dim = attn_hidden_states.shape
            num_experts = gate_logits.shape[-1]
            
            layer_data_list = []
            
            for sample_idx in range(num_samples):
                seq_len = seq_lengths[sample_idx].item()
                
                for layer_idx in range(num_layers):
                    layer_input = attn_hidden_states[sample_idx, layer_idx, :seq_len, :]
                    layer_output = gate_logits[sample_idx, layer_idx, :seq_len, :]
                    
                    layer_data_list.append((layer_idx, layer_input, layer_output))
            
            return layer_data_list
        else:
            raise ValueError(f"Unsupported pattern: {self.pattern}")
    
    def evaluate(self, top_k: int = 8, max_samples: Optional[int] = None):
        """
        评估预测器性能
        分别统计每层的准确率，然后给出平均准确率
        """
        self.predict_model.eval()
        
        num_layers = self.predict_model.num_layers
        layer_top1_correct = [0] * num_layers
        layer_topk_hits = [0] * num_layers
        layer_total_tokens = [0] * num_layers
        sample_count = 0
        
        print("Starting evaluation...")
        print("=" * 60)
        
        while True:
            if max_samples is not None and sample_count >= max_samples:
                break
            
            batch = self.interface.get_batch()
            
            if batch is None:
                if not self.interface.is_buffer_empty():
                    continue
                break
            
            for data in batch:
                layer_data_list = self.extract_data(data)
                if len(layer_data_list) > 0:
                    sample_top1_correct = [0] * num_layers
                    sample_topk_hits = [0] * num_layers
                    sample_tokens = [0] * num_layers
                    
                    for layer_idx, inputs, outputs in layer_data_list:
                        num_tokens = inputs.shape[0]
                        
                        inputs_tensor = inputs.to(self.device).float()
                        outputs_tensor = outputs.to(self.device).float()
                        
                        with torch.no_grad():
                            predictions = self.predict_model(inputs_tensor, layer_idx)
                        
                        _, topk_real_idx = torch.topk(outputs_tensor, k=top_k, dim=-1)
                        _, topk_pred_idx = torch.topk(predictions, k=top_k, dim=-1)
                        
                        top1_real = topk_real_idx[:, 0]
                        top1_pred = topk_pred_idx[:, 0]
                        top1_correct = (top1_real == top1_pred).float().sum().item()
                        
                        hit_counts = 0
                        for i in range(num_tokens):
                            real_set = set(topk_real_idx[i].tolist())
                            pred_set = set(topk_pred_idx[i].tolist())
                            hit_counts += len(real_set.intersection(pred_set))
                        
                        sample_top1_correct[layer_idx] = top1_correct
                        sample_topk_hits[layer_idx] = hit_counts
                        sample_tokens[layer_idx] = num_tokens
                    
                    for layer_idx in range(num_layers):
                        layer_top1_correct[layer_idx] += sample_top1_correct[layer_idx]
                        layer_topk_hits[layer_idx] += sample_topk_hits[layer_idx]
                        layer_total_tokens[layer_idx] += sample_tokens[layer_idx]
                    
                    sample_count += 1
                    
                    total_tokens = sum(sample_tokens)
                    sample_top1_acc = sum(sample_top1_correct) / total_tokens if total_tokens > 0 else 0.0
                    sample_topk_hit = sum(sample_topk_hits) / (total_tokens * top_k) if total_tokens > 0 else 0.0
                    
                    print(f"Sample {sample_count}: {total_tokens} tokens (avg {total_tokens//num_layers}/layer) | "
                          f"Top-1 Acc: {sample_top1_acc:.4f} | "
                          f"Top-{top_k} Hit Rate: {sample_topk_hit:.4f}")
                
                if max_samples is not None and sample_count >= max_samples:
                    break
        
        if sample_count == 0:
            print("No data collected for evaluation")
            return None
        
        print("=" * 60)
        print("Evaluation Summary by Layer:")
        print("=" * 60)
        
        avg_top1_acc = 0.0
        avg_topk_hit_rate = 0.0
        
        for layer_idx in range(num_layers):
            if layer_total_tokens[layer_idx] > 0:
                layer_top1_acc = layer_top1_correct[layer_idx] / layer_total_tokens[layer_idx]
                layer_topk_hit = layer_topk_hits[layer_idx] / (layer_total_tokens[layer_idx] * top_k)
                
                avg_top1_acc += layer_top1_acc
                avg_topk_hit_rate += layer_topk_hit
                
                print(f"  Layer {layer_idx:2d}: {layer_total_tokens[layer_idx]:6d} tokens | "
                      f"Top-1 Acc: {layer_top1_acc:.4f} | "
                      f"Top-{top_k} Hit Rate: {layer_topk_hit:.4f}")
        
        avg_top1_acc /= num_layers
        avg_topk_hit_rate /= num_layers
        
        print("=" * 60)
        print(f"Average across {num_layers} layers:")
        print(f"  Average Top-1 Accuracy: {avg_top1_acc:.4f}")
        print(f"  Average Top-{top_k} Hit Rate: {avg_topk_hit_rate:.4f}")
        print("=" * 60)
        
        metrics = {
            'top1_accuracy': avg_top1_acc,
            f'top{top_k}_hit_rate': avg_topk_hit_rate,
            'total_samples': sample_count,
            'num_layers': num_layers,
            'layer_top1_accuracy': [layer_top1_correct[i] / layer_total_tokens[i] if layer_total_tokens[i] > 0 else 0.0 for i in range(num_layers)],
            f'layer_top{top_k}_hit_rate': [layer_topk_hits[i] / (layer_total_tokens[i] * top_k) if layer_total_tokens[i] > 0 else 0.0 for i in range(num_layers)]
        }
        
        return metrics


def load_gsm8k_prompts(file_path, max_samples=200):
    """
    读取GSM8K测试集，并转换为测试用的Prompt列表
    """
    test_texts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            data = json.loads(line)
            prompt = f"Question: {data['question']}\nAnswer:"
            test_texts.append(prompt)
    return test_texts


def main():
    parser = argparse.ArgumentParser(description='Test PEPP Predictor')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the base model')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the predictor checkpoint')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the test dataset')
    parser.add_argument('--pattern', type=str, default='attn_gate', choices=['attn_gate', 'gate_input', 'token_gate'], help='Data pattern')
    parser.add_argument('--buffer_size_gb', type=float, default=4.0, help='Buffer size in GB')
    parser.add_argument('--batch_size', type=int, default=1, help='Sampler batch size')
    parser.add_argument('--max_seq_length', type=int, default=2048, help='Maximum sequence length')
    parser.add_argument('--top_k', type=int, default=8, help='Top-k for evaluation')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to evaluate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PEPP Predictor Evaluation")
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
    
    print("\n2. Extracting model configuration...")
    model_config = extract_model_config(model, args.model_path, args.max_seq_length)
    print(f"   Model name: {model_config.model_name}")
    print(f"   Number of layers: {model_config.num_layers}")
    print(f"   Hidden dimension: {model_config.hidden_dim}")
    print(f"   Number of experts: {model_config.num_experts}")
    
    print("\n3. Creating buffer...")
    buffer = create_buffer(
        model_config=model_config,
        pattern=args.pattern,
        buffer_size_gb=args.buffer_size_gb,
        device=args.device
    )
    print(f"   Buffer created with pattern: {args.pattern}")
    
    print("\n4. Loading predictor model...")
    predict_model = create_predictor_model(
        num_layers=model_config.num_layers,
        input_dim=model_config.hidden_dim,
        num_experts=model_config.num_experts,
        hidden_dim=2048,
        dropout=0.1
    )
    
    checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
    if 'model_state_dict' in checkpoint:
        predict_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        predict_model.load_state_dict(checkpoint)
    
    predict_model.eval()
    print(f"   Predictor loaded from: {args.checkpoint_path}")
    print(f"     Number of layers: {model_config.num_layers}")
    
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
    
    print("\n6. Starting online sampler...")
    sampler.start()
    print("   Online sampler started (running in background)")
    
    print("\n7. Creating evaluator...")
    evaluator = PredictorEvaluator(
        predict_model=predict_model,
        buffer=buffer,
        pattern=args.pattern,
        device=args.device
    )
    print(f"   Evaluator created:")
    print(f"     Top-k: {args.top_k}")
    
    print("\n8. Starting evaluation...")
    print("=" * 60)
    
    try:
        metrics = evaluator.evaluate(
            top_k=args.top_k,
            max_samples=args.max_samples
        )
        
        if metrics is not None:
            print("\n" + "=" * 60)
            print("Evaluation Results")
            print("=" * 60)
            print(f"Total tokens evaluated: {metrics['total_tokens']}")
            print(f"Total samples evaluated: {metrics['total_samples']}")
            print(f"Top-1 Accuracy: {metrics['top1_accuracy'] * 100:.2f}%")
            print(f"Top-{args.top_k} Hit Rate: {metrics[f'top{args.top_k}_hit_rate'] * 100:.2f}%")
            print("=" * 60)
        else:
            print("\nNo evaluation metrics available")
    
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        raise
    finally:
        print("\n9. Stopping online sampler...")
        sampler.stop()
        sampler.join()
        print("   Online sampler stopped")
    
    print("\n" + "=" * 60)
    print("Evaluation completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
