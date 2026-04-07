import os
import sys
import torch
import torch.nn.functional as F
import argparse
import json
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from transformers import AutoModelForCausalLM, AutoTokenizer
from online_sample import (
    create_buffer,
    extract_model_config,
    OnlineSampler,
    create_predictor_interface
)
from predict.PROBE.model import create_predictor_model


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

        print(f"PROBE PredictorEvaluator initialized:")
        print(f"  Pattern: {pattern}")
        print(f"  Device: {device}")

    def evaluate(self, top_k: int = 8, max_samples: Optional[int] = None):
        """
        评估预测器性能，报告每层和平均指标。

        指标（含论文 Figure 9 要求的额外指标）:
          - Top-1 Accuracy
          - Top-k Hit Rate
          - Top-Half-k Hit Rate      （预测前 ⌊k/2⌋ 个中包含真实前 k 中的几个）
          - 2x Top-k Recall           （预测前 2k 个中包含真实前 k 的比例）
          - KL-div(predictor || gate)
        """
        self.predict_model.eval()
        num_layers = self.predict_model.num_layers

        half_k = max(1, top_k // 2)

        # 每层累积统计
        layer_top1_correct = [0] * num_layers
        layer_topk_hits = [0] * num_layers
        layer_halfk_hits = [0] * num_layers     # 预测前 half_k 命中真实前 top_k
        layer_doublek_hits = [0] * num_layers    # 预测前 2*top_k 命中真实前 top_k
        layer_total_tokens = [0] * num_layers
        layer_kl_sum = [0.0] * num_layers

        sample_count = 0

        print("Starting PROBE evaluation...")
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
                attn_states = data.attn_hidden_states
                gate_logits = data.gate_logits
                seq_lengths = data.seq_lengths

                num_samples_dim, num_layers_dim, max_seq_len, hidden_dim = attn_states.shape

                sample_top1_correct = [0] * num_layers_dim
                sample_topk_hits = [0] * num_layers_dim
                sample_halfk_hits = [0] * num_layers_dim
                sample_doublek_hits = [0] * num_layers_dim
                sample_tokens = [0] * num_layers_dim
                sample_kl_sum = [0.0] * num_layers_dim

                for layer_idx in range(num_layers_dim):
                    # 拼合一个 batch 中所有样本该层的 token
                    input_list = []
                    target_list = []
                    for sample_idx in range(num_samples_dim):
                        seq_len = seq_lengths[sample_idx].item()
                        input_list.append(attn_states[sample_idx, layer_idx, :seq_len, :])
                        target_list.append(gate_logits[sample_idx, layer_idx, :seq_len, :])

                    if not input_list:
                        continue

                    all_inputs = torch.cat(input_list, dim=0)   # [N, hidden_dim]
                    all_targets = torch.cat(target_list, dim=0)  # [N, num_experts]  — raw logits

                    with torch.no_grad():
                        pred_logits = self.predict_model(all_inputs.to(self.device).float(), layer_idx)
                        pred_probs = F.softmax(pred_logits, dim=-1)
                        target_probs = F.softmax(all_targets.to(self.device).float(), dim=-1)

                    N_tokens = all_inputs.shape[0]

                    # Top-1 Accuracy
                    _, top1_real = torch.topk(target_probs, k=1, dim=-1)
                    _, top1_pred = torch.topk(pred_probs, k=1, dim=-1)
                    top1_real_ids = top1_real.squeeze(-1)
                    top1_pred_ids = top1_pred.squeeze(-1)
                    sample_top1_correct[layer_idx] = (top1_real_ids == top1_pred_ids).sum().item()

                    # Top-k & Half-k & 2×k
                    _, topk_real = torch.topk(target_probs, k=top_k, dim=-1)
                    real_topk_sets = [set(r.tolist()) for r in topk_real]

                    _, topk_pred = torch.topk(pred_probs, k=top_k, dim=-1)
                    pred_topk_sets = [set(p.tolist()) for p in topk_pred]

                    # 预测前 half_k
                    if all_inputs.shape[0] > 0:
                        half_k_actual = min(half_k, topk_pred.shape[-1])
                        _, top_halfk_pred = torch.topk(pred_probs, k=half_k_actual, dim=-1)
                        pred_halfk_sets = [set(p.tolist()) for p in top_halfk_pred]

                        # 预测前 2×k
                        num_experts = pred_probs.shape[-1]
                        double_k_actual = min(top_k * 2, num_experts)
                        _, top_doublek_pred = torch.topk(pred_probs, k=double_k_actual, dim=-1)
                        pred_doublek_sets = [set(p.tolist()) for p in top_doublek_pred]

                    for i in range(N_tokens):
                        real_set = real_topk_sets[i]
                        sample_topk_hits[layer_idx] += len(real_set.intersection(pred_topk_sets[i]))
                        sample_tokens[layer_idx] += 1

                        sample_halfk_hits[layer_idx] += len(real_set.intersection(pred_halfk_sets[i]))
                        sample_doublek_hits[layer_idx] += len(real_set.intersection(pred_doublek_sets[i]))

                        # KL-div
                        sample_kl_sum[layer_idx] += F.kl_div(
                            pred_probs[i].log(), target_probs[i], reduction='sum'
                        ).item()

                for layer_idx in range(num_layers_dim):
                    layer_top1_correct[layer_idx] += sample_top1_correct[layer_idx]
                    layer_topk_hits[layer_idx] += sample_topk_hits[layer_idx]
                    layer_halfk_hits[layer_idx] += sample_halfk_hits[layer_idx]
                    layer_doublek_hits[layer_idx] += sample_doublek_hits[layer_idx]
                    layer_total_tokens[layer_idx] += sample_tokens[layer_idx]
                    layer_kl_sum[layer_idx] += sample_kl_sum[layer_idx]

                sample_count += 1

                total_toks = sum(sample_tokens)
                if total_toks > 0:
                    avg_top1 = sum(sample_top1_correct) / total_toks
                    avg_topk = sum(sample_topk_hits) / (total_toks * top_k)
                    print(f"Sample {sample_count}: {total_toks} tokens | "
                          f"Top-1: {avg_top1:.4f} | Top-{top_k}: {avg_topk:.4f}")

            if max_samples is not None and sample_count >= max_samples:
                break

        if sample_count == 0:
            print("No data collected for evaluation")
            return None

        # ====== 汇总输出 ======
        valid_layers = sum(1 for t in layer_total_tokens if t > 0)
        if valid_layers == 0:
            print("No valid layers for evaluation")
            return None

        print("=" * 60)
        print(f"Evaluation Results (Top-k={top_k}, Half-k={half_k}, 2xk={top_k*2}):")
        print("=" * 60)

        for layer_idx in range(num_layers):
            t = layer_total_tokens[layer_idx]
            if t == 0:
                continue
            top1 = layer_top1_correct[layer_idx] / t
            topk = layer_topk_hits[layer_idx] / (t * top_k)
            halfk = layer_halfk_hits[layer_idx] / (t * top_k)
            dblk = layer_doublek_hits[layer_idx] / (t * top_k)
            kl_avg = layer_kl_sum[layer_idx] / t
            print(f"  Layer {layer_idx:2d}: {t:6d} tokens | "
                  f"Top-1: {top1:.4f} | Top-{top_k}: {topk:.4f} | "
                  f"Half-{top_k}: {halfk:.4f} | 2x{top_k}: {dblk:.4f} | KL: {kl_avg:.4f}")

        # 全局平均（层间平均）
        avg_top1 = sum(layer_top1_correct[l] / layer_total_tokens[l] for l in range(num_layers) if layer_total_tokens[l] > 0) / valid_layers
        avg_topk = sum(layer_topk_hits[l] / (layer_total_tokens[l] * top_k) for l in range(num_layers) if layer_total_tokens[l] > 0) / valid_layers
        avg_halfk = sum(layer_halfk_hits[l] / (layer_total_tokens[l] * top_k) for l in range(num_layers) if layer_total_tokens[l] > 0) / valid_layers
        avg_dblk = sum(layer_doublek_hits[l] / (layer_total_tokens[l] * top_k) for l in range(num_layers) if layer_total_tokens[l] > 0) / valid_layers
        avg_kl = sum(layer_kl_sum[l] / layer_total_tokens[l] for l in range(num_layers) if layer_total_tokens[l] > 0) / valid_layers

        print("=" * 60)
        print(f"Average across {valid_layers} layers:")
        print(f"  Average Top-1 Accuracy:        {avg_top1:.4f}")
        print(f"  Average Top-{top_k} Hit Rate:    {avg_topk:.4f}")
        print(f"  Average Top-Half-{top_k} Hit Rate: {avg_halfk:.4f}")
        print(f"  Average 2x Top-{top_k} Recall:   {avg_dblk:.4f}")
        print(f"  Average KL-div(predictor || gate): {avg_kl:.4f}")
        print("=" * 60)

        return {
            'top1_accuracy': avg_top1,
            f'top{top_k}_hit_rate': avg_topk,
            f'top_half_{top_k}_hit_rate': avg_halfk,
            f'top_double_{top_k}_recall': avg_dblk,
            'avg_kl_div': avg_kl,
            'total_samples': sample_count,
            'num_layers': num_layers,
            'valid_layers': valid_layers,
        }


def main():
    parser = argparse.ArgumentParser(description='Test PROBE Predictor')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the base model')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the predictor checkpoint')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the test dataset')
    parser.add_argument('--pattern', type=str, default='attn_gate',
                        choices=['attn_gate', 'gate_input', 'token_gate'], help='Data pattern')
    parser.add_argument('--buffer_size_gb', type=float, default=4.0, help='Buffer size in GB')
    parser.add_argument('--batch_size', type=int, default=1, help='Sampler batch size')
    parser.add_argument('--max_seq_length', type=int, default=2048, help='Maximum sequence length')
    parser.add_argument('--top_k', type=int, default=8, help='Top-k for evaluation')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to evaluate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')

    args = parser.parse_args()

    print("=" * 60)
    print("PROBE Predictor Evaluation")
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

    print("\n4. Loading PROBE predictor model...")
    predict_model = create_predictor_model(
        model=model,
        num_layers=model_config.num_layers,
        input_dim=model_config.hidden_dim,
        num_experts=model_config.num_experts,
    )

    checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
    if 'model_state_dict' in checkpoint:
        predict_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        predict_model.load_state_dict(checkpoint)

    predict_model.eval()
    print(f"   Predictor loaded from: {args.checkpoint_path}")

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

    print("\n8. Starting evaluation...")
    print("=" * 60)

    try:
        metrics = evaluator.evaluate(
            top_k=args.top_k,
            max_samples=args.max_samples
        )

        if metrics is not None:
            print("\n" + "=" * 60)
            print("Evaluation Summary")
            print("=" * 60)
            print(f"Total samples evaluated: {metrics['total_samples']}")
            print(f"Top-1 Accuracy:          {metrics['top1_accuracy'] * 100:.2f}%")
            print(f"Top-{args.top_k} Hit Rate:        {metrics[f'top{args.top_k}_hit_rate'] * 100:.2f}%")
            print(f"Top-Half-{args.top_k} Hit Rate:   {metrics[f'top_half_{args.top_k}_hit_rate'] * 100:.2f}%")
            print(f"2x Top-{args.top_k} Recall:       {metrics[f'top_double_{args.top_k}_recall'] * 100:.2f}%")
            print(f"Average KL-div:          {metrics['avg_kl_div']:.4f}")
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
    print("PROBE Evaluation completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
