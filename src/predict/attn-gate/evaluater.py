import torch
from typing import Optional, Dict, Any, List
import time
import wandb


class GatePredictorEvaluater:
    def __init__(
        self,
        model,
        eval_batch_size: int = 15,
        device: str = "cuda",
        top_k_values: List[int] = None,
        num_active_experts: int = 2,
        use_wandb: bool = True,
        wandb_project: str = "moe-gate-predictor-eval",
        wandb_run_name: Optional[str] = None
    ):
        self.model = model
        self.device = torch.device(device)
        self.eval_batch_size = eval_batch_size
        self.top_k_values = top_k_values if top_k_values is not None else [1, 2, 4]
        self.num_active_experts = num_active_experts
        self.use_wandb = use_wandb

        self.model.to(self.device)
        self.model.to(torch.bfloat16)
        self.model.eval()

        self.batch_buffer = []
        self.total_samples = 0
        self.total_batches = 0
        self.start_time = time.time()

        self.cumulative_top_k_avg = {k: 0.0 for k in self.top_k_values}
        self.cumulative_b_acc = 0.0
        self.cumulative_b_acc_bs1 = 0.0
        self.cumulative_error_rate = 0.0

        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config={
                    "eval_batch_size": eval_batch_size,
                    "device": device,
                    "top_k_values": self.top_k_values,
                    "num_active_experts": num_active_experts
                }
            )

        print(f"GatePredictorEvaluater initialized:")
        print(f"  Model: {model.__class__.__name__}")
        print(f"  Eval batch size: {eval_batch_size}")
        print(f"  Device: {device}")
        print(f"  Top-k values: {self.top_k_values}")
        print(f"  Num active experts: {num_active_experts}")
        print(f"  Use wandb: {use_wandb}")

    def add_sample(self, attn_hidden_states: torch.Tensor, gate_logits: torch.Tensor, seq_lengths: torch.Tensor):
        if attn_hidden_states.dim() == 4 and attn_hidden_states.shape[0] == 1:
            attn_hidden_states = attn_hidden_states.squeeze(0)
        if gate_logits.dim() == 4 and gate_logits.shape[0] == 1:
            gate_logits = gate_logits.squeeze(0)
        if seq_lengths.dim() == 2 and seq_lengths.shape[0] == 1:
            seq_lengths = seq_lengths.squeeze(0)

        self.batch_buffer.append({
            'attn_hidden_states': attn_hidden_states.to(self.device),
            'gate_logits': gate_logits.to(self.device),
            'seq_lengths': seq_lengths.to(self.device)
        })

        if len(self.batch_buffer) >= self.eval_batch_size:
            self._eval_batch()

    def _eval_batch(self):
        if len(self.batch_buffer) == 0:
            return

        batch_data = self._prepare_batch()

        with torch.no_grad():
            top_k_hit_rate = {layer_idx: {k: 0.0 for k in self.top_k_values} for layer_idx in range(self.model.num_layers)}
            batch_b_acc_values = []
            batch_b_acc_bs1_values = []
            batch_error_rate_values = []

            for layer_idx in range(self.model.num_layers):
                layer_attn = batch_data['attn_hidden_states'][:, layer_idx]
                layer_gate = batch_data['gate_logits'][:, layer_idx]
                seq_lengths = batch_data['seq_lengths']

                batch_size, seq_len, hidden_dim = layer_attn.shape
                layer_attn_flat = layer_attn.reshape(-1, hidden_dim)

                predictions = self.model(layer_attn_flat, layer_idx)

                gate_flat = layer_gate.reshape(-1, layer_gate.shape[-1])

                valid_mask = self._create_valid_mask(seq_lengths, seq_len)
                valid = valid_mask.reshape(-1)

                if valid.sum() > 0:
                    predictions_valid = predictions[valid]
                    gate_valid = gate_flat[valid]
                    num_valid_tokens = valid.sum().item()
                    num_experts = gate_valid.shape[-1]

                    for k in self.top_k_values:
                        if k > predictions_valid.shape[-1]:
                            continue
                        pred_top_k = predictions_valid.topk(k, dim=-1).indices
                        true_top_k = gate_valid.topk(k, dim=-1).indices
                        top_k_correct = self._compute_top_k_match(pred_top_k, true_top_k)
                        top_k_hit_rate[layer_idx][k] = top_k_correct / num_valid_tokens

                    true_top_active = gate_valid.topk(self.num_active_experts, dim=-1).indices
                    pred_top_active = predictions_valid.topk(self.num_active_experts, dim=-1).indices

                    R_batch = torch.zeros(num_experts, dtype=torch.bool, device=self.device)
                    R_hat_batch = torch.zeros(num_experts, dtype=torch.bool, device=self.device)
                    R_batch.scatter_(0, true_top_active.reshape(-1), True)
                    R_hat_batch.scatter_(0, pred_top_active.reshape(-1), True)

                    intersection = (R_batch & R_hat_batch).sum().float()
                    R_sum = R_batch.sum().float()
                    b_acc_i = (intersection / R_sum).item() if R_sum > 0 else 0.0
                    batch_b_acc_values.append(b_acc_i)

                    R_all = torch.zeros(num_valid_tokens, num_experts, dtype=torch.bool, device=self.device)
                    R_hat_all = torch.zeros(num_valid_tokens, num_experts, dtype=torch.bool, device=self.device)
                    R_all.scatter_(1, true_top_active, True)
                    R_hat_all.scatter_(1, pred_top_active, True)

                    intersections_t = (R_all & R_hat_all).sum(dim=-1).float()
                    R_sums_t = R_all.sum(dim=-1).float()
                    b_acc_bs1_tokens = intersections_t / R_sums_t.clamp(min=1)
                    batch_b_acc_bs1_values.append(b_acc_bs1_tokens.mean().item())

                    true_counts = torch.zeros(num_experts, dtype=torch.float, device=self.device)
                    pred_counts = torch.zeros(num_experts, dtype=torch.float, device=self.device)
                    true_counts.scatter_add_(0, true_top_active.reshape(-1).long(), torch.ones(true_top_active.numel(), dtype=torch.float, device=self.device))
                    pred_counts.scatter_add_(0, pred_top_active.reshape(-1).long(), torch.ones(pred_top_active.numel(), dtype=torch.float, device=self.device))

                    p = true_counts / true_counts.sum()
                    p_hat = pred_counts / pred_counts.sum()

                    error_rate_i = (p - p_hat).abs().mean().item() / (1.0 / num_experts)
                    batch_error_rate_values.append(error_rate_i)

            batch_avg_top_k = {}
            for k in self.top_k_values:
                batch_avg_top_k[k] = sum(top_k_hit_rate[layer_idx][k] for layer_idx in range(self.model.num_layers)) / self.model.num_layers

            b_acc = sum(batch_b_acc_values) / len(batch_b_acc_values) if batch_b_acc_values else 0.0
            b_acc_bs1 = sum(batch_b_acc_bs1_values) / len(batch_b_acc_bs1_values) if batch_b_acc_bs1_values else 0.0
            error_rate = sum(batch_error_rate_values) / len(batch_error_rate_values) if batch_error_rate_values else 0.0

        self.total_batches += 1
        self.total_samples += len(self.batch_buffer)

        for k in self.top_k_values:
            self.cumulative_top_k_avg[k] = (self.cumulative_top_k_avg[k] * (self.total_batches - 1) + batch_avg_top_k[k]) / self.total_batches
        self.cumulative_b_acc = (self.cumulative_b_acc * (self.total_batches - 1) + b_acc) / self.total_batches
        self.cumulative_b_acc_bs1 = (self.cumulative_b_acc_bs1 * (self.total_batches - 1) + b_acc_bs1) / self.total_batches
        self.cumulative_error_rate = (self.cumulative_error_rate * (self.total_batches - 1) + error_rate) / self.total_batches

        self._log_metrics(top_k_hit_rate, batch_avg_top_k, b_acc, b_acc_bs1, error_rate)

        self.batch_buffer.clear()

    def _compute_top_k_match(self, pred_top_k: torch.Tensor, true_top_k: torch.Tensor) -> int:
        num_tokens = pred_top_k.shape[0]
        correct_count = 0

        for i in range(num_tokens):
            pred_set = set(pred_top_k[i].tolist())
            true_set = set(true_top_k[i].tolist())
            if pred_set == true_set:
                correct_count += 1

        return correct_count

    def _prepare_batch(self) -> Dict[str, torch.Tensor]:
        attn_hidden_states_list = []
        gate_logits_list = []
        seq_lengths_list = []

        max_seq_len = max([item['seq_lengths'].max().item() for item in self.batch_buffer])

        for item in self.batch_buffer:
            attn = item['attn_hidden_states']
            gate = item['gate_logits']
            seq_len = item['seq_lengths']

            if attn.shape[1] < max_seq_len:
                pad_size = max_seq_len - attn.shape[1]
                attn = torch.nn.functional.pad(attn, (0, 0, 0, pad_size), value=0)
                gate = torch.nn.functional.pad(gate, (0, 0, 0, pad_size), value=0)

            attn_hidden_states_list.append(attn)
            gate_logits_list.append(gate)
            seq_lengths_list.append(seq_len)

        return {
            'attn_hidden_states': torch.stack(attn_hidden_states_list, dim=0),
            'gate_logits': torch.stack(gate_logits_list, dim=0),
            'seq_lengths': torch.stack(seq_lengths_list, dim=0)
        }

    def _create_valid_mask(self, seq_lengths: torch.Tensor, seq_len: int) -> torch.Tensor:
        batch_size = seq_lengths.shape[0]
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=self.device)

        for i in range(batch_size):
            mask[i, :seq_lengths[i]] = True

        return mask

    def _log_metrics(self, top_k_hit_rate, batch_avg_top_k, b_acc, b_acc_bs1, error_rate):
        elapsed_time = time.time() - self.start_time
        samples_per_sec = self.total_samples / elapsed_time if elapsed_time > 0 else 0.0

        acc_str = ", ".join([f"top{k}={v:.2%}" for k, v in batch_avg_top_k.items()])
        print(f"  Batch {self.total_batches}: {acc_str}, B_acc={b_acc:.4f}, B_acc_bs1={b_acc_bs1:.4f}, "
              f"Error_rate={error_rate:.4f}, Samples/sec={samples_per_sec:.2f}")

        if self.use_wandb:
            log_dict = {
                "batch": self.total_batches,
                "samples_per_second": samples_per_sec,
                "total_samples": self.total_samples,
                "elapsed_time": elapsed_time,
                "b_acc/batch": b_acc,
                "b_acc/cumulative_avg": self.cumulative_b_acc,
                "b_acc_bs1/batch": b_acc_bs1,
                "b_acc_bs1/cumulative_avg": self.cumulative_b_acc_bs1,
                "error_rate/batch": error_rate,
                "error_rate/cumulative_avg": self.cumulative_error_rate
            }

            for layer_idx in range(self.model.num_layers):
                for k in self.top_k_values:
                    log_dict[f"topk_accuracy/layer_{layer_idx}/k_{k}"] = top_k_hit_rate[layer_idx][k]

            for k in self.top_k_values:
                log_dict[f"topk_accuracy/batch_avg/k_{k}"] = batch_avg_top_k[k]
                log_dict[f"topk_accuracy/cumulative_avg/k_{k}"] = self.cumulative_top_k_avg[k]

            wandb.log(log_dict)

    def flush_remaining(self):
        if len(self.batch_buffer) > 0:
            print(f"  Flushing remaining {len(self.batch_buffer)} samples...")
            self._eval_batch()

    def get_stats(self) -> Dict[str, Any]:
        elapsed_time = time.time() - self.start_time

        return {
            "total_samples": self.total_samples,
            "total_batches": self.total_batches,
            "cumulative_top_k_avg": dict(self.cumulative_top_k_avg),
            "cumulative_b_acc": self.cumulative_b_acc,
            "cumulative_b_acc_bs1": self.cumulative_b_acc_bs1,
            "cumulative_error_rate": self.cumulative_error_rate,
            "elapsed_time": elapsed_time,
            "samples_per_second": self.total_samples / elapsed_time if elapsed_time > 0 else 0.0
        }

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        print(f"  Checkpoint loaded: {path}")
        print(f"    Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"    Total samples (from checkpoint): {checkpoint.get('total_samples', 'unknown')}")
        print(f"    Total batches (from checkpoint): {checkpoint.get('total_batches', 'unknown')}")

    def finish(self):
        self.flush_remaining()

        stats = self.get_stats()

        print("\n" + "=" * 80)
        print("Evaluation Summary:")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Total batches: {stats['total_batches']}")
        print(f"  Elapsed time: {stats['elapsed_time']:.2f}s")
        print(f"  Samples/second: {stats['samples_per_second']:.2f}")

        print("\n  Cumulative top-k accuracy:")
        for k, acc in stats['cumulative_top_k_avg'].items():
            print(f"    top-{k}: {acc:.2%}")

        print(f"\n  Cumulative B_acc: {stats['cumulative_b_acc']:.4f}")
        print(f"  Cumulative B_acc_bs1: {stats['cumulative_b_acc_bs1']:.4f}")
        print(f"  Cumulative Error_rate: {stats['cumulative_error_rate']:.4f}")
        print("=" * 80)

        if self.use_wandb:
            final_log = {
                "final/cumulative_b_acc": stats['cumulative_b_acc'],
                "final/cumulative_b_acc_bs1": stats['cumulative_b_acc_bs1'],
                "final/cumulative_error_rate": stats['cumulative_error_rate'],
                "final_total_samples": stats['total_samples'],
                "final_elapsed_time": stats['elapsed_time']
            }

            for k, acc in stats['cumulative_top_k_avg'].items():
                final_log[f"final/cumulative_top{k}_accuracy"] = acc

            wandb.log(final_log)
            wandb.finish()
