import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
import time
import wandb


class GatePredictorEvaluater:
    def __init__(
        self,
        model: nn.Module,
        eval_batch_size: int = 15,
        device: str = "cuda",
        top_k_values: List[int] = None,
        use_wandb: bool = True,
        wandb_project: str = "moe-gate-predictor-eval",
        wandb_run_name: Optional[str] = None
    ):
        self.model = model
        self.device = torch.device(device)
        self.eval_batch_size = eval_batch_size
        self.top_k_values = top_k_values if top_k_values is not None else [1, 2, 4]
        self.use_wandb = use_wandb
        
        self.model.to(self.device)
        self.model.to(torch.bfloat16)
        self.model.eval()
        
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        
        self.batch_buffer = []
        self.total_samples = 0
        self.total_batches = 0
        self.start_time = time.time()
        
        self.layer_losses = {layer_idx: 0.0 for layer_idx in range(model.num_layers)}
        self.layer_top_k_correct = {layer_idx: {k: 0 for k in self.top_k_values} for layer_idx in range(model.num_layers)}
        self.layer_total_tokens = {layer_idx: 0 for layer_idx in range(model.num_layers)}
        
        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config={
                    "eval_batch_size": eval_batch_size,
                    "device": device,
                    "top_k_values": self.top_k_values
                }
            )
        
        print(f"GatePredictorEvaluater initialized:")
        print(f"  Model: {model.__class__.__name__}")
        print(f"  Eval batch size: {eval_batch_size}")
        print(f"  Device: {device}")
        print(f"  Top-k values: {self.top_k_values}")
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
            batch_layer_losses = {layer_idx: 0.0 for layer_idx in range(self.model.num_layers)}
            batch_layer_top_k_correct = {layer_idx: {k: 0 for k in self.top_k_values} for layer_idx in range(self.model.num_layers)}
            batch_layer_total_tokens = {layer_idx: 0 for layer_idx in range(self.model.num_layers)}
            
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
                    
                    loss_per_token = self.criterion(predictions_valid, gate_valid.argmax(dim=-1))
                    batch_layer_losses[layer_idx] = loss_per_token.mean().item()
                    
                    num_valid_tokens = valid.sum().item()
                    batch_layer_total_tokens[layer_idx] = num_valid_tokens
                    
                    for k in self.top_k_values:
                        if k > predictions_valid.shape[-1]:
                            continue
                        
                        pred_top_k = predictions_valid.topk(k, dim=-1).indices
                        true_top_k = gate_valid.topk(k, dim=-1).indices
                        
                        top_k_correct = self._compute_top_k_match(pred_top_k, true_top_k)
                        batch_layer_top_k_correct[layer_idx][k] = top_k_correct
            
            for layer_idx in range(self.model.num_layers):
                self.layer_losses[layer_idx] += batch_layer_losses[layer_idx]
                self.layer_total_tokens[layer_idx] += batch_layer_total_tokens[layer_idx]
                for k in self.top_k_values:
                    self.layer_top_k_correct[layer_idx][k] += batch_layer_top_k_correct[layer_idx][k]
        
        self.total_batches += 1
        self.total_samples += len(self.batch_buffer)
        
        self._log_metrics(batch_layer_losses, batch_layer_top_k_correct, batch_layer_total_tokens)
        
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
    
    def _log_metrics(
        self,
        batch_layer_losses: Dict[int, float],
        batch_layer_top_k_correct: Dict[int, Dict[int, int]],
        batch_layer_total_tokens: Dict[int, int]
    ):
        elapsed_time = time.time() - self.start_time
        samples_per_sec = self.total_samples / elapsed_time if elapsed_time > 0 else 0.0
        
        avg_loss = sum(batch_layer_losses.values()) / len(batch_layer_losses) if batch_layer_losses else 0.0
        
        avg_top_k_accuracy = {}
        for k in self.top_k_values:
            total_correct = 0
            total_tokens = 0
            for layer_idx in batch_layer_top_k_correct.keys():
                total_correct += batch_layer_top_k_correct[layer_idx][k]
                total_tokens += batch_layer_total_tokens[layer_idx]
            avg_top_k_accuracy[k] = total_correct / total_tokens if total_tokens > 0 else 0.0
        
        acc_str = ", ".join([f"top{k}={acc:.2%}" for k, acc in avg_top_k_accuracy.items()])
        print(f"  Batch {self.total_batches}: Avg Loss={avg_loss:.4f}, {acc_str}, "
              f"Samples/sec={samples_per_sec:.2f}")
        
        if self.use_wandb:
            log_dict = {
                "batch": self.total_batches,
                "avg_loss": avg_loss,
                "samples_per_second": samples_per_sec,
                "total_samples": self.total_samples,
                "elapsed_time": elapsed_time
            }
            
            for k in self.top_k_values:
                log_dict[f"avg_top{k}_accuracy"] = avg_top_k_accuracy[k]
            
            for layer_idx in batch_layer_losses.keys():
                log_dict[f"layer_{layer_idx}_loss"] = batch_layer_losses[layer_idx]
                
                total_tokens = batch_layer_total_tokens[layer_idx]
                for k in self.top_k_values:
                    if total_tokens > 0:
                        accuracy = batch_layer_top_k_correct[layer_idx][k] / total_tokens
                        log_dict[f"layer_{layer_idx}_top{k}_accuracy"] = accuracy
            
            wandb.log(log_dict)
    
    def flush_remaining(self):
        if len(self.batch_buffer) > 0:
            print(f"  Flushing remaining {len(self.batch_buffer)} samples...")
            self._eval_batch()
    
    def get_stats(self) -> Dict[str, Any]:
        elapsed_time = time.time() - self.start_time
        
        avg_layer_losses = {}
        for layer_idx in self.layer_losses.keys():
            avg_layer_losses[layer_idx] = self.layer_losses[layer_idx] / self.total_batches if self.total_batches > 0 else 0.0
        
        avg_layer_top_k_accuracy = {}
        for layer_idx in self.layer_top_k_correct.keys():
            avg_layer_top_k_accuracy[layer_idx] = {}
            for k in self.top_k_values:
                if self.layer_total_tokens[layer_idx] > 0:
                    avg_layer_top_k_accuracy[layer_idx][k] = self.layer_top_k_correct[layer_idx][k] / self.layer_total_tokens[layer_idx]
                else:
                    avg_layer_top_k_accuracy[layer_idx][k] = 0.0
        
        overall_avg_loss = sum(avg_layer_losses.values()) / len(avg_layer_losses) if avg_layer_losses else 0.0
        
        return {
            "total_samples": self.total_samples,
            "total_batches": self.total_batches,
            "avg_layer_losses": avg_layer_losses,
            "avg_layer_top_k_accuracy": avg_layer_top_k_accuracy,
            "overall_avg_loss": overall_avg_loss,
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
        print(f"  Overall average loss: {stats['overall_avg_loss']:.4f}")
        print(f"  Elapsed time: {stats['elapsed_time']:.2f}s")
        print(f"  Samples/second: {stats['samples_per_second']:.2f}")
        
        print("\n  Per-layer losses:")
        for layer_idx, loss in stats['avg_layer_losses'].items():
            print(f"    Layer {layer_idx}: {loss:.4f}")
        
        print("\n  Per-layer top-k accuracy:")
        for layer_idx, top_k_acc in stats['avg_layer_top_k_accuracy'].items():
            acc_str = ", ".join([f"top{k}={acc:.2%}" for k, acc in top_k_acc.items()])
            print(f"    Layer {layer_idx}: {acc_str}")
        
        print("=" * 80)
        
        if self.use_wandb:
            final_log = {
                "final_avg_loss": stats['overall_avg_loss'],
                "final_total_samples": stats['total_samples'],
                "final_elapsed_time": stats['elapsed_time']
            }
            
            for k in self.top_k_values:
                total_correct = 0
                total_tokens = 0
                for layer_idx in stats['avg_layer_top_k_accuracy'].keys():
                    total_correct += self.layer_top_k_correct[layer_idx][k]
                    total_tokens += self.layer_total_tokens[layer_idx]
                final_log[f"final_avg_top{k}_accuracy"] = total_correct / total_tokens if total_tokens > 0 else 0.0
            
            for layer_idx, loss in stats['avg_layer_losses'].items():
                final_log[f"final_layer_{layer_idx}_loss"] = loss
            
            for layer_idx, top_k_acc in stats['avg_layer_top_k_accuracy'].items():
                for k, acc in top_k_acc.items():
                    final_log[f"final_layer_{layer_idx}_top{k}_accuracy"] = acc
            
            wandb.log(final_log)
            wandb.finish()
