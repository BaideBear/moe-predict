import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict, Any
import time
import wandb


class GatePredictorTrainer:
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        train_batch_size: int = 15,
        device: str = "cuda",
        use_wandb: bool = True,
        wandb_project: str = "moe-gate-predictor",
        wandb_run_name: Optional[str] = None
    ):
        self.model = model
        self.device = torch.device(device)
        self.train_batch_size = train_batch_size
        self.use_wandb = use_wandb
        
        self.model.to(self.device)
        self.model.to(torch.bfloat16)
        self.model.train()
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.batch_buffer = []
        self.total_samples = 0
        self.total_batches = 0
        self.total_loss = 0.0
        self.start_time = time.time()
        
        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config={
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "train_batch_size": train_batch_size,
                    "device": device
                }
            )
        
        print(f"GatePredictorTrainer initialized:")
        print(f"  Model: {model.__class__.__name__}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Weight decay: {weight_decay}")
        print(f"  Train batch size: {train_batch_size}")
        print(f"  Device: {device}")
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
        
        if len(self.batch_buffer) >= self.train_batch_size:
            self._train_batch()
    
    def _train_batch(self):
        if len(self.batch_buffer) == 0:
            return
        
        batch_data = self._prepare_batch()
        
        self.optimizer.zero_grad()
        
        total_loss = torch.tensor(0.0, device=self.device)
        num_tokens = 0
        total_correct = 0
        total_top1_correct = 0
        total_top2_correct = 0
        
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
                
                loss = self.criterion(predictions_valid, gate_valid.argmax(dim=-1))
                total_loss = total_loss + loss
                
                num_tokens += valid.sum().item()
                
                predicted_experts = predictions_valid.argmax(dim=-1)
                true_experts = gate_valid.argmax(dim=-1)
                
                correct = (predicted_experts == true_experts).sum().item()
                total_correct += correct
                
                true_top2_indices = gate_valid.topk(2, dim=-1).indices
                top2_indices = predictions_valid.topk(2, dim=-1).indices
                
                top1_correct = (top2_indices[:, 0] == true_top2_indices[:, 0]).sum().item()
                
                # 检查top2的expert是否有交集
                # pred_in_true_top2 = (top2_indices.unsqueeze(-1) == true_top2_indices.unsqueeze(1)).any(dim=-1)
                # top2_correct = pred_in_true_top2.any(dim=-1).sum().item()
                # 检查top2包含的expert是否完全相同
                top2_correct = ((top2_indices[:, 0] == true_experts) | (top2_indices[:, 1] == true_experts)).sum().item()
                
                total_top1_correct += top1_correct
                total_top2_correct += top2_correct
        
        if num_tokens > 0:
            avg_loss = total_loss / self.model.num_layers
            avg_loss.backward()
            self.optimizer.step()
        
        self.total_loss += avg_loss.item()
        self.total_batches += 1
        self.total_samples += len(self.batch_buffer)
        
        accuracy = total_correct / num_tokens if num_tokens > 0 else 0.0
        top1_accuracy = total_top1_correct / num_tokens if num_tokens > 0 else 0.0
        top2_accuracy = total_top2_correct / num_tokens if num_tokens > 0 else 0.0
        
        self._log_metrics(avg_loss.item(), num_tokens, accuracy, top1_accuracy, top2_accuracy)
        
        self.batch_buffer.clear()
    
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
    
    def _log_metrics(self, loss: float, num_tokens: int, accuracy: float, top1_accuracy: float, top2_accuracy: float):
        elapsed_time = time.time() - self.start_time
        avg_loss = self.total_loss / self.total_batches if self.total_batches > 0 else 0.0
        samples_per_per_sec = self.total_samples / elapsed_time if elapsed_time > 0 else 0.0
        
        print(f"  Batch {self.total_batches}: Loss={loss:.4f}, "
              f"Avg Loss={avg_loss:.4f}, Tokens={num_tokens}, "
              f"Acc={accuracy:.2%}, Top1={top1_accuracy:.2%}, Top2={top2_accuracy:.2%}, "
              f"Samples/sec={samples_per_per_sec:.2f}")
        
        if self.use_wandb:
            wandb.log({
                "batch": self.total_batches,
                "loss": loss,
                "avg_loss": avg_loss,
                "num_tokens": num_tokens,
                "accuracy": accuracy,
                "top1_accuracy": top1_accuracy,
                "top2_accuracy": top2_accuracy,
                "samples_per_second": samples_per_per_sec,
                "total_samples": self.total_samples,
                "elapsed_time": elapsed_time
            })
    
    def flush_remaining(self):
        if len(self.batch_buffer) > 0:
            print(f"  Flushing remaining {len(self.batch_buffer)} samples...")
            self._train_batch()
    
    def get_stats(self) -> Dict[str, Any]:
        elapsed_time = time.time() - self.start_time
        avg_loss = self.total_loss / self.total_batches if self.total_batches > 0 else 0.0
        
        return {
            "total_samples": self.total_samples,
            "total_batches": self.total_batches,
            "avg_loss": avg_loss,
            "elapsed_time": elapsed_time,
            "samples_per_second": self.total_samples / elapsed_time if elapsed_time > 0 else 0.0
        }
    
    def save_checkpoint(self, path: str, epoch: int, additional_info: Optional[Dict[str, Any]] = None):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_samples': self.total_samples,
            'total_batches': self.total_batches,
            'total_loss': self.total_loss,
            'stats': self.get_stats()
        }
        
        if additional_info:
            checkpoint.update(additional_info)
        
        torch.save(checkpoint, path)
        print(f"  Checkpoint saved: {path}")
        
        if self.use_wandb:
            wandb.save(path)
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_samples = checkpoint.get('total_samples', 0)
        self.total_batches = checkpoint.get('total_batches', 0)
        self.total_loss = checkpoint.get('total_loss', 0.0)
        
        print(f"  Checkpoint loaded: {path}")
        print(f"    Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"    Total samples: {self.total_samples}")
        print(f"    Total batches: {self.total_batches}")
    
    def finish(self):
        self.flush_remaining()
        
        stats = self.get_stats()
        
        print("\n" + "=" * 80)
        print("Training Summary:")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Total batches: {stats['total_batches']}")
        print(f"  Average loss: {stats['avg_loss']:.4f}")
        print(f"  Elapsed time: {stats['elapsed_time']:.2f}s")
        print(f"  Samples/second: {stats['samples_per_second']:.2f}")
        print("=" * 80)
        
        if self.use_wandb:
            wandb.log({
                "final_avg_loss": stats['avg_loss'],
                "final_total_samples": stats['total_samples'],
                "final_elapsed_time": stats['elapsed_time']
            })
            wandb.finish()
