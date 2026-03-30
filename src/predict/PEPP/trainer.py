import torch
import torch.optim as optim
from typing import Optional, Dict, Any
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from online_sample import ActivationBuffer, create_predictor_interface
from .model import PredictorModel
from .loss import compute_custom_loss


class PredictorTrainer:
    def __init__(
        self,
        buffer: ActivationBuffer,
        predict_model: PredictorModel,
        pattern: str = "attn_gate",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        top_k_routing: int = 8,
        lambda_rank: float = 1.0,
        margin: float = 0.1,
        checkpoint_interval: int = 50,
        device: str = "cuda"
    ):
        self.buffer = buffer
        self.predict_model = predict_model
        self.pattern = pattern
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.top_k_routing = top_k_routing
        self.lambda_rank = lambda_rank
        self.margin = margin
        self.checkpoint_interval = checkpoint_interval
        self.device = torch.device(device)
        
        self.predict_model.to(self.device)
        self.optimizer = optim.AdamW(
            self.predict_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        self.interface = create_predictor_interface(
            buffer=buffer,
            pattern=pattern,
            batch_size=1
        )
        
        print(f"PredictorTrainer initialized:")
        print(f"  Pattern: {pattern}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Checkpoint interval: {checkpoint_interval} samples")
        print(f"  Device: {device}")
    
    def extract_data(self, data):
        """
        从ActivationData中提取输入输出对
        对于attn_gate模式：
        - 输入：attn_hidden_states [num_samples, num_layers, seq_len, hidden_dim]
        - 输出：gate_logits [num_samples, num_layers, seq_len, num_experts]
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
    
    def train_step(self, layer_data_list):
        """
        训练一个样本的所有层
        先计算所有层的梯度，然后一次性更新所有参数
        """
        self.optimizer.zero_grad()
        
        total_loss = 0.0
        
        for layer_idx, inputs, outputs in layer_data_list:
            inputs = inputs.to(self.device).float()
            outputs = outputs.to(self.device).float()
            
            preds = self.predict_model(inputs, layer_idx)
            
            loss = compute_custom_loss(
                preds,
                outputs,
                top_k_routing=self.top_k_routing,
                lambda_rank=self.lambda_rank,
                margin=self.margin
            )
            
            loss.backward()
            total_loss += loss.item()
        
        self.optimizer.step()
        
        return total_loss / len(layer_data_list) if len(layer_data_list) > 0 else 0.0
    
    def train_epoch(self, max_batches: Optional[int] = None, global_sample_count: int = 0, save_dir: str = "./checkpoints"):
        """
        训练一个epoch，每个样本只被训练一次，每50个样本保存一次checkpoint
        """
        self.predict_model.train()
        total_loss = 0.0
        sample_count = 0
        
        print("  Waiting for data from buffer...")
        
        while True:
            if max_batches is not None and sample_count >= max_batches:
                print(f"  Reached max_samples limit: {max_batches}")
                break
            
            batch = self.interface.get_batch()
            
            if batch is None:
                if not self.interface.is_buffer_empty():
                    continue
                print("  Buffer is empty, epoch completed")
                break
            
            for data in batch:
                layer_data_list = self.extract_data(data)
                if len(layer_data_list) > 0:
                    loss = self.train_step(layer_data_list)
                    total_loss += loss
                    sample_count += 1
                    global_sample_count += 1
                    
                    avg_loss = total_loss / sample_count
                    stats = self.interface.get_stats()
                    print(f"  Sample {global_sample_count}, Loss: {loss:.5f}, Avg Loss: {avg_loss:.5f}, "
                          f"Buffer: {stats['used_samples']} samples, "
                          f"{stats['used_memory_gb']:.2f} GB ({stats['utilization']*100:.1f}%)")
                    
                    if global_sample_count % self.checkpoint_interval == 0:
                        checkpoint_path = os.path.join(save_dir, f"predictor_sample_{global_sample_count}.pt")
                        torch.save({
                            'sample': global_sample_count,
                            'model_state_dict': self.predict_model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': loss,
                            'avg_loss': avg_loss
                        }, checkpoint_path)
                        print(f"  Checkpoint saved: {checkpoint_path}")
        
        avg_loss = total_loss / sample_count if sample_count > 0 else 0.0
        return avg_loss, sample_count
    
    def train(self, num_epochs: int = 10, save_dir: str = "./checkpoints", max_samples_per_epoch: Optional[int] = None):
        """
        训练预测器，每50个样本保存一次checkpoint
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Starting training...")
        print(f"Save directory: {save_dir}")
        print(f"Checkpoint interval: every {self.checkpoint_interval} samples")
        print(f"Max samples per epoch: {max_samples_per_epoch}")
        print(f"Total epochs: {num_epochs}")
        print("-" * 60)
        
        total_samples = 0
        global_sample_count = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            avg_loss, sample_count = self.train_epoch(max_batches=max_samples_per_epoch, 
                                                        global_sample_count=global_sample_count,
                                                        save_dir=save_dir)
            
            print(f"\nEpoch {epoch + 1} completed:")
            print(f"  Samples in this epoch: {sample_count}")
            print(f"  Average loss: {avg_loss:.5f}")
            
            total_samples += sample_count
        
        print("\n" + "=" * 60)
        print(f"Training completed!")
        print(f"Total samples trained: {total_samples}")
        print("=" * 60)
        
        final_checkpoint_path = os.path.join(save_dir, "predictor_final.pt")
        torch.save({
            'model_state_dict': self.predict_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_samples': total_samples,
            'total_epochs': num_epochs
        }, final_checkpoint_path)
        print(f"Final checkpoint saved to: {final_checkpoint_path}")
