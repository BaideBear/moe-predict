import torch
import torch.optim as optim
import torch.nn.functional as F
from typing import Optional, Dict, Any
from tqdm import tqdm
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from online_sample import ActivationBuffer, create_predictor_interface
from .model import PredictorModel
from .loss import compute_ce_loss


class PredictorTrainer:
    """PROBE 在线蒸馏训练器，支持 tqdm 进度条"""

    def __init__(
        self,
        buffer: ActivationBuffer,
        predict_model: PredictorModel,
        pattern: str = "attn_gate",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        top_k_routing: int = 8,
        checkpoint_interval: int = 50,
        device: str = "cuda"
    ):
        self.buffer = buffer
        self.predict_model = predict_model
        self.pattern = pattern
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.top_k_routing = top_k_routing
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

        print(f"PROBE PredictorTrainer initialized:")
        print(f"  Pattern: {pattern}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Checkpoint interval: {checkpoint_interval} samples")
        print(f"  Device: {device}")

    def extract_data(self, data):
        """从 ActivationData 中提取输入输出对"""
        if self.pattern == "attn_gate":
            attn_hidden_states = data.attn_hidden_states
            gate_logits = data.gate_logits
            seq_lengths = data.seq_lengths

            num_samples, num_layers, max_seq_len, hidden_dim = attn_hidden_states.shape

            layer_data_list = []

            for sample_idx in range(num_samples):
                seq_len = seq_lengths[sample_idx].item()
                for layer_idx in range(num_layers):
                    layer_input = attn_hidden_states[sample_idx, layer_idx, :seq_len, :]
                    layer_target = gate_logits[sample_idx, layer_idx, :seq_len, :]
                    layer_data_list.append((layer_idx, layer_input, layer_target))

            return layer_data_list
        else:
            raise ValueError(f"Unsupported pattern: {self.pattern}")

    def train_step(self, layer_data_list):
        """训练一个样本的所有层，并计算实时准确率"""
        self.optimizer.zero_grad()
        total_loss = 0.0
        total_tokens = 0
        total_hits = 0

        for layer_idx, inputs, targets in layer_data_list:
            inputs = inputs.to(self.device).float()
            targets = targets.to(self.device).float()

            num_tokens = inputs.shape[0]
            total_tokens += num_tokens

            preds = self.predict_model(inputs, layer_idx)

            loss = compute_ce_loss(preds, targets)
            loss.backward()
            total_loss += loss.item()

            with torch.no_grad():
                _, topk_real = torch.topk(targets, k=self.top_k_routing, dim=-1)
                _, topk_pred = torch.topk(preds, k=self.top_k_routing, dim=-1)
                for i in range(num_tokens):
                    real_set = set(topk_real[i].tolist())
                    pred_set = set(topk_pred[i].tolist())
                    total_hits += len(real_set.intersection(pred_set))

        self.optimizer.step()

        avg_loss = total_loss / len(layer_data_list) if len(layer_data_list) > 0 else 0.0
        hit_rate = total_hits / (total_tokens * self.top_k_routing) if total_tokens > 0 else 0.0

        return avg_loss, hit_rate, total_tokens

    def train_epoch(self, max_batches: Optional[int] = None,
                    global_sample_count: int = 0, save_dir: str = "./checkpoints"):
        self.predict_model.train()
        total_loss = 0.0
        sample_count = 0

        print("  Waiting for data from buffer...")

        pbar = tqdm(desc="Training Samples", unit="sample")

        while True:
            if max_batches is not None and sample_count >= max_batches:
                pbar.close()
                break

            batch = self.interface.get_batch()

            if batch is None:
                if not self.interface.is_buffer_empty():
                    continue
                pbar.close()
                break

            for data in batch:
                layer_data_list = self.extract_data(data)
                if len(layer_data_list) > 0:
                    start_time = time.time()

                    loss, hit_rate, tokens_processed = self.train_step(layer_data_list)

                    duration = time.time() - start_time
                    throughput = tokens_processed / duration if duration > 0 else 0

                    total_loss += loss
                    sample_count += 1
                    global_sample_count += 1

                    avg_loss = total_loss / sample_count

                    # 更新 tqdm 状态栏
                    stats = self.interface.get_stats()
                    pbar.set_postfix({
                        "Loss": f"{loss:.4f}",
                        "HitRate": f"{hit_rate:.4f}",
                        "Tput": f"{throughput:.1f}",
                        "Buf": f"{stats['used_samples']}"
                    })
                    pbar.update(1)

                    if global_sample_count % self.checkpoint_interval == 0:
                        checkpoint_path = os.path.join(save_dir, f"predictor_sample_{global_sample_count}.pt")
                        torch.save({
                            'sample': global_sample_count,
                            'model_state_dict': self.predict_model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': loss,
                            'avg_loss': avg_loss
                        }, checkpoint_path)

        avg_loss = total_loss / sample_count if sample_count > 0 else 0.0
        return avg_loss, sample_count

    def train(self, num_epochs: int = 10, save_dir: str = "./checkpoints",
              max_samples_per_epoch: Optional[int] = None):
        os.makedirs(save_dir, exist_ok=True)
        print(f"Starting PROBE training...")

        total_samples = 0
        global_sample_count = 0

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            avg_loss, sample_count = self.train_epoch(
                max_batches=max_samples_per_epoch,
                global_sample_count=global_sample_count,
                save_dir=save_dir
            )
            total_samples += sample_count
            global_sample_count += sample_count

        final_checkpoint_path = os.path.join(save_dir, "predictor_final.pt")
        torch.save({
            'model_state_dict': self.predict_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_samples': total_samples,
            'total_epochs': num_epochs
        }, final_checkpoint_path)
        print(f"Final checkpoint saved to: {final_checkpoint_path}")
