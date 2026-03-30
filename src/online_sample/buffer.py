import torch
import threading
from typing import Dict, Optional, List, Any
from collections import deque
from .data_structures import ModelConfig, ActivationPattern, ActivationData, BufferStats


class ActivationBuffer:
    def __init__(
        self,
        model_config: ModelConfig,
        pattern: str,
        buffer_size_gb: float = 4.0,
        device: str = "cuda"
    ):
        if not ActivationPattern.validate(pattern):
            raise ValueError(f"Invalid pattern: {pattern}. Must be one of {ActivationPattern.ALL_PATTERNS}")
        
        self.model_config = model_config
        self.pattern = pattern
        self.buffer_size_gb = buffer_size_gb
        self.device = torch.device(device)
        self.dtype = torch.bfloat16
        
        self._lock = threading.Lock()
        self._write_cond = threading.Condition(self._lock)
        self._read_cond = threading.Condition(self._lock)
        
        self._buffer: deque[ActivationData] = deque()
        self._total_memory_bytes = int(buffer_size_gb * 1024**3)
        self._used_memory_bytes = 0
        
        self._is_running = True
        self._write_finished = False
        
        print(f"ActivationBuffer initialized:")
        print(f"  Pattern: {pattern}")
        print(f"  Buffer size: {buffer_size_gb:.2f} GB")
        print(f"  Device: {device}")
        print(f"  Dtype: {self.dtype}")
    
    def write(self, data: ActivationData, timeout: Optional[float] = None) -> bool:
        if not data.validate(self.pattern):
            raise ValueError(f"Data does not match pattern {self.pattern}")
        
        data = data.to(self.device)
        data_size = data.get_memory_size()
        
        with self._write_cond:
            while self._is_running:
                if not self._is_running:
                    return False
                
                if self._used_memory_bytes + data_size <= self._total_memory_bytes:
                    self._buffer.append(data)
                    self._used_memory_bytes += data_size
                    self._read_cond.notify_all()
                    return True
                
                if timeout is not None:
                    if not self._write_cond.wait(timeout):
                        return False
                else:
                    self._write_cond.wait()
            
            return False
    
    def read(self, batch_size: int = 1, timeout: Optional[float] = None) -> Optional[List[ActivationData]]:
        with self._read_cond:
            while self._is_running or len(self._buffer) > 0:
                if len(self._buffer) >= batch_size:
                    batch_data = []
                    freed_memory = 0
                    
                    for _ in range(batch_size):
                        if len(self._buffer) == 0:
                            break
                        data = self._buffer.popleft()
                        batch_data.append(data)
                        freed_memory += data.get_memory_size()
                    
                    self._used_memory_bytes -= freed_memory
                    self._write_cond.notify_all()
                    return batch_data
                
                if self._write_finished and len(self._buffer) < batch_size:
                    if len(self._buffer) > 0:
                        batch_data = []
                        freed_memory = 0
                        
                        while len(self._buffer) > 0:
                            data = self._buffer.popleft()
                            batch_data.append(data)
                            freed_memory += data.get_memory_size()
                        
                        self._used_memory_bytes -= freed_memory
                        self._write_cond.notify_all()
                        return batch_data
                    return None
                
                if timeout is not None:
                    if not self._read_cond.wait(timeout):
                        if len(self._buffer) > 0:
                            batch_data = []
                            freed_memory = 0
                            
                            while len(self._buffer) > 0:
                                data = self._buffer.popleft()
                                batch_data.append(data)
                                freed_memory += data.get_memory_size()
                            
                            self._used_memory_bytes -= freed_memory
                            self._write_cond.notify_all()
                            return batch_data
                        return None
                else:
                    self._read_cond.wait()
            
            return None
    
    def get_size(self) -> int:
        with self._lock:
            return len(self._buffer)
    
    def is_empty(self) -> bool:
        with self._lock:
            return len(self._buffer) == 0
    
    def is_full(self) -> bool:
        with self._lock:
            return self._used_memory_bytes >= self._total_memory_bytes
    
    def get_stats(self) -> BufferStats:
        with self._lock:
            return BufferStats(
                total_samples=len(self._buffer),
                used_samples=len(self._buffer),
                free_samples=0,
                buffer_size_gb=self.buffer_size_gb,
                used_memory_gb=self._used_memory_bytes / (1024**3)
            )
    
    def mark_write_finished(self):
        with self._write_cond:
            self._write_finished = True
            self._read_cond.notify_all()
    
    def stop(self):
        with self._write_cond:
            self._is_running = False
            self._write_cond.notify_all()
        
        with self._read_cond:
            self._read_cond.notify_all()
    
    def clear(self):
        with self._lock:
            self._buffer.clear()
            self._used_memory_bytes = 0
            self._write_cond.notify_all()
            self._read_cond.notify_all()
    
    def __del__(self):
        self.stop()


def create_buffer(
    model_config: ModelConfig,
    pattern: str,
    buffer_size_gb: float = 4.0,
    device: str = "cuda"
) -> ActivationBuffer:
    return ActivationBuffer(
        model_config=model_config,
        pattern=pattern,
        buffer_size_gb=buffer_size_gb,
        device=device
    )
