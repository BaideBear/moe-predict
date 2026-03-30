from typing import Optional, List, Dict, Any
from .buffer import ActivationBuffer
from .data_structures import ActivationData, ActivationPattern


class PredictorInterface:
    def __init__(
        self,
        buffer: ActivationBuffer,
        pattern: str,
        batch_size: int = 1,
        timeout: Optional[float] = None
    ):
        self.buffer = buffer
        self.pattern = pattern
        self.batch_size = batch_size
        self.timeout = timeout
        
        if not ActivationPattern.validate(pattern):
            raise ValueError(f"Invalid pattern: {pattern}. Must be one of {ActivationPattern.ALL_PATTERNS}")
        
        print(f"PredictorInterface initialized:")
        print(f"  Pattern: {pattern}")
        print(f"  Batch size: {batch_size}")
        print(f"  Timeout: {timeout}")
    
    def get_batch(self) -> Optional[List[ActivationData]]:
        batch = self.buffer.read(batch_size=self.batch_size, timeout=self.timeout)
        
        if batch is None:
            return None
        
        for data in batch:
            if not data.validate(self.pattern):
                raise ValueError(f"Data does not match pattern {self.pattern}")
        
        return batch
    
    def get_stats(self) -> Dict[str, Any]:
        stats = self.buffer.get_stats()
        return {
            'total_samples': stats.total_samples,
            'used_samples': stats.used_samples,
            'free_samples': stats.free_samples,
            'buffer_size_gb': stats.buffer_size_gb,
            'used_memory_gb': stats.used_memory_gb,
            'utilization': stats.used_memory_gb / stats.buffer_size_gb if stats.buffer_size_gb > 0 else 0.0
        }
    
    def is_buffer_empty(self) -> bool:
        return self.buffer.is_empty()
    
    def is_buffer_full(self) -> bool:
        return self.buffer.is_full()
    
    def wait_for_data(self, min_samples: int = 1, timeout: Optional[float] = None) -> bool:
        import time
        start_time = time.time()
        
        while True:
            if self.buffer.get_size() >= min_samples:
                return True
            
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return False
            
            import threading
            threading.Event().wait(0.1)


class PredictorTrainerExample:
    def __init__(
        self,
        buffer: ActivationBuffer,
        pattern: str,
        batch_size: int = 1,
        num_epochs: int = 10,
        learning_rate: float = 1e-4
    ):
        self.interface = PredictorInterface(
            buffer=buffer,
            pattern=pattern,
            batch_size=batch_size
        )
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        
        print(f"PredictorTrainerExample initialized:")
        print(f"  Num epochs: {num_epochs}")
        print(f"  Learning rate: {learning_rate}")
    
    def train(self):
        print("Starting training...")
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            batch_count = 0
            while True:
                batch = self.interface.get_batch()
                
                if batch is None:
                    print("No more data available")
                    break
                
                batch_count += 1
                
                if batch_count % 10 == 0:
                    stats = self.interface.get_stats()
                    print(f"  Batch {batch_count}, Buffer: {stats['used_samples']} samples, "
                          f"{stats['used_memory_gb']:.2f} GB ({stats['utilization']*100:.1f}%)")
                
                for data in batch:
                    self._train_step(data)
            
            print(f"Epoch {epoch + 1} completed, processed {batch_count} batches")
        
        print("\nTraining completed!")
    
    def _train_step(self, data: ActivationData):
        pass


def create_predictor_interface(
    buffer: ActivationBuffer,
    pattern: str,
    batch_size: int = 1,
    timeout: Optional[float] = None
) -> PredictorInterface:
    return PredictorInterface(
        buffer=buffer,
        pattern=pattern,
        batch_size=batch_size,
        timeout=timeout
    )
