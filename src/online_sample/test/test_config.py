import sys
import os
from pathlib import Path

test_dir = Path(__file__).parent
src_dir = test_dir.parent
project_root = src_dir.parent.parent.parent

sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(project_root))

import torch

DEVICE = "cuda"
DTYPE = torch.bfloat16

print(f"Test environment:")
print(f"  Device: {DEVICE}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA device count: {torch.cuda.device_count()}")
    print(f"  Current device: {torch.cuda.current_device()}")
    print(f"  Device name: {torch.cuda.get_device_name(0)}")
print(f"  Test directory: {test_dir}")
print(f"  Source directory: {src_dir}")
print(f"  Project root: {project_root}")

if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
