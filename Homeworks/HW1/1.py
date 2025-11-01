import torch
import triton

def add_torch(x: torch.tensor, const_val: float) -> torch.tensor:
    return x + const_val

def add_triton()