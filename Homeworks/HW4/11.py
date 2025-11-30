import math

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512}, num_stages=2),
    ],
    key=["n_elements"],
)
@triton.jit
def _quantize_rowwise(x_ptr, output_ptr, output_maxs, n_elements, P2: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * n_elements + tl.arange(0, P2)
    mask = offs < (pid + 1) * n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    abs_max = tl.max(tl.abs(x))
    tl.store(output_maxs + pid, abs_max.to(tl.float16))
    xq = x / abs_max * 127.0
    xq = xq + tl.where(xq >= 0, 0.5, -0.5)
    xq = tl.clamp(xq, -127.0, 127.0)
    tl.store(output_ptr + offs, xq.to(tl.int8), mask=mask)


def quantize_rowwise(x: torch.Tensor):
    qx = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    absmaxs = torch.empty(x.shape[0], dtype=torch.float16, device=x.device)
    _quantize_rowwise[(x.shape[0],)](x, qx, absmaxs, x.shape[1], triton.next_power_of_2(x.shape[1]))

    return qx, absmaxs
