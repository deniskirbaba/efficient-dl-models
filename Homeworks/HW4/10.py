import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_stages=1),
    ],
    key=["n_elements"],
)
@triton.jit
def _quantize_global(x_ptr, absmax_inv_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    absmax_inv = tl.load(absmax_inv_ptr)
    # xq = tl.extra.cuda.libdevice.round(absmax_inv * x * 127.0).to(tl.int8)
    xq = (absmax_inv * x * 127.0).to(tl.int8)
    tl.store(output_ptr + offs, xq, mask=mask)


def quantize_global(x: torch.Tensor):
    absmax = x.abs().max().unsqueeze(0)
    xq = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
    _quantize_global[grid](x, 1 / absmax, xq, x.numel())
    return xq, absmax
