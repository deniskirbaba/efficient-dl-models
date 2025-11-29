import torch
import triton
import triton.language as tl


def sum_torch(x: torch.Tensor) -> torch.Tensor:
    return x.sum(1)


# @triton.jit
# def _long_sum_kernel(x_ptr, out_ptr, N0, T, B1: tl.constexpr):
#     # YOUR CODE HERE


# def sum_triton(x: torch.Tensor) -> torch.Tensor:
#     # YOUR CODE HERE


@triton.jit
def _long_sum_kernel(x_ptr, out_ptr, N0, T, B1: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    x_offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x_mask = x_offs < N0

    res = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for start_i in range(0, T, B1):
        inner_offs = start_i + tl.arange(0, B1)
        inner_mask = inner_offs < T

        block_offs = x_offs[:, None] * T + inner_offs[None, :]
        block_mask = x_mask[:, None] & inner_mask[None, :]

        x = tl.load(x_ptr + block_offs, mask=block_mask, other=0.0)
        res += tl.sum(x, 1)

    tl.store(out_ptr + x_offs, res, mask=x_mask)


def sum_triton(x: torch.Tensor, block_size: int = 64, inner_block_size: int = 128) -> torch.Tensor:
    assert x.is_contiguous()
    assert x.ndim == 2

    result = torch.zeros((x.shape[0]), dtype=x.dtype, device=x.device)
    grid = lambda meta: (triton.cdiv(x.shape[0], meta["BLOCK_SIZE"]),)
    _long_sum_kernel[grid](x, result, x.shape[0], x.shape[1], B1=inner_block_size, BLOCK_SIZE=block_size)

    return result
