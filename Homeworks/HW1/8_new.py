# import math

import torch
import triton
import triton.language as tl


def softmax_torch(x: torch.Tensor) -> torch.Tensor:
    x_max = x.max(1, keepdim=True)[0]
    x = x - x_max
    x_exp = x.exp()
    return x_exp / x_exp.sum(1, keepdim=True)


# @triton.jit
# def _long_softmax_kernel(x_ptr, out_ptr, N0, T, B1: tl.constexpr):
#     # YOUR CODE HERE

# def softmax_triton(x: torch.Tensor) -> torch.Tensor:
#     # YOUR CODE HERE


@triton.jit
def _long_softmax_kernel(x_ptr, out_ptr, N0, T, B1: tl.constexpr, BATCH_BLOCK_SIZE: tl.constexpr):
    E = 2.718281828459045

    pid = tl.program_id(0)
    x_offs = pid * BATCH_BLOCK_SIZE + tl.arange(0, BATCH_BLOCK_SIZE)
    x_mask = x_offs < N0

    run_max = tl.full((BATCH_BLOCK_SIZE,), float("-inf"), dtype=tl.float32)
    run_sum = tl.zeros((BATCH_BLOCK_SIZE,), dtype=tl.float32)

    for start_i in range(0, T, B1):
        # load x
        inner_offs = start_i + tl.arange(0, B1)
        inner_mask = inner_offs < T

        block_offs = x_offs[:, None] * T + inner_offs[None, :]
        block_mask = x_mask[:, None] & inner_mask[None, :]

        x = tl.load(x_ptr + block_offs, mask=block_mask, other=float("-inf"))

        # update run_max and run_sum
        cur_max = tl.max(x, 1)
        x -= cur_max[:, None]
        cur_sum = tl.sum(tl.exp2(x * tl.log2(E)), 1)

        new_max = tl.maximum(run_max, cur_max)
        alpha = tl.exp2((run_max - new_max) * tl.log2(E))
        beta = tl.exp2((cur_max - new_max) * tl.log2(E))

        run_max = new_max
        run_sum = run_sum * alpha + cur_sum * beta

    for start_i in range(0, T, B1):
        # load x
        inner_offs = start_i + tl.arange(0, B1)
        inner_mask = inner_offs < T

        block_offs = x_offs[:, None] * T + inner_offs[None, :]
        block_mask = x_mask[:, None] & inner_mask[None, :]

        x = tl.load(x_ptr + block_offs, mask=block_mask, other=float("-inf"))

        # calculate softmax
        x -= run_max[:, None]
        block_res = tl.exp2(x * tl.log2(E)) / run_sum[:, None]

        tl.store(out_ptr + block_offs, block_res, mask=block_mask)


def softmax_triton(x: torch.Tensor, batch_block_size: int = 64, dim_block_size: int = 128) -> torch.Tensor:
    assert x.is_contiguous()
    assert x.ndim == 2

    result = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(x.shape[0], meta["BATCH_BLOCK_SIZE"]),)
    _long_softmax_kernel[grid](x, result, x.shape[0], x.shape[1], B1=dim_block_size, BATCH_BLOCK_SIZE=batch_block_size)

    return result
