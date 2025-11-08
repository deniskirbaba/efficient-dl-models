import math
from collections import defaultdict

import pandas as pd
import torch
import triton
import triton.language as tl
from tqdm import tqdm


def softmax_torch(x: torch.Tensor) -> torch.Tensor:
    x_max = x.max(1, keepdim=True)[0]
    x = x - x_max
    x_exp = x.exp()
    return x_exp / x_exp.sum(1, keepdim=True)


@triton.jit
def _softmax_triton(x_ptr, res_ptr, n_batch, n_dim, DIM_BLOCK_SIZE: tl.constexpr, BATCH_BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    x_offs = pid * BATCH_BLOCK_SIZE + tl.arange(0, BATCH_BLOCK_SIZE)
    x_mask = x_offs < n_batch

    run_max = tl.full((BATCH_BLOCK_SIZE,), float("-inf"), dtype=tl.float32)
    run_sum = tl.zeros((BATCH_BLOCK_SIZE,), dtype=tl.float32)

    for start_i in range(0, n_dim, DIM_BLOCK_SIZE):
        # load x
        inner_offs = start_i + tl.arange(0, DIM_BLOCK_SIZE)
        inner_mask = inner_offs < n_dim

        block_offs = x_offs[:, None] * n_dim + inner_offs[None, :]
        block_mask = x_mask[:, None] & inner_mask[None, :]

        x = tl.load(x_ptr + block_offs, mask=block_mask, other=float("-inf"))

        # update run_max and run_sum
        cur_max = x.max(1)
        x -= cur_max[:, None]
        cur_sum = tl.exp2(x * tl.log2(math.e)).sum(1)

        new_max = tl.maximum(run_max, cur_max)
        alpha = tl.exp2((run_max - new_max) * tl.log2(math.e))
        beta = tl.exp2((cur_max - new_max) * tl.log2(math.e))

        run_max = new_max
        run_sum = run_sum * alpha + cur_sum * beta

    for start_i in range(0, n_dim, DIM_BLOCK_SIZE):
        # load x
        inner_offs = start_i + tl.arange(0, DIM_BLOCK_SIZE)
        inner_mask = inner_offs < n_dim

        block_offs = x_offs[:, None] * n_dim + inner_offs[None, :]
        block_mask = x_mask[:, None] & inner_mask[None, :]

        x = tl.load(x_ptr + block_offs, mask=block_mask, other=float("-inf"))

        # calculate softmax
        x -= run_max[:, None]
        block_res = tl.exp2(x * tl.log2(math.e)) / run_sum[:, None]

        tl.store(res_ptr + block_offs, block_res, mask=block_mask)


def softmax_triton(x: torch.Tensor, batch_block_size: int = 64, dim_block_size: int = 128) -> torch.Tensor:
    assert x.is_contiguous()
    assert x.ndim == 2

    result = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(x.shape[0], meta["BATCH_BLOCK_SIZE"]),)
    _softmax_triton[grid](
        x, result, x.shape[0], x.shape[1], DIM_BLOCK_SIZE=dim_block_size, BATCH_BLOCK_SIZE=batch_block_size
    )

    return result


if __name__ == "__main__":
    shapes = [(10, 20), (32, 64), (64, 128), (256, 256), (512, 1024)]
    dtype = torch.float32
    device = torch.device("cuda")

    # precision test
    for shape in tqdm(shapes, desc="Precision tests", leave=False):
        x = torch.randn(shape, dtype=dtype, device=device)
        assert torch.allclose(softmax_torch(x), softmax_triton(x), rtol=1e-4, atol=1e-6)

    # perf test
    stats = defaultdict(dict)
    for shape in tqdm(shapes, desc="Perf tests", leave=False):
        x = torch.randn(shape, dtype=dtype, device=device)
        time_torch = triton.testing.do_bench(lambda: softmax_torch(x), warmup=5, rep=25)
        time_triton = triton.testing.do_bench(lambda: softmax_triton(x), warmup=5, rep=25)
        stats["torch"][shape] = time_torch
        stats["triton"][shape] = time_triton
    print(pd.DataFrame(stats).T.to_markdown())
