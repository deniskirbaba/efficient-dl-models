from collections import defaultdict

import pandas as pd
import torch
import triton
import triton.language as tl
from tqdm import tqdm


def sum_torch(x: torch.Tensor) -> torch.Tensor:
    return x.sum(1)


@triton.jit
def _sum_triton(x_ptr, res_ptr, n_batches, n_dim, INNER_BLOCK_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    x_offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x_mask = x_offs < n_batches

    res = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for start_i in range(0, n_dim, INNER_BLOCK_SIZE):
        inner_offs = start_i + tl.arange(0, INNER_BLOCK_SIZE)
        inner_mask = inner_offs < n_dim

        block_offs = x_offs[:, None] * n_dim + inner_offs[None, :]
        block_mask = x_mask[:, None] & inner_mask[None, :]

        x = tl.load(x_ptr + block_offs, mask=block_mask, other=0.0)
        res += x.sum(1)

    tl.store(res_ptr + x_offs, res, mask=x_mask)


def sum_triton(x: torch.Tensor, block_size: int = 64, inner_block_size: int = 128) -> torch.Tensor:
    assert x.is_contiguous()
    assert x.ndim == 2

    result = torch.zeros((x.shape[0]), dtype=x.dtype, device=x.device)
    grid = lambda meta: (triton.cdiv(x.shape[0], meta["BLOCK_SIZE"]),)
    _sum_triton[grid](x, result, x.shape[0], x.shape[1], INNER_BLOCK_SIZE=inner_block_size, BLOCK_SIZE=block_size)

    return result


if __name__ == "__main__":
    shapes = [(10, 20), (32, 64), (64, 128), (256, 256)]
    dtype = torch.float32
    device = torch.device("cuda")

    # precision test
    for shape in tqdm(shapes, desc="Precision tests", leave=False):
        x = torch.randn(shape, dtype=dtype, device=device)
        assert torch.allclose(sum_torch(x), sum_triton(x))

    # perf test
    stats = defaultdict(dict)
    for shape in tqdm(shapes, desc="Perf tests", leave=False):
        x = torch.randn(shape, dtype=dtype, device=device)
        time_torch = triton.testing.do_bench(lambda: sum_torch(x), warmup=5, rep=25)
        time_triton = triton.testing.do_bench(lambda: sum_triton(x), warmup=5, rep=25)
        stats["torch"][shape] = time_torch
        stats["triton"][shape] = time_triton
    print(pd.DataFrame(stats).T.to_markdown())
