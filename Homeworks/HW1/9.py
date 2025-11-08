import math
from collections import defaultdict

import pandas as pd
import torch
import triton
import triton.language as tl
from tqdm import tqdm


def flashatt_torch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    x = q[:, None] * k[None, :]
    x_max = x.max(1, keepdim=True)[0]
    x = x - x_max
    x_exp = x.exp()
    soft = x_exp / x_exp.sum(1, keepdim=True)
    return (v[None, :] * soft).sum(1)


@triton.jit
def _flashatt_triton(q_ptr, k_ptr, v_ptr, res_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    q_offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    q_mask = q_offs < n_elements

    q_val = tl.load(q_ptr + q_offs, mask=q_mask, other=0.0)

    m = tl.full((BLOCK_SIZE,), float("-inf"), dtype=tl.float32)
    l = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for start_i in range(0, n_elements, BLOCK_SIZE):
        inner_offs = start_i + tl.arange(0, BLOCK_SIZE)
        inner_mask = inner_offs < n_elements

        k_val = tl.load(k_ptr + inner_offs, mask=inner_mask, other=0.0)
        v_val = tl.load(v_ptr + inner_offs, mask=inner_mask, other=0.0)

        s_j = q_val[:, None] * k_val[None, :]
        s_j = tl.where(inner_mask[None, :], s_j, float("-inf"))

        block_max = tl.max(s_j, 1)
        new_m = tl.maximum(m, block_max)

        alpha = tl.exp2((m - new_m) * tl.log2(math.e))
        beta = tl.exp2((block_max - new_m) * tl.log2(math.e))

        p_block = tl.exp2((s_j - block_max[:, None]) * tl.log2(math.e))
        block_acc = tl.sum(p_block * v_val[None, :], 1)

        block_l = tl.sum(p_block, 1)

        acc = acc * alpha + block_acc * beta
        l = l * alpha + block_l * beta
        m = new_m

    res = acc / l
    tl.store(res_ptr + q_offs, res, mask=q_mask)


def flashatt_triton(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert q.shape == k.shape == v.shape
    assert q.dtype == k.dtype == v.dtype
    assert q.device == k.device == v.device

    result = torch.empty_like(q)
    grid = lambda meta: (triton.cdiv(q.numel(), meta["BLOCK_SIZE"]),)
    _flashatt_triton[grid](q, k, v, result, q.numel(), BLOCK_SIZE=block_size)

    return result


if __name__ == "__main__":
    shapes = [32, 64, 128, 256, 512, 1024, 2048]
    dtype = torch.float32
    device = torch.device("cuda")

    # precision test
    for shape in tqdm(shapes, desc="Precision tests", leave=False):
        q = torch.randn(shape, dtype=dtype, device=device)
        k = torch.randn(shape, dtype=dtype, device=device)
        v = torch.randn(shape, dtype=dtype, device=device)
        assert torch.allclose(flashatt_torch(q, k, v), flashatt_triton(q, k, v), rtol=1e-4, atol=1e-5)

    # perf test
    stats = defaultdict(dict)
    for shape in tqdm(shapes, desc="Perf tests", leave=False):
        q = torch.randn(shape, dtype=dtype, device=device)
        k = torch.randn(shape, dtype=dtype, device=device)
        v = torch.randn(shape, dtype=dtype, device=device)
        time_torch = triton.testing.do_bench(lambda: flashatt_torch(q, k, v), warmup=5, rep=25)
        time_triton = triton.testing.do_bench(lambda: flashatt_triton(q, k, v), warmup=5, rep=25)
        stats["torch"][shape] = time_torch
        stats["triton"][shape] = time_triton
    print(pd.DataFrame(stats).T.to_markdown())
