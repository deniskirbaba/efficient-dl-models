from collections import defaultdict

import pandas as pd
import torch
import triton
import triton.language as tl
from tqdm import tqdm


def add_vec_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x[None, :] + y[:, None]


@triton.jit
def _add_vec_triton(
    x_ptr, y_ptr, res_ptr, x_elements, y_elements, BLOCK_SIZE_X: tl.constexpr, BLOCK_SIZE_Y: tl.constexpr
):
    x_pid = tl.program_id(0)
    y_pid = tl.program_id(1)

    x_offsets = x_pid * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    x_mask = x_offsets < x_elements
    y_offsets = y_pid * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    y_mask = y_offsets < y_elements
    res_offsets = y_offsets[:, None] * x_elements + x_offsets[None, :]
    res_mask = y_mask[:, None] & x_mask[None, :]

    x = tl.load(x_ptr + x_offsets, mask=x_mask)
    y = tl.load(y_ptr + y_offsets, mask=y_mask)
    res = x[None, :] + y[:, None]

    tl.store(res_ptr + res_offsets, res, mask=res_mask)


def add_vec_triton(x: torch.Tensor, y: torch.Tensor, block_size_x: int = 128, block_size_y: int = 128) -> torch.Tensor:
    assert x.device == y.device
    assert x.ndim == y.ndim == 1
    assert x.dtype == y.dtype
    assert x.is_contiguous() and y.is_contiguous()

    result = torch.empty((y.numel(), x.numel()), dtype=x.dtype, device=x.device)

    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE_X"]), triton.cdiv(y.numel(), meta["BLOCK_SIZE_Y"]))
    _add_vec_triton[grid](x, y, result, x.numel(), y.numel(), BLOCK_SIZE_X=block_size_x, BLOCK_SIZE_Y=block_size_y)

    return result


if __name__ == "__main__":
    shapes = [(32, 64), (64, 128), (256, 512), (512, 1024)]
    dtype = torch.bfloat16
    device = torch.device("cuda")

    # precision test
    for shape in tqdm(shapes, desc="Precision tests", leave=False):
        x = torch.randn(shape[0], dtype=dtype, device=device)
        y = torch.randn(shape[1], dtype=dtype, device=device)
        assert torch.allclose(add_vec_torch(x, y), add_vec_triton(x, y))

    # perf test
    stats = defaultdict(dict)
    for shape in tqdm(shapes, desc="Perf tests", leave=False):
        x = torch.randn(shape[0], dtype=dtype, device=device)
        y = torch.randn(shape[1], dtype=dtype, device=device)
        time_torch = triton.testing.do_bench(lambda: add_vec_torch(x, y), warmup=5, rep=25)
        time_triton = triton.testing.do_bench(lambda: add_vec_triton(x, y), warmup=5, rep=25)
        stats["torch"][shape] = time_torch
        stats["triton"][shape] = time_triton
    print(pd.DataFrame(stats).T.to_markdown())
