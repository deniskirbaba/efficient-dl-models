from collections import defaultdict

import pandas as pd
import torch
import triton
import triton.language as tl
from tqdm import tqdm


def mul_relu_block_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.relu(x[None, :] * y[:, None])


@triton.jit
def _mul_relu_block_triton(
    x_ptr, y_ptr, res_ptr, x_elements, y_elements, BLOCK_SIZE_X: tl.constexpr, BLOCK_SIZE_Y: tl.constexpr
):
    x_pid = tl.program_id(0)
    y_pid = tl.program_id(1)

    x_offs = x_pid * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    y_offs = y_pid * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    res_offs = y_offs[:, None] * x_elements + x_offs[None, :]

    x_mask = x_offs < x_elements
    y_mask = y_offs < y_elements
    res_mask = y_mask[:, None] & x_mask[None, :]

    x = tl.load(x_ptr + x_offs, mask=x_mask)
    y = tl.load(y_ptr + y_offs, mask=y_mask)
    result = tl.maximum(y[:, None] * x[None, :], 0)
    tl.store(res_ptr + res_offs, result, mask=res_mask)


def mul_relu_block_triton(
    x: torch.Tensor, y: torch.Tensor, block_size_x: int = 128, block_size_y: int = 128
) -> torch.Tensor:
    assert x.is_contiguous() and y.is_contiguous()
    assert x.ndim == y.ndim == 1
    assert x.device == y.device
    assert x.dtype == y.dtype

    result = torch.empty((y.numel(), x.numel()), dtype=x.dtype, device=x.device)
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE_X"]), triton.cdiv(y.numel(), meta["BLOCK_SIZE_Y"]))
    _mul_relu_block_triton[grid](x, y, result, x.numel(), y.numel(), block_size_x, block_size_y)

    return result


if __name__ == "__main__":
    shapes = [(32, 64), (64, 128), (256, 512), (512, 1024)]
    dtype = torch.bfloat16
    device = torch.device("cuda")

    # precision test
    for shape in tqdm(shapes, desc="Precision tests", leave=False):
        x = torch.randn(shape[0], dtype=dtype, device=device)
        y = torch.randn(shape[1], dtype=dtype, device=device)
        assert torch.allclose(mul_relu_block_torch(x, y), mul_relu_block_triton(x, y))

    # perf test
    stats = defaultdict(dict)
    for shape in tqdm(shapes, desc="Perf tests", leave=False):
        x = torch.randn(shape[0], dtype=dtype, device=device)
        y = torch.randn(shape[1], dtype=dtype, device=device)
        time_torch = triton.testing.do_bench(lambda: mul_relu_block_torch(x, y), warmup=5, rep=25)
        time_triton = triton.testing.do_bench(lambda: mul_relu_block_triton(x, y), warmup=5, rep=25)
        stats["torch"][shape] = time_torch
        stats["triton"][shape] = time_triton
    print(pd.DataFrame(stats).T.to_markdown())
