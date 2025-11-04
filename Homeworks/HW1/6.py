from collections import defaultdict

import pandas as pd
import torch
import triton
import triton.language as tl
from tqdm import tqdm


def mul_relu_block_back_torch(x: torch.Tensor, y: torch.Tensor, dz: torch.Tensor) -> torch.Tensor:
    x = x.clone().requires_grad_(True)
    y = y.clone().requires_grad_(True)
    z = torch.relu(x * y[:, None])
    z.backward(dz)
    dx = x.grad
    return dx


@triton.jit
def _mul_relu_block_back_triton(
    x_ptr, y_ptr, dz_ptr, res_ptr, x_elements, y_elements, BLOCK_SIZE_X: tl.constexpr, BLOCK_SIZE_Y: tl.constexpr
):
    x_pid = tl.program_id(0)
    y_pid = tl.program_id(1)

    x_offs = x_pid * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)  # also for result
    y_offs = y_pid * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    dz_offs = y_offs[:, None] * x_elements + x_offs[None, :]  # also for all matrices (m, n)

    x_mask = x_offs < x_elements
    y_mask = y_offs < y_elements
    dz_mask = y_mask[:, None] & x_mask[None, :]

    x = tl.load(x_ptr + x_offs, mask=x_mask)
    y = tl.load(y_ptr + y_offs, mask=y_mask)
    dz = tl.load(dz_ptr + dz_offs, mask=dz_mask)

    dz *= (y[:, None] * x[None, :]) > 0
    dz *= y[:, None]
    result = dz.sum(0)

    tl.atomic_add(res_ptr + x_offs, result, mask=x_mask)


def mul_relu_block_back_triton(
    x: torch.Tensor, y: torch.Tensor, dz: torch.Tensor, block_size_x: int = 128, block_size_y: int = 128
) -> torch.Tensor:
    """x: (n,); y: (m,); dz: (m, n)"""
    assert x.is_contiguous() and y.is_contiguous() and dz.is_contiguous()
    assert x.device == y.device == dz.device
    assert x.dtype == y.dtype == dz.dtype
    assert x.ndim == y.ndim == 1
    assert dz.shape == (y.numel(), x.numel())

    result = torch.zeros_like(x)
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE_X"]), triton.cdiv(y.numel(), meta["BLOCK_SIZE_Y"]))
    _mul_relu_block_back_triton[grid](x, y, dz, result, x.numel(), y.numel(), block_size_x, block_size_y)

    return result


if __name__ == "__main__":
    shapes = [(10, 20), (32, 64), (64, 128), (256, 512), (512, 1024)]
    dtype = torch.float32
    device = torch.device("cuda")

    # precision test
    for shape in tqdm(shapes, desc="Precision tests", leave=False):
        x = torch.randn(shape[1], dtype=dtype, device=device)
        y = torch.randn(shape[0], dtype=dtype, device=device)
        dz = torch.randn(shape, dtype=dtype, device=device)
        assert torch.allclose(
            mul_relu_block_back_torch(x, y, dz), mul_relu_block_back_triton(x, y, dz), rtol=1e-4, atol=1e-5
        )

    # perf test
    stats = defaultdict(dict)
    for shape in tqdm(shapes, desc="Perf tests", leave=False):
        x = torch.randn(shape[1], dtype=dtype, device=device)
        y = torch.randn(shape[0], dtype=dtype, device=device)
        dz = torch.randn(shape, dtype=dtype, device=device)
        time_torch = triton.testing.do_bench(lambda: mul_relu_block_back_torch(x, y, dz), warmup=5, rep=25)
        time_triton = triton.testing.do_bench(lambda: mul_relu_block_back_triton(x, y, dz), warmup=5, rep=25)
        stats["torch"][shape] = time_torch
        stats["triton"][shape] = time_triton
    print(pd.DataFrame(stats).T.to_markdown())
