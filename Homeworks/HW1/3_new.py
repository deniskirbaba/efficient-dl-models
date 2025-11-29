import torch
import triton
import triton.language as tl


def add_vec_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x[None, :] + y[:, None]


@triton.jit
def _outer_add_kernel(x_ptr, y_ptr, z_ptr, N0: tl.constexpr, N1: tl.constexpr):
    x_offsets = tl.arange(0, N0)
    y_offsets = tl.arange(0, N1)
    x = tl.load(x_ptr + x_offsets)
    y = tl.load(y_ptr + y_offsets)

    res_offsets = y_offsets[:, None] * N0 + x_offsets[None, :]
    # res_offsets = tl.arange(0, BLOCK_SIZE_Y * BLOCK_SIZE_X).reshape(BLOCK_SIZE_Y, BLOCK_SIZE_X)  # more longer

    res = x[None, :] + y[:, None]
    tl.store(z_ptr + res_offsets, res)


def outer_vector_add_triton(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.device == y.device
    assert x.ndim == y.ndim == 1
    assert x.dtype == y.dtype
    assert x.is_contiguous() and y.is_contiguous()

    result = torch.empty((y.numel(), x.numel()), dtype=x.dtype, device=x.device)

    grid = lambda meta: (1, 1)
    _outer_add_kernel[grid](x, y, result, N0=x.numel(), N1=y.numel())

    return result
