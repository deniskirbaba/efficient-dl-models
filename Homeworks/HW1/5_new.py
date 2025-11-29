import torch
import triton
import triton.language as tl


def mul_relu_block_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.relu(x[None, :] * y[:, None])


@triton.jit
def _mul_relu_block_kernel(x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr):
    x_pid = tl.program_id(0)
    y_pid = tl.program_id(1)

    x_offs = x_pid * B0 + tl.arange(0, B0)
    y_offs = y_pid * B1 + tl.arange(0, B1)
    res_offs = y_offs[:, None] * N0 + x_offs[None, :]

    x_mask = x_offs < N0
    y_mask = y_offs < N1
    res_mask = y_mask[:, None] & x_mask[None, :]

    x = tl.load(x_ptr + x_offs, mask=x_mask)
    y = tl.load(y_ptr + y_offs, mask=y_mask)
    result = tl.maximum(y[:, None] * x[None, :], 0)
    tl.store(z_ptr + res_offs, result, mask=res_mask)


def mul_relu_block_triton(
    x: torch.Tensor, y: torch.Tensor, block_size_x: int = 128, block_size_y: int = 128
) -> torch.Tensor:
    assert x.is_contiguous() and y.is_contiguous()
    assert x.ndim == y.ndim == 1
    assert x.device == y.device
    assert x.dtype == y.dtype

    result = torch.empty((y.numel(), x.numel()), dtype=x.dtype, device=x.device)
    grid = lambda meta: (triton.cdiv(x.numel(), meta["B0"]), triton.cdiv(y.numel(), meta["B1"]))
    _mul_relu_block_kernel[grid](x, y, result, x.numel(), y.numel(), block_size_x, block_size_y)

    return result
