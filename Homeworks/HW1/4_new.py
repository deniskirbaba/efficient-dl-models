import torch
import triton
import triton.language as tl


@triton.jit
def _outer_add_block_kernel(x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr):
    x_pid = tl.program_id(0)
    y_pid = tl.program_id(1)

    x_offsets = x_pid * B0 + tl.arange(0, B0)
    x_mask = x_offsets < N0
    y_offsets = y_pid * B1 + tl.arange(0, B1)
    y_mask = y_offsets < N1
    res_offsets = y_offsets[:, None] * N0 + x_offsets[None, :]
    res_mask = y_mask[:, None] & x_mask[None, :]

    x = tl.load(x_ptr + x_offsets, mask=x_mask)
    y = tl.load(y_ptr + y_offsets, mask=y_mask)
    res = x[None, :] + y[:, None]

    tl.store(z_ptr + res_offsets, res, mask=res_mask)


def add_vec_block_triton(
    x: torch.Tensor, y: torch.Tensor, block_size_x: int = 128, block_size_y: int = 128
) -> torch.Tensor:
    assert x.device == y.device
    assert x.ndim == y.ndim == 1
    assert x.dtype == y.dtype
    assert x.is_contiguous() and y.is_contiguous()

    result = torch.empty((y.numel(), x.numel()), dtype=x.dtype, device=x.device)

    grid = lambda meta: (triton.cdiv(x.numel(), meta["B0"]), triton.cdiv(y.numel(), meta["B1"]))
    _outer_add_block_kernel[grid](x, y, result, x.numel(), y.numel(), B0=block_size_x, B1=block_size_y)

    return result
