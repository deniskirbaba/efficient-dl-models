import torch
import triton
import triton.language as tl


def flashatt_torch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    x = q[:, None] * k[None, :]
    x_max = x.max(1, keepdim=True)[0]
    x = x - x_max
    x_exp = x.exp()
    soft = x_exp / x_exp.sum(1, keepdim=True)
    return (v[None, :] * soft).sum(1)


# @triton.jit
# def _flashatt_kernel(q_ptr, k_ptr, v_ptr, out_ptr, T, B1: tl.constexpr, LOG2E: tl.constexpr):
#     # YOUR CODE HERE

# def flashatt_triton(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
#     # YOUR CODE HERE


@triton.jit
def _flashatt_kernel(q_ptr, k_ptr, v_ptr, out_ptr, T, B1: tl.constexpr, LOG2E: tl.constexpr):
    pid = tl.program_id(0)

    q_offs = pid * B1 + tl.arange(0, B1)
    q_mask = q_offs < T

    q_val = tl.load(q_ptr + q_offs, mask=q_mask, other=0.0)

    m = tl.full((B1,), float("-inf"), dtype=tl.float32)
    l = tl.zeros((B1,), dtype=tl.float32)
    acc = tl.zeros((B1,), dtype=tl.float32)

    for start_i in range(0, T, B1):
        inner_offs = start_i + tl.arange(0, B1)
        inner_mask = inner_offs < T

        k_val = tl.load(k_ptr + inner_offs, mask=inner_mask, other=0.0)
        v_val = tl.load(v_ptr + inner_offs, mask=inner_mask, other=0.0)

        s_j = q_val[:, None] * k_val[None, :]
        s_j = tl.where(inner_mask[None, :], s_j, float("-inf"))

        block_max = tl.max(s_j, 1)
        new_m = tl.maximum(m, block_max)

        alpha = tl.exp2((m - new_m) * LOG2E)
        beta = tl.exp2((block_max - new_m) * LOG2E)

        p_block = tl.exp2((s_j - block_max[:, None]) * LOG2E)
        block_acc = tl.sum(p_block * v_val[None, :], 1)

        block_l = tl.sum(p_block, 1)

        acc = acc * alpha + block_acc * beta
        l = l * alpha + block_l * beta
        m = new_m

    res = acc / l
    tl.store(out_ptr + q_offs, res, mask=q_mask)


def flashatt_triton(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert q.shape == k.shape == v.shape
    assert q.dtype == k.dtype == v.dtype
    assert q.device == k.device == v.device

    result = torch.empty_like(q)
    grid = lambda meta: (triton.cdiv(q.numel(), meta["B1"]),)
    _flashatt_kernel[grid](q, k, v, result, q.numel(), B1=block_size, LOG2E=1.4426950408889634)

    return result
