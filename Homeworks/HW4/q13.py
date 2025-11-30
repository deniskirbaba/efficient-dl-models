# Задача: Global INT8 Quantize + Transpose (Triton)
# (дедлайн МСК: 06.12.2025 23:59 MSK)

# Задача: глобальная симметричная квантизация в int8 с транспозированием

# Дан 2D тензор X формы (M, N) (fp16/fp32, row-major, contiguous). Требуется:
# 1) вычислить absmax = max(|X|) (скаляр fp32);
# 2) квантизовать элементы по формуле Q = round(127 * X / denom), где denom = max(absmax, 127*1e-8);
# 3) записать результат в выходной тензор B формы (N, M) в int8 (то есть сразу хранить Q^T);
# 4) вернуть (B, absmax[1]) — обратите внимание: возвращаем именно absmax без клэмпа, a denom используется только во избежание деления на 0.

# Требования:
# - Реализовать одно Triton-ядро, которое загружает тайлы из A (X), квантизует и пишет в B (транспонируя индексы);
# - Использовать tl.load / tl.store и маски по краям;
# - Обязательно использовать @triton.autotune с не менее чем 3 конфигурациями (BLOCK_M/N, GROUP_M, num_warps/num_stages);
# - Поддержать произвольные (но совместимые) страйды: у входа хотя бы одно измерение unit-stride, у выхода тоже хотя бы одно измерение unit-stride;
# - Python-обёртка quantize_global_transpose(x) возвращает (q_T:int8[N,M], absmax:fp32[1]).

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "GROUP_M": 8}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "GROUP_M": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "GROUP_M": 8}, num_warps=4, num_stages=2),
    ],
    key=["M", "N"],
)
@triton.jit
def _quantize_global_transpose(
    A,
    absmax_inv_ptr,
    B,
    stride_am,
    stride_an,
    stride_bn,
    stride_bm,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    m_pid = tl.program_id(0)
    n_pid = tl.program_id(1)

    m_offs = m_pid * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = n_pid * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_offs < M
    n_mask = n_offs < N

    a_offs = m_offs[:, None] * stride_am + n_offs[None, :] * stride_an
    a_mask = m_mask[:, None] & n_mask[None, :]

    absmax_inv = tl.load(absmax_inv_ptr)
    a = tl.load(A + a_offs, mask=a_mask)
    b = 127 * absmax_inv * a
    b = tl.clamp(b + tl.where(b >= 0, 0.5, -0.5), -127.0, 127.0).to(tl.int8)

    b_offs = m_offs[:, None] * stride_bm + n_offs[None, :] * stride_bn
    tl.store(B + b_offs, b, mask=a_mask)


def quantize_global_transpose(x: torch.Tensor):
    """Return (q_T:int8[N,M], absmax:fp32[1])."""
    m, n = x.shape
    absmax = x.abs().max().unsqueeze(0).to(torch.float32)
    inv_absmax = torch.tensor([1.0 / max(absmax.item(), 127 * 1e-8)]).to(device=x.device)
    # print(inv_absmax)
    q_T = torch.empty((n, m), dtype=torch.int8, device=x.device)
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]), triton.cdiv(n, meta["BLOCK_N"]))
    _quantize_global_transpose[grid](x, inv_absmax, q_T, n, 1, m, 1, m, n)
    return q_T, absmax
