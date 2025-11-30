# Задача: INT8 MatMul with Fused Dequant (Triton)
# (дедлайн МСК: 06.12.2025 23:59 MSK)

# Задача: INT8 матмул с фьюзом деквантизации и биаса

# Даны тензоры:
# - X_q формы (B, IN) — int8 входные активации (row-major, contiguous);
# - W_q формы (IN, OUT) — int8 веса (row-major, contiguous);
# - s_x — fp32 скаляр масштаба для X_q (per-tensor);
# - s_w — fp32 масштабы для W_q: либо скаляр (per-tensor), либо вектор формы (OUT,) для per-OUT-channel;
# - bias — опционально fp16 вектор формы (OUT,).

# Нужно вычислить выход Y формы (B, OUT) в fp16.
# Математика (словами):
# 1) накапливайте произведения int8*int8 в int32: ACC[b, o] = sum_k X_q[b, k] * W_q[k, o];
# 2) вычислите масштаб на столбец o: alpha[o] = s_x * (s_w[o] при per-channel, иначе s_w);
# 3) примените масштаб после окончания цикла по k: Y[b, o] = fp16( fp32(ACC[b, o]) * alpha[o] + bias[o] (если задан) ).

# Требования:
# - Реализовать Triton-ядро, которое загружает тайлы int8 из X_q и W_q, аккумулирует в int32, применяет масштабы один раз после K-цикла (fused dequant), добавляет bias внутри ядра (если он есть) и пишет Y как fp16 (row-major).
# - Поддержать per-channel и per-tensor варианты масштабов весов (constexpr-флаг PER_CHANNEL).
# - Обязательно маскировать хвосты по всем измерениям.
# - Использовать tl.load и tl.store.
# - Обязательно использовать @triton.autotune с не менее чем 3 различными конфигурациями (BLOCK_M/N/K, num_warps, num_stages).


import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        # triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128}, num_warps=4, num_stages=2),
        # triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        # triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=4, num_stages=2),
        # triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        # triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 128}, num_warps=4, num_stages=2),
        # triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        # triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=4, num_stages=2),
        # triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=2),
        # triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128}, num_warps=8, num_stages=2),
        # triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=2),
        # triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=2),
        # triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=2),
        # triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 128}, num_warps=8, num_stages=2),
        # triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=2),
        # triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=2),
        # triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        # triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128}, num_warps=4, num_stages=3),
        # triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        # triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=4, num_stages=3),
        # triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        # triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 128}, num_warps=4, num_stages=3),
        # triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        # triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=4, num_stages=3),
        # triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        # triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128}, num_warps=8, num_stages=3),
        # triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        # triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=3),
        # triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        # triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 128}, num_warps=8, num_stages=3),
        # triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        # triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=3),
    ],
    key=["B", "IN", "OUT"],
)
@triton.jit
def _forward_int8_fused_kernel(
    x_q_ptr,
    x_scale_ptr,
    w_q_ptr,
    w_scale_ptr,
    b_ptr,
    y_ptr,
    B,
    IN,
    OUT,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PER_CHANNEL: tl.constexpr,
):
    m_pid = tl.program_id(0)
    n_pid = tl.program_id(1)
    m_offs = m_pid * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = n_pid * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_offs < B
    n_mask = n_offs < OUT

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for k_start_i in range(0, IN, BLOCK_K):
        k_offs = k_start_i + tl.arange(0, BLOCK_K)
        k_mask = k_offs < IN

        x_q_offs = m_offs[:, None] * IN + k_offs[None, :]
        x_q_mask = m_mask[:, None] & k_mask[None, :]
        x_q = tl.load(x_q_ptr + x_q_offs, mask=x_q_mask, other=0)

        w_q_offs = k_offs[:, None] * OUT + n_offs[None, :]
        w_q_mask = k_mask[:, None] & n_mask[None, :]
        w_q = tl.load(w_q_ptr + w_q_offs, mask=w_q_mask, other=0)

        acc += tl.dot(x_q, w_q, out_dtype=tl.int32)

    acc = acc.to(tl.float32)
    x_scale = tl.load(x_scale_ptr)
    w_scale = tl.load(w_scale_ptr)
    acc = acc / x_scale / w_scale

    acc = tl.cast(acc, dtype=tl.float16, fp_downcast_rounding="rtne")

    y_offs = m_offs[:, None] * OUT + n_offs[None, :]
    y_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(y_ptr + y_offs, acc, mask=y_mask)


def matmul_int8_fused(
    x_q: torch.Tensor,
    x_scale: torch.Tensor,
    w_q: torch.Tensor,
    w_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    per_channel: bool = True
) -> torch.Tensor:
    """Вернуть Y = dequant(X_q) @ dequant(W_q) + bias, dtype fp16, shape (B, OUT)."""
    m, n, k = x_q.shape[0], w_q.shape[1], x_q.shape[1]
    y = torch.empty((m, n), dtype=torch.float16, device=x_q.device)
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]), triton.cdiv(n, meta["BLOCK_N"]))
    _forward_int8_fused_kernel[grid](x_q, x_scale, w_q, w_scale, bias, y, m, k, n, PER_CHANNEL=per_channel)
    return y
