# Задача: INT8 Backward dX with Fused Dequant (Triton)
# (дедлайн МСК: 06.12.2025 23:59 MSK)

# Задача: беквард по входу с фьюзом деквантизации весов (INT8)

# Дано:
# - dY формы (B, OUT) — fp16 (row-major, contiguous) — апстрим-градиент;
# - W_q формы (IN, OUT) — int8 (row-major, contiguous) — квантованные веса;
# - s_w — fp32 масштабы для W_q: либо скаляр (per-tensor), либо вектор формы (OUT,) (per-OUT-channel).

# Нужно вычислить dX формы (B, IN) в fp16:
# - накапливайте dY @ (W_q_deq)^T в fp32, где W_q_deq = W_q * s_w (пер-столбец или скаляр);
# - применяйте масштабы один раз после цикла по K (фьюз деквантизации на выходе тайла);
# - запишите результат как fp16.

# Математика (словами):
# - W_q_deq[:, o] = W_q[:, o] * s_w[o] при per-channel, иначе * s_w_scalar;
# - dX[b, i] = sum_k dY[b, k] * W_q_deq[i, k];
# - хранение и доступ — row-major.

# Требования:
# - Реализовать Triton-ядро, которое грузит тайлы dY (fp16) и W_q (int8), аккумулирует в fp32, умножает на s_w (вектор по OUT или скаляр) внутри K-цикла или сразу после него один раз (разрешено умножать загружаемый тайл весов; важно, чтобы масштабы не применялись к каждому элементу повторно после суммирования);
# - Маскировать хвосты по всем измерениям;
# - Использовать tl.load и tl.store;
# - Обязательно использовать @triton.autotune с не менее чем 3 конфигурациями (BLOCK_M/N/K, num_warps, num_stages);
# - Поддержать режимы per_channel=True/False (constexpr-флаг PER_CHANNEL).

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128}, num_warps=4, num_stages=3),
    ],
    key=["B", "IN", "OUT"],
)
@triton.jit
def _backward_dx_fused_kernel(
    dy_ptr,
    wq_ptr,
    w_scale_ptr,
    dx_ptr,
    B,
    IN,
    OUT,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PER_CHANNEL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < B
    mask_n = offs_n < IN

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, OUT, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < OUT

        dy_offs = offs_m[:, None] * OUT + offs_k[None, :]
        dy_mask = mask_m[:, None] & mask_k[None, :]
        dy_tile = tl.load(dy_ptr + dy_offs, mask=dy_mask, other=0.0)

        wq_offs = offs_n[:, None] * OUT + offs_k[None, :]
        wq_mask = mask_n[:, None] & mask_k[None, :]
        wq_tile = tl.load(wq_ptr + wq_offs, mask=wq_mask, other=0.0)

        if PER_CHANNEL:
            ws_mask = mask_k
            w_scale = tl.load(w_scale_ptr + offs_k, mask=ws_mask, other=1.0)
            w_scale = w_scale[None, :]
        else:
            w_scale = tl.load(w_scale_ptr)

        wq_deq = wq_tile.to(tl.float16) * w_scale.to(tl.float16)

        acc = tl.dot(dy_tile, tl.trans(wq_deq), acc)

    dx_tile = acc.to(tl.float16)

    dx_offs = offs_m[:, None] * IN + offs_n[None, :]
    dx_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(dx_ptr + dx_offs, dx_tile, mask=dx_mask)


def backward_dx_int8_fused(
    dy: torch.Tensor, w_q: torch.Tensor, w_scale: torch.Tensor, *, per_channel: bool = True
) -> torch.Tensor:
    B, OUT = dy.shape
    IN, OUT_W = w_q.shape

    dx = torch.empty((B, IN), dtype=torch.float16, device=dy.device)

    grid = lambda meta: (triton.cdiv(B, meta["BLOCK_M"]), triton.cdiv(IN, meta["BLOCK_N"]))

    _backward_dx_fused_kernel[grid](
        dy_ptr=dy, wq_ptr=w_q, w_scale_ptr=w_scale, dx_ptr=dx, B=B, IN=IN, OUT=OUT, PER_CHANNEL=per_channel
    )

    return dx
