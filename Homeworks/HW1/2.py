# from collections import defaultdict

# import pandas as pd
import torch
import triton
import triton.language as tl

# from tqdm import tqdm


def add_torch(x: torch.Tensor, const_val: float) -> torch.Tensor:
    return x + const_val


@triton.jit
def _add_triton(x_ptr, const_val, res_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    x = tl.load(x_ptr + offsets, mask=mask) + const_val
    tl.store(res_ptr + offsets, x, mask=mask)


def add_triton(x: torch.Tensor, const_val: float, block_size: int = 64) -> torch.Tensor:
    assert x.is_contiguous()
    result: torch.tensor = torch.empty_like(x)

    grid = lambda meta: (
        triton.cdiv(
            x.numel(),
            meta["BLOCK_SIZE"],
        ),
    )
    _add_triton[grid](x, const_val, result, x.numel(), BLOCK_SIZE=block_size)

    return result


# if __name__ == "__main__":
#     shapes = [(64,), (64, 64), (256,), (256, 256), (1024,), (1024, 1024)]
#     dtype = torch.bfloat16
#     device = torch.device("cuda")

#     # precision test
#     for shape in tqdm(shapes, desc="Precision tests", leave=False):
#         x = torch.randn(*shape, dtype=dtype, device=device)
#         assert torch.allclose(add_torch(x, 1.0), add_triton(x, 1.0))

#     # perf test
#     stats = defaultdict(dict)
#     for shape in tqdm(shapes, desc="Perf tests", leave=False):
#         x = torch.randn(*shape, dtype=dtype, device=device)
#         time_torch = triton.testing.do_bench(lambda: add_torch(x, 1.0), warmup=5, rep=25)
#         time_triton = triton.testing.do_bench(lambda: add_triton(x, 1.0), warmup=5, rep=25)
#         stats["torch"][shape] = time_torch
#         stats["triton"][shape] = time_triton
#     print(pd.DataFrame(stats).T.to_markdown())
