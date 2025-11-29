# from collections import defaultdict

# import pandas as pd
import torch
import triton
import triton.language as tl

# from tqdm import tqdm


def add_torch(x: torch.Tensor, const_val: float) -> torch.Tensor:
    return x + const_val


@triton.jit
def _add_triton(x_ptr, const_val, res_ptr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets) + const_val
    tl.store(res_ptr + offsets, x)


def add_triton(x: torch.Tensor, const_val: float) -> torch.Tensor:
    assert x.is_contiguous()
    result: torch.tensor = torch.empty_like(x)

    grid = lambda meta: (1,)
    _add_triton[grid](x, const_val, result, BLOCK_SIZE=x.numel())

    return result


# if __name__ == "__main__":
#     shapes = [
#         (64,),
#         (64, 64),
#         (256,),
#         (256, 256),
#     ]
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
