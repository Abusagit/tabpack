"""Grouped Newton-Schulz iterations for CUDA Muon parameter packs."""

from itertools import accumulate

import torch
import triton
import triton.language as tl
from torch import Tensor

from tabpackage import nn

Options = tuple[int, int, int, int, int]


@triton.jit
def _grouped_gemm(
    A,
    B,
    C,
    D,
    sizes,
    prefix,
    count,
    AM,
    AR,
    AC,
    BM,
    BR,
    BC,
    CM,
    CR,
    CC,
    DM,
    DR,
    DC,
    ALPHA: tl.constexpr,
    BETA: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Write one tile of D = ALPHA * (A @ B) + BETA * C per program.

    sizes holds the M, N, and K dimensions for count matrices. prefix holds
    their cumulative tile counts. The M/R/C suffixes on A/B/C/D stride
    arguments mean matrix, row, and column. BLOCK_M/N/K set tile sizes.
    Binary search picks the matrix, then the loop reduces over its valid K.
    """
    pid = tl.program_id(0)
    low, high = 0, count
    while low < high:
        mid = (low + high) // 2
        right = tl.load(prefix + mid) <= pid
        low = tl.where(right, mid + 1, low)
        high = tl.where(right, high, mid)
    matrix = low - 1
    local = pid - tl.load(prefix + matrix)
    m = tl.load(sizes + matrix)
    n = tl.load(sizes + count + matrix)
    k = tl.load(sizes + 2 * count + matrix)
    column_tiles = tl.cdiv(n, BLOCK_N)
    r = (local // column_tiles) * BLOCK_M + tl.arange(0, BLOCK_M)
    c = (local % column_tiles) * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for start in range(tl.cdiv(k, BLOCK_K)):
        q = start * BLOCK_K + tl.arange(0, BLOCK_K)
        a = tl.load(
            A + matrix * AM + r[:, None] * AR + q[None, :] * AC,
            (r[:, None] < m) & (q[None, :] < k),
            other=0,
        )
        b = tl.load(
            B + matrix * BM + q[:, None] * BR + c[None, :] * BC,
            (q[:, None] < k) & (c[None, :] < n),
            other=0,
        )
        acc = tl.dot(a, b, acc)
    if BETA != 0:
        add = tl.load(
            C + matrix * CM + r[:, None] * CR + c[None, :] * CC,
            (r[:, None] < m) & (c[None, :] < n),
            other=0,
        )
        acc = ALPHA * acc + BETA * add.to(tl.float32)
    else:
        acc = ALPHA * acc
    tl.store(
        D + matrix * DM + r[:, None] * DR + c[None, :] * DC,
        acc,
        (r[:, None] < m) & (c[None, :] < n),
    )


class NewtonSchulz:
    def __init__(
        self, module: nn.LinearPack, device: torch.device, options: Options
    ) -> None:
        self.options = options
        rows = (
            [module.max_out_features] * module.pack_size
            if module.out_features is None
            else module.out_features.tolist()
        )
        cols = (
            [module.max_in_features] * module.pack_size
            if module.in_features is None
            else module.in_features.tolist()
        )
        self.transpose = module.max_out_features > module.max_in_features
        if self.transpose:
            rows, cols = cols, rows
        block_m, block_n, _, _, _ = options

        def layout(
            m: list[int], n: list[int], k: list[int]
        ) -> tuple[Tensor, Tensor, int]:
            counts = (
                triton.cdiv(r, block_m) * triton.cdiv(c, block_n) for r, c in zip(m, n)
            )
            prefix = list(accumulate(counts, initial=0))
            return (
                torch.tensor([m, n, k], dtype=torch.int32, device=device),
                torch.tensor(prefix, dtype=torch.int32, device=device),
                prefix[-1],
            )

        self.layouts = (
            layout(rows, rows, cols),
            layout(rows, rows, rows),
            layout(rows, cols, rows),
        )

    def __call__(self, update: Tensor, steps: int) -> Tensor:
        x = update.bfloat16()
        if self.transpose:
            x = x.mT
        x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        if steps:
            count, rows, _ = x.shape
            a = x.new_zeros(count, rows, rows)
            b = torch.zeros_like(a)
            result = torch.zeros_like(x)
            for _ in range(steps):
                result.zero_()
                self._gemm(x, x.mT, a, a, 0)
                self._gemm(a, a, a, b, 1, alpha=2.0315, beta=-4.7750)
                self._gemm(b, x, x, result, 2, beta=3.4445)
                x, result = result, x
        return x.mT if self.transpose else x

    def _gemm(
        self,
        lhs: Tensor,
        rhs: Tensor,
        addend: Tensor,
        output: Tensor,
        op: int,
        *,
        alpha: float = 1.0,
        beta: float = 0.0,
    ) -> None:
        sizes, prefix, tiles = self.layouts[op]
        block_m, block_n, block_k, num_warps, num_stages = self.options
        _grouped_gemm[(tiles,)](
            lhs,
            rhs,
            addend,
            output,
            sizes,
            prefix,
            lhs.shape[0],
            *lhs.stride(),
            *rhs.stride(),
            *addend.stride(),
            *output.stride(),
            alpha,
            beta,
            block_m,
            block_n,
            block_k,
            num_warps=num_warps,
            num_stages=num_stages,
        )
