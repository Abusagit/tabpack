"""Normalization primitives used by the fixed-slot optimizers."""

import triton
import triton.language as tl


@triton.jit
def _norm_partials(X, P, D, Tasks, BLOCK: tl.constexpr):
    task = tl.program_id(0)
    member, chunk = tl.load(Tasks + 2 * task), tl.load(Tasks + 2 * task + 1)
    desc = D + member * 5
    base, rows, cols = tl.load(desc), tl.load(desc + 1), tl.load(desc + 2)
    partial = tl.load(desc + 3)
    index = chunk * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + base + index, index < rows * cols, 0).to(tl.float32)
    tl.store(P + partial + chunk, tl.sum(x * x, 0))


@triton.jit
def _normalize(X, Y, Norms, D, Tasks, BLOCK: tl.constexpr):
    task = tl.program_id(0)
    member, chunk = tl.load(Tasks + 2 * task), tl.load(Tasks + 2 * task + 1)
    desc = D + member * 5
    base, rows, cols = tl.load(desc), tl.load(desc + 1), tl.load(desc + 2)
    index = chunk * BLOCK + tl.arange(0, BLOCK)
    source = tl.where(rows > cols, (index % rows) * cols + index // rows, index)
    x = tl.load(X + base + source, index < rows * cols, 0).to(tl.float32)
    denominator = tl.load(Norms + member).to(tl.float32)
    tl.store(Y + base + index, x / denominator, index < rows * cols)
