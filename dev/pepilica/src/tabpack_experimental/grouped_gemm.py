"""Runtime descriptor-driven GEMM for rectangular layers and Muon products."""

import triton
import triton.language as tl


@triton.jit
def grouped_dot(
    A,
    B,
    Bias,
    C,
    Descriptors,
    Tasks,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    BIAS: tl.constexpr,
    Addend,
    ALPHA: tl.constexpr,
    BETA: tl.constexpr,
):
    task = tl.program_id(0)
    problem = tl.load(Tasks + task * 3)
    tile_m = tl.load(Tasks + task * 3 + 1)
    tile_n = tl.load(Tasks + task * 3 + 2)
    desc = Descriptors + problem * 13
    ao = tl.load(desc)
    bo = tl.load(desc + 1)
    co = tl.load(desc + 2)
    m = tl.load(desc + 3)
    n = tl.load(desc + 4)
    k = tl.load(desc + 5)
    am = tl.load(desc + 6)
    ak = tl.load(desc + 7)
    bk = tl.load(desc + 8)
    bn = tl.load(desc + 9)
    cm = tl.load(desc + 10)
    bias_offset = tl.load(desc + 12)
    rows = tile_m * BM + tl.arange(0, BM)
    cols = tile_n * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    accumulator = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(k, BK)):
        reduction = start * BK + kk
        a = tl.load(
            A + ao + rows[:, None] * am + reduction[None, :] * ak,
            (rows[:, None] < m) & (reduction[None, :] < k),
            0,
        )
        b = tl.load(
            B + bo + reduction[:, None] * bk + cols[None, :] * bn,
            (reduction[:, None] < k) & (cols[None, :] < n),
            0,
        )
        accumulator = tl.dot(a, b, accumulator)
    if BIAS:
        accumulator += tl.load(Bias + bias_offset + cols, cols < n, 0)[None, :]
    offsets = co + rows[:, None] * cm + cols[None, :]
    valid = (rows[:, None] < m) & (cols[None, :] < n)
    if ALPHA != 1.0:
        accumulator = accumulator * ALPHA
    if BETA != 0.0:
        accumulator += BETA * tl.load(Addend + offsets, valid, 0).to(tl.float32)
    tl.store(C + offsets, accumulator, valid)
