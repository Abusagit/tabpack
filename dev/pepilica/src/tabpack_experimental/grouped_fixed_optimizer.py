"""Grouped Muon/AdamW for square matrices in stable fixed slots (CUDA prototype)."""

import math
from typing import Any

import torch
import triton as _triton
import triton.language as _tl
from torch import nn

from tabpack_experimental.ragged_muon import _norm_partials, _normalize

triton: Any = _triton
tl: Any = _tl


@triton.jit
def _prepare(W, G, M, U, D, T, H, BLOCK: tl.constexpr, NESTEROV: tl.constexpr):
    t = tl.program_id(0)
    i = tl.load(T + 2 * t)
    chunk = tl.load(T + 2 * t + 1)
    off = tl.load(D + 5 * i)
    width = tl.load(D + 5 * i + 1)
    j = chunk * BLOCK + tl.arange(0, BLOCK)
    valid = j < width * width
    lr = tl.load(H + 8 * i)
    decay = tl.load(H + 8 * i + 2)
    momentum = tl.load(H + 8 * i + 3)
    grad = tl.load(G + off + j, valid, 0)
    old = tl.load(M + off + j, valid, 0)
    new = old + (grad - old) * (1.0 - momentum)
    update = grad + (new - grad) * momentum if NESTEROV else new
    param = tl.load(W + off + j, valid, 0)
    tl.store(M + off + j, new, valid)
    tl.store(U + off + j, update, valid)
    tl.store(W + off + j, param * (1.0 - lr * decay), valid)


@triton.jit
def _finish_norm(P, N, D, ACTIVE, BLOCK: tl.constexpr):
    i = tl.load(ACTIVE + tl.program_id(0))
    start = tl.load(D + 5 * i + 3)
    count = tl.load(D + 5 * i + 4)
    j = tl.arange(0, BLOCK)
    total = tl.sum(tl.load(P + start + j, j < count, 0), 0)
    norm = tl.sqrt(total).to(tl.bfloat16).to(tl.float32)
    tl.store(N + i, (norm + 1e-7).to(tl.bfloat16))


@triton.jit
def _mm(
    A,
    B,
    C,
    ADD,
    D,
    TASKS,
    TRANSPOSE_B: tl.constexpr,
    ALPHA: tl.constexpr,
    BETA: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    t = tl.program_id(0)
    i = tl.load(TASKS + 3 * t)
    mi = tl.load(TASKS + 3 * t + 1)
    ni = tl.load(TASKS + 3 * t + 2)
    off = tl.load(D + 5 * i)
    width = tl.load(D + 5 * i + 1)
    rows = mi * BM + tl.arange(0, BM)
    cols = ni * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(width, BK)):
        k = start * BK + kk
        a = tl.load(
            A + off + rows[:, None] * width + k[None, :],
            (rows[:, None] < width) & (k[None, :] < width),
            0,
        )
        if TRANSPOSE_B:
            b = tl.load(
                B + off + cols[None, :] * width + k[:, None],
                (cols[None, :] < width) & (k[:, None] < width),
                0,
            )
        else:
            b = tl.load(
                B + off + k[:, None] * width + cols[None, :],
                (k[:, None] < width) & (cols[None, :] < width),
                0,
            )
        acc += tl.dot(a, b)
    idx = off + rows[:, None] * width + cols[None, :]
    valid = (rows[:, None] < width) & (cols[None, :] < width)
    if BETA != 0.0:
        acc = ALPHA * acc + BETA * tl.load(ADD + idx, valid, 0).to(tl.float32)
    tl.store(C + idx, acc, valid)


@triton.jit
def _apply(W, U, D, T, H, BLOCK: tl.constexpr):
    t = tl.program_id(0)
    i = tl.load(T + 2 * t)
    chunk = tl.load(T + 2 * t + 1)
    off = tl.load(D + 5 * i)
    width = tl.load(D + 5 * i + 1)
    j = chunk * BLOCK + tl.arange(0, BLOCK)
    valid = j < width * width
    param = tl.load(W + off + j, valid, 0)
    update = tl.load(U + off + j, valid, 0).to(tl.float32)
    tl.store(W + off + j, param - tl.load(H + 8 * i) * update, valid)


@triton.jit
def _adam_bias(B, G, M, V, STEP, D, ACTIVE, H, BLOCK: tl.constexpr):
    i = tl.load(ACTIVE + tl.program_id(0))
    off = tl.load(D + 5 * i + 2)
    width = tl.load(D + 5 * i + 1)
    j = tl.arange(0, BLOCK)
    valid = j < width
    lr = tl.load(H + 8 * i + 1)
    beta1 = tl.load(H + 8 * i + 4)
    beta2 = tl.load(H + 8 * i + 5)
    eps = tl.load(H + 8 * i + 6)
    step = tl.load(STEP + i) + 1
    tl.store(STEP + i, step)
    grad = tl.load(G + off + j, valid, 0)
    old = tl.load(M + off + j, valid, 0)
    old2 = tl.load(V + off + j, valid, 0)
    m = old + (grad - old) * (1.0 - beta1)
    v = old2 * beta2 + grad * grad * (1.0 - beta2)
    correction1 = 1.0 - tl.exp(step * tl.log(beta1))
    correction2 = 1.0 - tl.exp(step * tl.log(beta2))
    param = tl.load(B + off + j, valid, 0)
    tl.store(M + off + j, m, valid)
    tl.store(V + off + j, v, valid)
    tl.store(
        B + off + j,
        param - (lr / correction1) * m / (tl.sqrt(v) / tl.sqrt(correction2) + eps),
        valid,
    )


class GroupedFixedMuonAdamW:
    """Square FP32 master slots; active subsets share kernels and keep fixed offsets."""

    def __init__(
        self,
        weight,
        bias,
        widths,
        *,
        muon_lr,
        adamw_lr,
        weight_decay,
        muon_momentum=0.95,
        adamw_beta1=0.9,
        adamw_beta2=0.999,
        adamw_eps=1e-8,
        muon_newton_schulz_steps=5,
        muon_nesterov=True,
    ):
        self.widths = tuple(map(int, widths))
        n = len(self.widths)
        if (
            not n
            or min(self.widths) < 1
            or not isinstance(weight, nn.Parameter)
            or not isinstance(bias, nn.Parameter)
            or weight.device.type != 'cuda'
            or bias.device != weight.device
            or weight.dtype != torch.float32
            or bias.dtype != torch.float32
            or weight.ndim != 1
            or bias.ndim != 1
            or not weight.is_contiguous()
            or not bias.is_contiguous()
            or weight.numel() != sum(w * w for w in self.widths)
            or bias.numel() != sum(self.widths)
        ):
            raise ValueError(
                'Expected positive square widths and contiguous FP32 CUDA master slots'
            )
        if (
            not isinstance(muon_newton_schulz_steps, int)
            or muon_newton_schulz_steps < 0
        ):
            raise ValueError('Invalid Newton-Schulz steps')
        self.weight, self.bias = weight, bias
        self._ptr = (weight.data_ptr(), bias.data_ptr())
        self.steps = muon_newton_schulz_steps
        self.nesterov = bool(muon_nesterov)

        def expand(v):
            if isinstance(v, (int, float)):
                v = [float(v)] * n
            if len(v) != n:
                raise ValueError('Expected one hyperparameter per model')
            return list(map(float, v))

        lr, alr, wd, mom, b1, b2, eps = map(
            expand,
            (
                muon_lr,
                adamw_lr,
                weight_decay,
                muon_momentum,
                adamw_beta1,
                adamw_beta2,
                adamw_eps,
            ),
        )
        if any(not math.isfinite(v) or v < 0 for v in lr + alr + wd + eps) or any(
            not 0 <= v < 1 for v in mom + b1 + b2
        ):
            raise ValueError('Invalid optimizer hyperparameters')
        self.hyper = torch.tensor(
            list(zip(lr, alr, wd, mom, b1, b2, eps, [0.0] * n)),
            device=weight.device,
            dtype=torch.float32,
        )
        self.momentum = torch.zeros_like(weight)
        self.exp_avg = torch.zeros_like(bias)
        self.exp_avg_sq = torch.zeros_like(bias)
        self.bias_steps = torch.zeros(n, device=weight.device, dtype=torch.int64)
        desc = []
        off = col = partial = 0
        for width in self.widths:
            chunks = (width * width + 2047) // 2048
            desc.append((off, width, col, partial, chunks))
            off += width * width
            col += width
            partial += chunks
        self.desc = torch.tensor(desc, device=weight.device, dtype=torch.int64)
        # The normalization helper expects a column count in the third entry.
        self.norm_desc = torch.tensor(
            [(a, b, b, d, e) for a, b, c, d, e in desc],
            device=weight.device,
            dtype=torch.int64,
        )
        self.partials = torch.empty(partial, device=weight.device, dtype=torch.float32)
        self.norms = torch.empty(n, device=weight.device, dtype=torch.bfloat16)
        self.u, self.x, self.y, self.a, self.b = [
            torch.empty_like(weight, dtype=torch.bfloat16) for _ in range(5)
        ]
        self.cache = {}

    def _tasks(self, active):
        active = tuple(map(int, active))
        if len(set(active)) != len(active) or any(
            i < 0 or i >= len(self.widths) for i in active
        ):
            raise ValueError('Expected unique valid active indices')
        if active not in self.cache:
            chunks = [
                (i, j)
                for i in active
                for j in range((self.widths[i] ** 2 + 2047) // 2048)
            ]
            tiles = [
                (i, m, n)
                for i in active
                for m in range((self.widths[i] + 31) // 32)
                for n in range((self.widths[i] + 31) // 32)
            ]
            self.cache[active] = (
                torch.tensor(active, device=self.weight.device, dtype=torch.int32),
                torch.tensor(
                    chunks, device=self.weight.device, dtype=torch.int32
                ).reshape(-1, 2),
                torch.tensor(
                    tiles, device=self.weight.device, dtype=torch.int32
                ).reshape(-1, 3),
            )
        return self.cache[active]

    @torch.no_grad()
    def step(self, active):
        if (self.weight.data_ptr(), self.bias.data_ptr()) != self._ptr:
            raise RuntimeError('Fixed-slot storage changed; rebuild the optimizer')
        ids, chunks, tiles = self._tasks(active)
        if not len(ids):
            return
        if self.weight.grad is not None:
            if self.weight.grad.is_sparse or not self.weight.grad.is_contiguous():
                raise ValueError('Expected contiguous dense weight gradient')
            prepare: Any = _prepare
            partial: Any = _norm_partials
            finish: Any = _finish_norm
            normalize: Any = _normalize
            mm: Any = _mm
            apply: Any = _apply
            prepare[(len(chunks),)](
                self.weight,
                self.weight.grad,
                self.momentum,
                self.u,
                self.desc,
                chunks,
                self.hyper,
                2048,
                self.nesterov,
                num_warps=4,
                enable_fp_fusion=False,
            )
            torch.autograd.graph.increment_version(self.weight)
            partial[(len(chunks),)](
                self.u, self.partials, self.norm_desc, chunks, 2048, num_warps=4
            )
            finish[(len(ids),)](
                self.partials,
                self.norms,
                self.norm_desc,
                ids,
                triton.next_power_of_2(
                    max((w * w + 2047) // 2048 for w in self.widths)
                ),
                num_warps=4,
            )
            normalize[(len(chunks),)](
                self.u, self.x, self.norms, self.norm_desc, chunks, 2048, num_warps=4
            )
            x, y = self.x, self.y
            for _ in range(self.steps):
                mm[(len(tiles),)](
                    x,
                    x,
                    self.a,
                    x,
                    self.desc,
                    tiles,
                    True,
                    1.0,
                    0.0,
                    32,
                    32,
                    32,
                    num_warps=4,
                )
                mm[(len(tiles),)](
                    self.a,
                    self.a,
                    self.b,
                    self.a,
                    self.desc,
                    tiles,
                    False,
                    2.0315,
                    -4.775,
                    32,
                    32,
                    32,
                    num_warps=4,
                )
                mm[(len(tiles),)](
                    self.b,
                    x,
                    y,
                    x,
                    self.desc,
                    tiles,
                    False,
                    1.0,
                    3.4445,
                    32,
                    32,
                    32,
                    num_warps=4,
                )
                x, y = y, x
            apply[(len(chunks),)](
                self.weight,
                x,
                self.desc,
                chunks,
                self.hyper,
                2048,
                num_warps=4,
                enable_fp_fusion=False,
            )
        if self.bias.grad is not None:
            if self.bias.grad.is_sparse or not self.bias.grad.is_contiguous():
                raise ValueError('Expected contiguous dense bias gradient')
            adam: Any = _adam_bias
            adam[(len(ids),)](
                self.bias,
                self.bias.grad,
                self.exp_avg,
                self.exp_avg_sq,
                self.bias_steps,
                self.desc,
                ids,
                self.hyper,
                triton.next_power_of_2(max(self.widths)),
                num_warps=4,
                enable_fp_fusion=False,
            )
            torch.autograd.graph.increment_version(self.bias)

    def zero_grad(self, *, set_to_none=True):
        for p in (self.weight, self.bias):
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

    def state_dict(self):
        return {
            'widths': self.widths,
            'hyper': self.hyper.clone(),
            'momentum': self.momentum.clone(),
            'exp_avg': self.exp_avg.clone(),
            'exp_avg_sq': self.exp_avg_sq.clone(),
            'bias_steps': self.bias_steps.clone(),
            'steps': self.steps,
            'nesterov': self.nesterov,
        }

    def load_state_dict(self, state):
        if (
            tuple(state['widths']) != self.widths
            or state['steps'] != self.steps
            or state['nesterov'] != self.nesterov
        ):
            raise ValueError('Optimizer checkpoint configuration mismatch')
        for name in ('hyper', 'momentum', 'exp_avg', 'exp_avg_sq', 'bias_steps'):
            target = getattr(self, name)
            value = state[name]
            if value.shape != target.shape or value.dtype != target.dtype:
                raise ValueError('Invalid optimizer state shape or dtype')
        for name in ('hyper', 'momentum', 'exp_avg', 'exp_avg_sq', 'bias_steps'):
            getattr(self, name).copy_(state[name])
