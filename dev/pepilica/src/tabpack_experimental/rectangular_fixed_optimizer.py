"""Rectangular stable-slot Muon using the shared runtime ragged GEMM backend."""

import math
from collections import OrderedDict

import torch
import triton
import triton.language as tl
from torch import nn

from tabpack_experimental.fixed_slot_matrix import (
    FixedSlotGemm,
    active_members,
    matrix_shapes,
)
from tabpack_experimental.ragged_muon import _norm_partials, _normalize


@triton.jit
def _prepare_rect(W, G, M, U, D, T, H, BLOCK: tl.constexpr, NESTEROV: tl.constexpr):
    t = tl.program_id(0)
    i, chunk = tl.load(T + t * 2), tl.load(T + t * 2 + 1)
    d = D + i * 7
    off, rows, cols = tl.load(d), tl.load(d + 1), tl.load(d + 2)
    j = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = j < rows * cols
    lr, wd, momentum = (
        tl.load(H + i * 8),
        tl.load(H + i * 8 + 2),
        tl.load(H + i * 8 + 3),
    )
    grad = tl.load(G + off + j, mask, 0)
    old = tl.load(M + off + j, mask, 0)
    new = old + (grad - old) * (1 - momentum)
    update = grad + (new - grad) * momentum if NESTEROV else new
    param = tl.load(W + off + j, mask, 0)
    tl.store(M + off + j, new, mask)
    tl.store(U + off + j, update, mask)
    tl.store(W + off + j, param * (1 - lr * wd), mask)


@triton.jit
def _apply_rect(W, U, D, T, H, BLOCK: tl.constexpr):
    t = tl.program_id(0)
    i, chunk = tl.load(T + t * 2), tl.load(T + t * 2 + 1)
    d = D + i * 7
    off, rows, cols = tl.load(d), tl.load(d + 1), tl.load(d + 2)
    j = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = j < rows * cols
    source = tl.where(rows > cols, (j % cols) * rows + j // cols, j)
    update = tl.load(U + off + source, mask, 0).to(tl.float32)
    param = tl.load(W + off + j, mask, 0)
    scale = tl.sqrt(tl.maximum(1.0, rows.to(tl.float32) / cols))
    tl.store(W + off + j, param - tl.load(H + i * 8) * scale * update, mask)


class RectangularFixedMuonAdamW:
    """Internal rectangular engine for GroupedFixedMuonAdamW.

    States and scratch retain stable slots for all candidates; only active tasks
    execute. Gram scratch has sum(min(rows, columns)**2) elements per buffer.
    """

    def __init__(
        self,
        weight,
        bias,
        shapes,
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
        self.shapes = matrix_shapes(shapes)
        self.widths = tuple(h for h, d in self.shapes)
        n = len(self.shapes)
        if (
            not isinstance(weight, nn.Parameter)
            or not isinstance(bias, nn.Parameter)
            or weight.device.type != 'cuda'
            or bias.device != weight.device
            or weight.dtype != torch.float32
            or bias.dtype != torch.float32
            or weight.ndim != 1
            or bias.ndim != 1
            or not weight.is_contiguous()
            or not bias.is_contiguous()
            or weight.numel() != sum(h * d for h, d in self.shapes)
            or bias.numel() != sum(self.widths)
        ):
            raise ValueError(
                'Expected contiguous FP32 CUDA parameter slots matching shapes'
            )
        if (
            not isinstance(muon_newton_schulz_steps, int)
            or muon_newton_schulz_steps < 0
        ):
            raise ValueError('Invalid Newton-Schulz steps')
        self.weight, self.bias = weight, bias
        self._ptr = weight.data_ptr(), bias.data_ptr()
        self.steps, self.nesterov = muon_newton_schulz_steps, bool(muon_nesterov)

        def expand(v):
            values = (
                [float(v)] * n if isinstance(v, (int, float)) else list(map(float, v))
            )
            if len(values) != n:
                raise ValueError('Expected one hyperparameter per candidate')
            return values

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
        self.exp_avg, self.exp_avg_sq = torch.zeros_like(bias), torch.zeros_like(bias)
        self.bias_steps = torch.zeros(n, device=weight.device, dtype=torch.int64)
        desc, norm, bias_desc = [], [], []
        self.products = [[], [], []]
        off = bo = go = po = 0
        for h, d in self.shapes:
            r, c = min(h, d), max(h, d)
            count = triton.cdiv(h * d, 2048)
            desc.append((off, h, d, bo, go, po, count))
            norm.append((off, h, d, po, count))
            bias_desc.append((off, h, bo, po, count))
            self.products[0].append((off, off, go, r, r, c, c, 1, 1, c, r, 0, 0))
            self.products[1].append((go, go, go, r, r, r, r, 1, r, 1, r, 0, 0))
            self.products[2].append((go, off, off, r, c, r, r, 1, c, 1, c, 0, 0))
            off += h * d
            bo += h
            go += r * r
            po += count
        self.desc = torch.tensor(desc, device=weight.device, dtype=torch.int64)
        self.norm_desc = torch.tensor(norm, device=weight.device, dtype=torch.int64)
        self.bias_desc = torch.tensor(
            bias_desc, device=weight.device, dtype=torch.int64
        )
        self.partials = torch.empty(po, device=weight.device, dtype=torch.float32)
        self.norms = torch.empty(n, device=weight.device, dtype=torch.bfloat16)
        self.norm_block = triton.next_power_of_2(max(d[-1] for d in desc))
        self.u, self.x, self.y = [
            torch.empty_like(weight, dtype=torch.bfloat16) for _ in range(3)
        ]
        self.a, self.b = [
            torch.empty(go, device=weight.device, dtype=torch.bfloat16)
            for _ in range(2)
        ]
        self.cache = OrderedDict()

    def _plans(self, active):
        ids = active_members(active, len(self.shapes))
        if ids not in self.cache:
            chunks = [
                (i, j)
                for i in ids
                for j in range(triton.cdiv(math.prod(self.shapes[i]), 2048))
            ]
            self.cache[ids] = (
                torch.tensor(ids, device=self.weight.device, dtype=torch.int32),
                torch.tensor(
                    chunks, device=self.weight.device, dtype=torch.int32
                ).reshape(-1, 2),
                tuple(
                    FixedSlotGemm(d, ids, device=self.weight.device)
                    for d in self.products
                ),
            )
            if len(self.cache) > 32:
                self.cache.popitem(last=False)
        self.cache.move_to_end(ids)
        return self.cache[ids]

    @torch.no_grad()
    def step(self, active):
        from tabpack_experimental.grouped_fixed_optimizer import (
            _adam_bias,
            _finish_norm,
        )

        if (self.weight.data_ptr(), self.bias.data_ptr()) != self._ptr:
            raise RuntimeError('Fixed-slot storage changed; rebuild the optimizer')
        ids, chunks, (gram, square, product) = self._plans(active)
        if not len(ids):
            return
        for parameter in (self.weight, self.bias):
            if parameter.grad is not None and (
                parameter.grad.is_sparse or not parameter.grad.is_contiguous()
            ):
                raise ValueError('Expected contiguous dense gradients')
        if self.weight.grad is not None:
            _prepare_rect[(len(chunks),)](
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
            _norm_partials[(len(chunks),)](
                self.u, self.partials, self.norm_desc, chunks, 2048, num_warps=4
            )
            _finish_norm[(len(ids),)](
                self.partials,
                self.norms,
                self.norm_desc,
                ids,
                self.norm_block,
                num_warps=4,
            )
            _normalize[(len(chunks),)](
                self.u, self.x, self.norms, self.norm_desc, chunks, 2048, num_warps=4
            )
            x, y = self.x, self.y
            for _ in range(self.steps):
                gram.run(x, x, self.a)
                square.run(
                    self.a, self.a, self.b, addend=self.a, alpha=2.0315, beta=-4.775
                )
                product.run(self.b, x, y, addend=x, beta=3.4445)
                x, y = y, x
            _apply_rect[(len(chunks),)](
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
            _adam_bias[(len(ids),)](
                self.bias,
                self.bias.grad,
                self.exp_avg,
                self.exp_avg_sq,
                self.bias_steps,
                self.bias_desc,
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
            'shapes': self.shapes,
            'steps': self.steps,
            'nesterov': self.nesterov,
            **{
                k: getattr(self, k).clone()
                for k in ('hyper', 'momentum', 'exp_avg', 'exp_avg_sq', 'bias_steps')
            },
        }

    def load_state_dict(self, state):
        if (
            tuple(map(tuple, state['shapes'])) != self.shapes
            or state['steps'] != self.steps
            or state['nesterov'] != self.nesterov
        ):
            raise ValueError('Optimizer checkpoint configuration mismatch')
        keys = ('hyper', 'momentum', 'exp_avg', 'exp_avg_sq', 'bias_steps')
        for k in keys:
            if (
                state[k].shape != getattr(self, k).shape
                or state[k].dtype != getattr(self, k).dtype
            ):
                raise ValueError('Invalid optimizer state shape or dtype')
        for k in keys:
            getattr(self, k).copy_(state[k])
