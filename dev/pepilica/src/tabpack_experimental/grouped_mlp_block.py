"""One fixed-slot MLP block for rectangular and square candidate matrices."""

import math
from collections import OrderedDict
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from torch import nn
from torch.autograd.function import once_differentiable

from tabpack_experimental.fixed_slot_matrix import (
    RectangularFixedSlotPlan,
    _bias_gradient,
    active_members,
)


@triton.jit(
    do_not_specialize=['SIZE', 'COLS'],
    do_not_specialize_on_alignment=['SIZE', 'COLS'],
)
def _block_pointwise(
    X,
    Z,
    Y,
    DerivativeMask,
    Mask,
    Keep,
    Seed,
    SIZE,
    COLS,
    ALL: tl.constexpr,
    CARRY: tl.constexpr,
    RELU: tl.constexpr,
    DROPOUT: tl.constexpr,
    RANDOM: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    j = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = j < SIZE
    if ALL:
        active = tl.full((BLOCK,), True, tl.int1)
    else:
        active = tl.load(Mask + j % COLS, valid, 0) != 0
    if MODE == 'carry':
        v = tl.load(X + j, valid & ~active, 0)
        tl.store(Y + j, v, valid & ~active)
    else:
        if DROPOUT:
            q = tl.load(Keep + j % COLS, valid & active, 1)
            keep = q > 0
            if RANDOM:
                seed = tl.load(Seed).to(tl.uint64)
                keep = keep & (tl.rand(seed, j) < q)
            scale = tl.where(keep, 1.0 / tl.where(q > 0, q, 1), 0)
        if MODE == 'forward':
            v = tl.load(Z + j, valid & active, 0).to(tl.float32)
            if RELU:
                if DerivativeMask is not None:
                    tl.store(DerivativeMask + j, active & (v > 0), valid)
                v = tl.maximum(v, 0)
        else:
            v = tl.load(X + j, valid & active, 0).to(tl.float32)
            if RELU:
                z = tl.load(Z + j, valid & active, 0)
                v = tl.where(z > 0, v, 0)
        if DROPOUT:
            v = tl.where(keep, v * scale, 0)
        if MODE == 'forward' and CARRY:
            carry = tl.load(X + j, valid & ~active, 0).to(tl.float32)
            v = tl.where(active, v, carry)
        else:
            v = tl.where(active, v, 0)
        tl.store(Y + j, v, valid)


class _SquareGemm:
    def __init__(self, widths, batch, layout, participating, device, tile):
        from tabpack_experimental.active_fixed_slot import ActiveSlotPlan

        self.plan = ActiveSlotPlan(widths, batch, layout, participating, device, tile)
        self.inputs = self.outputs = self.plan.stride
        self.bias_size = self.plan.bias_size
        self.weight_size = self.plan.weight_size
        self.batch = batch
        self.participating = participating

    def forward(self, x, weight, bias, out):
        from tabpack_experimental.active_fixed_slot import active_slot_mm

        p = self.plan
        if len(p.forward_tasks):
            active_slot_mm[(len(p.forward_tasks),)](
                x,
                weight,
                bias,
                out,
                p.descriptors,
                p.forward_tasks,
                p.stride,
                p.batch,
                p.tile,
                p.tile,
                32,
                num_warps=4,
            )

    def backward(self, dz, x, weight, dx, dw, db):
        from tabpack_experimental.active_fixed_slot import (
            active_slot_db,
            active_slot_dw,
            active_slot_dx,
        )

        p = self.plan
        if dx is not None and len(p.forward_tasks):
            active_slot_dx[(len(p.forward_tasks),)](
                dz,
                weight,
                dx,
                p.descriptors,
                p.forward_tasks,
                p.stride,
                p.batch,
                p.tile,
                p.tile,
                32,
                num_warps=4,
            )
        if dw is not None and len(p.weight_tasks):
            active_slot_dw[(len(p.weight_tasks),)](
                dz,
                x,
                dw,
                p.descriptors,
                p.weight_tasks,
                p.stride,
                p.batch,
                p.tile,
                p.tile,
                32,
                num_warps=4,
            )
        if db is not None and len(p.bias_tasks):
            active_slot_db[(len(p.bias_tasks),)](
                dz,
                db,
                p.descriptors,
                p.bias_tasks,
                p.stride,
                p.batch,
                128,
                32,
                num_warps=4,
            )


class _RectangularGemm:
    def __init__(self, shapes, batch, active, device, tile):
        self.plan = RectangularFixedSlotPlan(
            shapes, batch, active, device=device, tile=tile
        )
        for name in ('inputs', 'outputs', 'weight_size', 'bias_size', 'batch'):
            setattr(self, name, getattr(self.plan, name))

    def forward(self, x, weight, bias, out):
        self.plan.forward.run(x, weight, out, bias=bias)

    def backward(self, dz, x, weight, dx, dw, db):
        p = self.plan
        if dx is not None:
            p.input_grad.run(dz, weight, dx)
        if dw is not None:
            p.weight_grad.run(dz, x, dw)
        if db is not None and len(p.bias_tasks):
            _bias_gradient[(len(p.bias_tasks),)](
                dz, db, p.forward.descriptors, p.bias_tasks, 128, 32, num_warps=4
            )


@dataclass(frozen=True)
class _BlockPlan:
    gemm: object
    mask: torch.Tensor
    keep: torch.Tensor
    all_participating: bool
    carry: bool
    has_dropout: bool
    stochastic: bool

    @property
    def inputs(self):
        return self.gemm.inputs

    @property
    def outputs(self):
        return self.gemm.outputs

    def pointwise(
        self,
        x,
        z,
        out,
        mode,
        activation,
        dropout=False,
        seed=None,
        derivative_mask=None,
    ):
        if out.numel():
            _block_pointwise[(triton.cdiv(out.numel(), 1024),)](
                x,
                z,
                out,
                derivative_mask,
                self.mask,
                self.keep,
                seed,
                out.numel(),
                self.outputs,
                self.all_participating,
                self.carry,
                activation == 'relu',
                dropout,
                dropout and self.stochastic,
                mode,
                1024,
            )


class _GroupedLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, seed, plan, activation, dropout):
        g = plan.gemm
        out = x.new_empty((g.batch, g.outputs))
        linear = torch.empty_like(out)
        g.forward(x, weight, bias, linear)
        preactivation = (
            torch.empty_like(out, dtype=torch.bool) if activation == 'relu' else None
        )
        plan.pointwise(
            x, linear, out, 'forward', activation, dropout, seed, preactivation
        )
        ctx.save_for_backward(x, weight, preactivation, seed, plan.keep)
        ctx.plan, ctx.activation, ctx.dropout = plan, activation, dropout
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        x, weight, preactivation, seed, _keep = ctx.saved_tensors
        plan, g = ctx.plan, ctx.plan.gemm
        dy = grad.contiguous()
        dz = torch.empty_like(dy)
        plan.pointwise(
            dy, preactivation, dz, 'backward', ctx.activation, ctx.dropout, seed
        )
        dx = torch.empty_like(x) if ctx.needs_input_grad[0] else None
        dw = torch.zeros_like(weight) if ctx.needs_input_grad[1] else None
        db = x.new_zeros(g.bias_size) if ctx.needs_input_grad[2] else None
        g.backward(dz, x, weight, dx, dw, db)
        if dx is not None and plan.carry and not plan.all_participating:
            plan.pointwise(dy, None, dx, 'carry', ctx.activation)
        return dx, dw, db, None, None, None, None


class GroupedMLPBlock(nn.Module):
    """Stable parameters and routed batch-major activations for one MLP block.

    Widths are specified in (input, output) order through separate arguments.
    Square blocks carry nonparticipating columns through unchanged. Rectangular
    blocks transform all routed members.
    ReLU and identity activation share packed dropout and first-order backward.
    """

    def __init__(
        self,
        input_widths,
        output_widths,
        *,
        activation='relu',
        dropout=0.0,
        device='cuda',
        dtype=torch.float32,
        tile=64,
        cache_size=32,
    ):
        super().__init__()
        self.input_widths = tuple(map(int, input_widths))
        self.output_widths = tuple(map(int, output_widths))
        n = len(self.input_widths)
        if (
            not n
            or len(self.output_widths) != n
            or min(self.input_widths + self.output_widths) < 1
        ):
            raise ValueError('Expected one positive input/output width per candidate')
        self.shapes = tuple(zip(self.output_widths, self.input_widths))
        self.widths = self.output_widths
        self.square = self.input_widths == self.output_widths
        self.activation = activation
        self.dropout = (
            (float(dropout),) * n
            if isinstance(dropout, (int, float))
            else tuple(map(float, dropout))
        )
        if (
            activation not in ('relu', 'identity')
            or len(self.dropout) != n
            or any(not 0 <= p <= 1 for p in self.dropout)
            or tile not in (32, 64)
            or cache_size < 1
            or dtype not in (torch.float16, torch.bfloat16, torch.float32)
        ):
            raise ValueError('Invalid activation, dropout, tile, cache size or dtype')
        self.tile, self.cache_size = tile, cache_size
        self.stochastic = any(0 < p < 1 for p in self.dropout)
        self.has_dropout = any(p > 0 for p in self.dropout)
        self.weight = nn.Parameter(
            torch.empty(sum(h * d for h, d in self.shapes), device=device, dtype=dtype)
        )
        self.bias = nn.Parameter(
            torch.zeros(sum(self.output_widths), device=device, dtype=dtype)
        )
        with torch.no_grad():
            off = 0
            for h, d in self.shapes:
                self.weight[off : off + h * d].normal_(std=math.sqrt(2 / d))
                off += h * d
        self._plans = OrderedDict()

    def prepare(self, batch, active=None, participating=None):
        active = active_members(active, len(self.shapes))
        selected = (
            active
            if participating is None
            else active_members(participating, len(self.shapes))
        )
        if any(i not in active for i in selected):
            raise ValueError(
                'Participating candidates must belong to the active layout'
            )
        if not self.square and set(selected) != set(active):
            raise ValueError('Rectangular blocks must transform every active candidate')
        if batch < 1:
            raise ValueError('Expected a positive batch size')
        key = (batch, active, selected, self.weight.device, self.tile)
        if key not in self._plans:
            layout = active
            with torch.inference_mode(False):
                if self.square:
                    gemm = _SquareGemm(
                        self.widths,
                        batch,
                        layout,
                        selected,
                        self.weight.device,
                        self.tile,
                    )
                else:
                    gemm = _RectangularGemm(
                        self.shapes,
                        batch,
                        active,
                        self.weight.device,
                        (self.tile, self.tile, 32),
                    )
                mask = torch.tensor(
                    [
                        i in selected
                        for i in layout
                        for _ in range(self.output_widths[i])
                    ],
                    dtype=torch.uint8,
                    device=self.weight.device,
                )
                keep = torch.tensor(
                    [
                        1 - self.dropout[i]
                        for i in layout
                        for _ in range(self.output_widths[i])
                    ],
                    dtype=torch.float32,
                    device=self.weight.device,
                )
                plan = _BlockPlan(
                    gemm,
                    mask,
                    keep,
                    len(selected) == len(layout),
                    self.square,
                    any(self.dropout[i] > 0 for i in selected),
                    any(0 < self.dropout[i] < 1 for i in selected),
                )
            self._plans[key] = plan
            if len(self._plans) > self.cache_size:
                self._plans.popitem(last=False)
        self._plans.move_to_end(key)
        return self._plans[key]

    def forward(self, x, active=None, seed=None, *, participating=None):
        if (
            x.ndim != 2
            or not x.is_contiguous()
            or x.device.type != 'cuda'
            or x.device != self.weight.device
            or x.dtype not in (torch.float16, torch.bfloat16)
        ):
            raise ValueError(
                'Expected contiguous CUDA FP16/BF16 batch-major activations'
            )
        plan = self.prepare(x.shape[0], active, participating)
        if x.shape[1] != plan.inputs:
            raise ValueError('Input width does not match the routed layout')
        dropout = self.training and plan.has_dropout
        if dropout and plan.stochastic and seed is None:
            seed = torch.randint(2**63 - 1, (), device=x.device, dtype=torch.int64)
        if seed is not None and (
            seed.device != x.device or seed.dtype != torch.int64 or seed.numel() != 1
        ):
            raise ValueError('Expected a scalar int64 seed on the activation device')
        return _GroupedLayer.apply(
            x,
            self.weight.to(x.dtype),
            self.bias.to(x.dtype),
            seed,
            plan,
            self.activation,
            dropout,
        )
