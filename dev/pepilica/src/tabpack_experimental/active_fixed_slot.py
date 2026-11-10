"""Experimental active-only activations backed by stable fixed parameter slots."""
import math
from collections import OrderedDict
from types import SimpleNamespace

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

from tabpack_experimental.compact_mlp import CompactBlock, CompactFixedSlotMLP

@triton.jit
def active_slot_mm(
    X,
    W,
    B,
    Y,
    D,
    T,
    STRIDE,
    BATCH: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    task = tl.program_id(0)
    member = tl.load(T + task * 3)
    mi = tl.load(T + task * 3 + 1)
    ni = tl.load(T + task * 3 + 2)
    d = D + member * 4
    col = tl.load(d)
    wo = tl.load(d + 1)
    width = tl.load(d + 2)
    bias_col = tl.load(d + 3)
    rows = mi * BM + tl.arange(0, BM)
    cols = ni * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(width, BK)):
        k = start * BK + kk
        x = tl.load(
            X + rows[:, None] * STRIDE + col + k[None, :],
            (rows[:, None] < BATCH) & (k[None, :] < width),
            0,
        )
        w = tl.load(
            W + wo + cols[None, :] * width + k[:, None],
            (cols[None, :] < width) & (k[:, None] < width),
            0,
        )
        acc += tl.dot(x, w)
    bias = tl.load(B + bias_col + cols, cols < width, 0)
    tl.store(
        Y + rows[:, None] * STRIDE + col + cols[None, :],
        acc + bias[None, :],
        (rows[:, None] < BATCH) & (cols[None, :] < width),
    )

@triton.jit
def active_slot_dx(
    DY,
    W,
    DX,
    D,
    T,
    STRIDE,
    BATCH: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    task = tl.program_id(0)
    member, mi, ni = (
        tl.load(T + task * 3),
        tl.load(T + task * 3 + 1),
        tl.load(T + task * 3 + 2),
    )
    col, wo, width = (
        tl.load(D + member * 4),
        tl.load(D + member * 4 + 1),
        tl.load(D + member * 4 + 2),
    )
    rows = mi * BM + tl.arange(0, BM)
    cols = ni * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(width, BK)):
        k = start * BK + kk
        dy = tl.load(
            DY + rows[:, None] * STRIDE + col + k[None, :],
            (rows[:, None] < BATCH) & (k[None, :] < width),
            0,
        )
        w = tl.load(
            W + wo + k[:, None] * width + cols[None, :],
            (k[:, None] < width) & (cols[None, :] < width),
            0,
        )
        acc += tl.dot(dy, w)
    tl.store(
        DX + rows[:, None] * STRIDE + col + cols[None, :],
        acc,
        (rows[:, None] < BATCH) & (cols[None, :] < width),
    )

@triton.jit
def active_slot_dw(
    DY,
    X,
    DW,
    D,
    T,
    STRIDE,
    BATCH: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    task = tl.program_id(0)
    member, mi, ni = (
        tl.load(T + task * 3),
        tl.load(T + task * 3 + 1),
        tl.load(T + task * 3 + 2),
    )
    col, wo, width = (
        tl.load(D + member * 4),
        tl.load(D + member * 4 + 1),
        tl.load(D + member * 4 + 2),
    )
    rows = mi * BM + tl.arange(0, BM)
    cols = ni * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(BATCH, BK)):
        k = start * BK + kk
        dy = tl.load(
            DY + k[None, :] * STRIDE + col + rows[:, None],
            (k[None, :] < BATCH) & (rows[:, None] < width),
            0,
        )
        x = tl.load(
            X + k[:, None] * STRIDE + col + cols[None, :],
            (k[:, None] < BATCH) & (cols[None, :] < width),
            0,
        )
        acc += tl.dot(dy, x)
    tl.store(
        DW + wo + rows[:, None] * width + cols[None, :],
        acc,
        (rows[:, None] < width) & (cols[None, :] < width),
    )

@triton.jit
def active_slot_db(
    DY,
    DB,
    D,
    T,
    STRIDE,
    BATCH: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    task = tl.program_id(0)
    member, ni = tl.load(T + task * 3), tl.load(T + task * 3 + 2)
    col, width = tl.load(D + member * 4), tl.load(D + member * 4 + 2)
    bias_col = tl.load(D + member * 4 + 3)
    cols = ni * BN + tl.arange(0, BN)
    rows = tl.arange(0, BM)
    acc = tl.full((BN,), 0, tl.float32)
    for start in range(tl.cdiv(BATCH, BM)):
        row = start * BM + rows
        values = tl.load(
            DY + row[:, None] * STRIDE + col + cols[None, :],
            (row[:, None] < BATCH) & (cols[None, :] < width),
            0,
        )
        acc += tl.sum(values.to(tl.float32), axis=0)
    tl.store(DB + bias_col + cols, acc, cols < width)



class _CallBlock(SimpleNamespace):
    pointwise = CompactBlock.pointwise


class ActiveSlotPlan:
    """Metadata for an active-only layout and a layer's participating members."""

    def __init__(self, widths, batch, active, participating, device, tile=64):
        self.batch, self.active = batch, active
        self.stride = sum(widths[i] for i in active)
        self.bias_size = sum(widths)
        self.weight_size = sum(w * w for w in widths)
        self.tile = tile
        columns = {}
        offset = 0
        for i in active:
            columns[i] = offset
            offset += widths[i]
        desc, col, weight = [], 0, 0
        for i, w in enumerate(widths):
            desc.append((columns.get(i, 0), weight, w, col))
            col += w
            weight += w * w
        forward = [(i, m, n) for i in participating
                   for m in range(triton.cdiv(batch, tile))
                   for n in range(triton.cdiv(widths[i], tile))]
        forward.sort(key=lambda t: -widths[t[0]])
        weight_tasks = [(i, m, n) for i in participating
                        for m in range(triton.cdiv(widths[i], tile))
                        for n in range(triton.cdiv(widths[i], tile))]
        bias_tasks = [(i, 0, n) for i in participating
                      for n in range(triton.cdiv(widths[i], 32))]
        self.descriptors = torch.tensor(desc, device=device, dtype=torch.int64)
        self.forward_tasks = torch.tensor(forward, device=device, dtype=torch.int32).reshape(-1, 3)
        self.weight_tasks = torch.tensor(weight_tasks, device=device, dtype=torch.int32).reshape(-1, 3)
        self.bias_tasks = torch.tensor(bias_tasks, device=device, dtype=torch.int32).reshape(-1, 3)


class _ActiveLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, seed, block, plan):
        z = torch.empty_like(x)
        active_slot_mm[(len(plan.forward_tasks),)](
            x, weight, bias, z, plan.descriptors, plan.forward_tasks,
            plan.stride, plan.batch, plan.tile, plan.tile, 32, num_warps=4)
        output = torch.empty_like(x)
        dropout = block.training and block.has_dropout
        block.pointwise(x, z, output, 'forward', seed, dropout)
        ctx.save_for_backward(x, weight, output, seed)
        ctx.block, ctx.plan, ctx.dropout = block, plan, dropout
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        x, weight, output, seed = ctx.saved_tensors
        block, plan = ctx.block, ctx.plan
        dy = grad_output.contiguous()
        dz = torch.empty_like(dy)
        block.pointwise(dy, output, dz, 'activation_grad', seed, ctx.dropout)
        dx = torch.empty_like(x) if ctx.needs_input_grad[0] else None
        dw = torch.zeros_like(weight) if ctx.needs_input_grad[1] else None
        db = torch.zeros(plan.bias_size, device=x.device, dtype=x.dtype) if ctx.needs_input_grad[2] else None
        if dx is not None:
            active_slot_dx[(len(plan.forward_tasks),)](
                dz, weight, dx, plan.descriptors, plan.forward_tasks,
                plan.stride, plan.batch, plan.tile, plan.tile, 32, num_warps=4)
            if block.has_inactive:
                block.pointwise(dy, output, dx, 'carry_grad')
        if dw is not None:
            active_slot_dw[(len(plan.weight_tasks),)](
                dz, x, dw, plan.descriptors, plan.weight_tasks,
                plan.stride, plan.batch, plan.tile, plan.tile, 32, num_warps=4)
        if db is not None:
            active_slot_db[(len(plan.bias_tasks),)](
                dz, db, plan.descriptors, plan.bias_tasks,
                plan.stride, plan.batch, 128, 32, num_warps=4)
        return dx, dw, db, None, None, None


class ActiveFixedSlotMLP(CompactFixedSlotMLP):
    """Fixed parameter slots with active-only [batch, sum(active widths)] activations.

    The input columns are ordered by `active`. Membership changes require a new
    layout; existing activations can be subset/reordered using repack().
    Metadata is prepared once per layout and cached. Only unfused blocks are
    currently implemented. Full-shaped parameter gradients remain unchanged.
    """

    def __init__(self, widths, batch, depths=3, *, device='cuda:0',
                 dtype=torch.bfloat16, dropout=0.0, tile=64, cache_size=32):
        widths = tuple(widths)
        requested_depths = ((depths,) * len(widths) if isinstance(depths, int)
                            else tuple(map(int, depths)))
        if len(requested_depths) != len(widths) or any(d < 0 for d in requested_depths):
            raise ValueError('Expected one nonnegative depth per model')
        super().__init__(widths, batch, tuple(max(1, d) for d in requested_depths),
                         device=device, dtype=dtype, dropout=dropout)
        self.depths = requested_depths
        if tile not in (32, 64):
            raise ValueError('Supported tile sizes are 32 and 64')
        if cache_size < 1:
            raise ValueError('cache_size must be positive')
        self.tile = tile
        self.cache_size = cache_size
        self._active_layouts = OrderedDict()
        self._repack_indices = OrderedDict()

    def _ids(self, active):
        ids = tuple(map(int, active))
        if not ids or len(set(ids)) != len(ids) or any(i < 0 or i >= len(self.widths) for i in ids):
            raise ValueError('Expected nonempty unique valid active IDs')
        return ids

    def _put(self, cache, key, value):
        cache[key] = value
        cache.move_to_end(key)
        if len(cache) > self.cache_size:
            cache.popitem(last=False)
        return value

    def prepare(self, active, device=None):
        active = self._ids(active)
        device = torch.device(device or self.blocks[0].weight.device)
        key = (active, device)
        if key in self._active_layouts:
            self._active_layouts.move_to_end(key)
            return self._active_layouts[key]
        with torch.inference_mode(False):
            keep = torch.tensor([1 - self.dropout[i] for i in active
                                 for _ in range(self.widths[i])], dtype=torch.float32, device=device)
            layers = []
            for layer, module in enumerate(self.blocks):
                selected = tuple(i for i in active if self.depths[i] > layer)
                if not selected:
                    layers.append(None)
                    continue
                plan = ActiveSlotPlan(self.widths, self.batch, active, selected, device, self.tile)
                mask = torch.tensor([int(i in selected) for i in active
                                     for _ in range(self.widths[i])], dtype=torch.uint8, device=device)
                info = _CallBlock(
                    total_width=plan.stride,
                    active_columns=mask, keep_probabilities=keep,
                    has_inactive=len(active) != len(selected),
                    has_dropout=any(self.dropout[i] > 0 for i in selected),
                    stochastic=any(0 < self.dropout[i] < 1 for i in selected))
                layers.append((plan, info))
        return self._put(self._active_layouts, key, layers)

    def repack(self, x, source, target):
        """Subset/reorder an existing active layout in one gather operation."""
        source, target = self._ids(source), self._ids(target)
        if any(i not in source for i in target):
            raise ValueError('Newly active models require fresh input activations')
        if x.ndim != 2 or x.shape[1] != sum(self.widths[i] for i in source):
            raise ValueError('Input does not match source layout')
        if source == target:
            return x
        key = (source, target, x.device)
        if key not in self._repack_indices:
            offsets, pos = {}, 0
            for i in source:
                offsets[i] = pos
                pos += self.widths[i]
            indices = [j for i in target for j in range(offsets[i], offsets[i] + self.widths[i])]
            with torch.inference_mode(False):
                indices = torch.tensor(indices, dtype=torch.int64, device=x.device)
            self._put(self._repack_indices, key, indices)
        else:
            self._repack_indices.move_to_end(key)
        return x.index_select(1, self._repack_indices[key])

    def forward(self, x, active=None):
        active = self._ids(range(len(self.widths)) if active is None else active)
        if (x.shape != (self.batch, sum(self.widths[i] for i in active))
                or not x.is_contiguous() or x.device != self.blocks[0].weight.device
                or x.dtype not in (torch.float16, torch.bfloat16)):
            raise ValueError('Expected contiguous active-only BF16/FP16 input on the model device')
        layers = self.prepare(active, x.device)
        count = sum(item is not None and self.training and item[1].stochastic for item in layers)
        seeds = iter(torch.randint(2**63 - 1, (count,), device=x.device, dtype=torch.int64).unbind()
                     if count else ())
        for module, item in zip(self.blocks, layers):
            if item is None:
                continue
            plan, template = item
            block = _CallBlock(**vars(template))
            block.training = self.training
            seed = next(seeds) if self.training and block.stochastic else None
            x = _ActiveLayer.apply(x, module.weight.to(x.dtype), module.bias.to(x.dtype), seed, block, plan)
        return x
