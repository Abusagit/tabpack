"""Connected hidden-layer prototype with stable compact activation storage."""

import math
from typing import Any

import torch
import triton as _triton
import triton.language as _tl
from torch import nn
from torch.autograd.function import once_differentiable

from tabpack_experimental.ragged import LinearPlan, Tile

# Triton annotations describe GPU compilation, not Python call signatures.
triton: Any = _triton
tl: Any = _tl


@triton.jit
def _pointwise(
    X,
    Z,
    Y,
    ActiveColumns,
    KeepProbabilities,
    Seed,
    SIZE: tl.constexpr,
    COLS: tl.constexpr,
    ALL_ACTIVE: tl.constexpr,
    DROPOUT: tl.constexpr,
    RANDOM: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < SIZE
    if ALL_ACTIVE:
        active = tl.full((BLOCK,), True, tl.int1)
    else:
        active = tl.load(ActiveColumns + offsets % COLS, valid, 0) != 0
    if DROPOUT and MODE != 'carry_grad':
        probability = tl.load(KeepProbabilities + offsets % COLS, valid & active, 1)
        keep = probability > 0
        if RANDOM:
            seed = tl.load(Seed).to(tl.uint64)
            keep = keep & (tl.rand(seed, offsets) < probability)
        scale = tl.where(keep, 1.0 / tl.where(probability > 0, probability, 1), 0)
    if MODE == 'forward':
        z = tl.load(Z + offsets, valid & active, 0).to(tl.float32)
        x = tl.load(X + offsets, valid & ~active, 0).to(tl.float32)
        values = tl.maximum(z, 0)
        if DROPOUT:
            values = tl.where(keep, values * scale, 0)
        tl.store(Y + offsets, tl.where(active, values, x), valid)
    elif MODE == 'activation_grad':
        dy = tl.load(X + offsets, valid & active, 0).to(tl.float32)
        y = tl.load(Z + offsets, valid & active, 0)
        if DROPOUT:
            dy = tl.where(keep, dy * scale, 0)
        tl.store(Y + offsets, tl.where(active & (y > 0), dy, 0), valid)
    else:
        values = tl.load(X + offsets, valid & ~active, 0)
        tl.store(Y + offsets, values, valid & ~active)


@triton.jit
def _bias_gradient(DZ, DB, Descriptors, Tasks, BM: tl.constexpr, BN: tl.constexpr):
    task = tl.program_id(0)
    member = tl.load(Tasks + task * 2)
    column_tile = tl.load(Tasks + task * 2 + 1)
    desc = Descriptors + member * 13
    base = tl.load(desc + 2)
    batch = tl.load(desc + 3)
    width = tl.load(desc + 4)
    stride = tl.load(desc + 10)
    bias_base = tl.load(desc + 12)
    rows = tl.arange(0, BM)
    columns = column_tile * BN + tl.arange(0, BN)
    total = tl.zeros((BN,), tl.float32)
    for start in range(tl.cdiv(batch, BM)):
        r = start * BM + rows
        values = tl.load(
            DZ + base + r[:, None] * stride + columns[None, :],
            (r[:, None] < batch) & (columns[None, :] < width),
            0,
        )
        total += tl.sum(values.to(tl.float32), 0)
    tl.store(DB + bias_base + columns, total, columns < width)


class _CompactLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, seed, block):
        ctx.dropout = block.training and block.has_dropout
        if block.fused:
            output = block.forward_plan.run(
                x,
                weight,
                bias,
                keep_probabilities=block.keep_probabilities if ctx.dropout else None,
                seed=seed if ctx.dropout else None,
                stochastic=ctx.dropout and block.stochastic,
                scalar_probability=ctx.dropout and block.scalar_dropout,
            )
        else:
            linear = block.forward_plan.run(x, weight, bias)
            output = torch.empty_like(x)
            block.pointwise(x, linear, output, 'forward', seed, ctx.dropout)
        ctx.save_for_backward(
            x,
            weight,
            output,
            seed,
            block.keep_probabilities if ctx.dropout else None,
        )
        ctx.block = block
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        x, weight, activation, seed, probabilities = ctx.saved_tensors
        block = ctx.block
        grad_output = grad_output.contiguous()
        dz = torch.empty_like(grad_output)
        block.pointwise(
            grad_output,
            activation,
            dz,
            'activation_grad',
            seed,
            ctx.dropout,
            probabilities,
        )
        dx = dw = db = None
        if ctx.needs_input_grad[0]:
            dx = block.input_plan.run(dz, weight)
            if block.has_inactive:
                block.pointwise(grad_output, activation, dx, 'carry_grad')
        if ctx.needs_input_grad[1]:
            dw = block.weight_plan.run(dz, x)
        if ctx.needs_input_grad[2]:
            db = torch.empty(
                block.forward_plan.bias_size, dtype=x.dtype, device=x.device
            )
            kernel: Any = _bias_gradient
            kernel[(block.bias_tasks.shape[0],)](
                dz, db, block.forward_plan.descriptors, block.bias_tasks, 32, 128
            )
        return dx, dw, db, None, None


def width_tiles(widths, batch, mode, *, large=(64, 64, 32)):
    """Candidate tile policy, deliberately exposed for benchmarking."""
    result = {}
    for width in widths:
        m, n, k = (
            (width, width, batch) if mode == 'weight_grad' else (batch, width, width)
        )
        base = (16, 32, 32) if width <= 64 else large if width >= 1024 else (32, 64, 32)
        result[width] = Tile(
            *(
                min(block, triton.next_power_of_2(size))
                for block, size in zip(base, (m, n, k))
            )
        )
    return result


class CompactBlock(nn.Module):
    bias_tasks: torch.Tensor
    active_columns: torch.Tensor
    keep_probabilities: torch.Tensor

    def __init__(
        self,
        widths,
        batch,
        members,
        *,
        device,
        dtype,
        schedule,
        tiled,
        large_tile,
        dropout,
        fused,
        scalar_dropout,
        initialize_parameters=True,
    ):
        super().__init__()
        shapes = [(batch, width, width) for width in widths]
        self.has_inactive = len(members) != len(widths)
        self.total_width = sum(widths)
        self.fused = bool(fused)
        self.scalar_dropout = bool(scalar_dropout)
        self.has_dropout = any(dropout[i] > 0 for i in members)
        self.stochastic = any(0 < dropout[i] < 1 for i in members)
        self.register_buffer(
            'keep_probabilities',
            torch.tensor(
                [1 - p for p, width in zip(dropout, widths) for _ in range(width)],
                dtype=torch.float32,
                device=device,
            ),
            persistent=False,
        )
        plans = []
        for mode in ('forward', 'input_grad', 'weight_grad'):
            plans.append(
                LinearPlan(
                    shapes,
                    mode,
                    device=device,
                    members=members,
                    schedule=schedule,
                    fused_relu=self.fused and mode == 'forward',
                    tiles_by_width=(
                        width_tiles(widths, batch, mode, large=large_tile)
                        if tiled
                        else None
                    ),
                )
            )
        self.forward_plan, self.input_plan, self.weight_plan = plans
        plan = self.forward_plan
        if initialize_parameters:
            self.weight = nn.Parameter(
                torch.empty(plan.w_size, device=device, dtype=dtype)
            )
            self.bias = nn.Parameter(
                torch.zeros(plan.bias_size, device=device, dtype=dtype)
            )
            with torch.no_grad():
                for offset, (_, inputs, outputs) in zip(plan.w_offsets, plan.shapes):
                    self.weight[offset : offset + inputs * outputs].normal_(
                        std=math.sqrt(2 / inputs)
                    )
        self.register_buffer(
            'bias_tasks',
            torch.tensor(
                [
                    (i, j)
                    for i, (_, _, width) in enumerate(plan.shapes)
                    for j in range(triton.cdiv(width, 128))
                ],
                dtype=torch.int32,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            'active_columns',
            torch.tensor(
                [
                    int(member in members)
                    for member, width in enumerate(widths)
                    for _ in range(width)
                ]
                if self.has_inactive
                else [],
                dtype=torch.uint8,
                device=device,
            ),
            persistent=False,
        )

    def pointwise(self, x, z, out, mode, seed=None, dropout=False, probabilities=None):
        kernel: Any = _pointwise
        kernel[(triton.cdiv(x.numel(), 1024),)](
            x,
            z,
            out,
            self.active_columns,
            self.keep_probabilities if probabilities is None else probabilities,
            seed,
            x.numel(),
            self.total_width,
            not self.has_inactive,
            dropout,
            dropout and self.stochastic,
            mode,
            1024,
        )

    def forward(self, x, seed=None):  # ty: ignore[invalid-mutable-override]
        if self.training and self.stochastic and seed is None:
            seed = torch.randint(2**63 - 1, (), device=x.device, dtype=torch.int64)
        return _CompactLayer.apply(x, self.weight, self.bias, seed, self)


class CompactMLP(nn.Module):
    """Batch-major hidden stack with ReLU, packed dropout and first-order autograd.

    Input/output have shape [batch, sum(widths)]. Members own feature columns
    and keep them through layers after finishing. Device, dtype, widths and
    batch are fixed at construction. No prefix, head or optimizer.
    """

    def __init__(
        self,
        widths,
        batch,
        depths=3,
        *,
        device='cuda:0',
        dtype=torch.bfloat16,
        schedule='member',
        tiled=False,
        large_tile=(64, 64, 32),
        dropout: Any = 0.0,
        fused=False,
        scalar_dropout=False,
    ):
        super().__init__()
        if scalar_dropout and not fused:
            raise ValueError('Scalar dropout probabilities require fused=True')
        self.widths = tuple(map(int, widths))
        self.batch = int(batch)
        self.depths = (
            (depths,) * len(self.widths) if isinstance(depths, int) else tuple(depths)
        )
        if (
            not self.widths
            or min(self.widths) < 1
            or self.batch < 1
            or len(self.depths) != len(self.widths)
            or min(self.depths) < 1
        ):
            raise ValueError('Expected positive widths, batch and per-member depths')
        self.dropout: tuple[float, ...] = (
            (float(dropout),) * len(self.widths)
            if isinstance(dropout, int | float)
            else tuple(map(float, dropout))
        )
        if len(self.dropout) != len(self.widths) or any(
            not 0 <= p <= 1 for p in self.dropout
        ):
            raise ValueError('Expected one dropout probability in [0, 1] per member')
        self.blocks: Any = nn.ModuleList(
            CompactBlock(
                self.widths,
                self.batch,
                tuple(i for i, depth in enumerate(self.depths) if depth > layer),
                device=device,
                dtype=dtype,
                schedule=schedule,
                tiled=tiled,
                large_tile=large_tile,
                dropout=self.dropout,
                fused=fused,
                scalar_dropout=scalar_dropout,
            )
            for layer in range(max(self.depths))
        )

    def forward(self, x):  # ty: ignore[invalid-mutable-override]
        if tuple(x.shape) != (self.batch, sum(self.widths)) or not x.is_contiguous():
            raise ValueError('Expected contiguous [batch, sum(widths)] activations')
        count = sum(block.training and block.stochastic for block in self.blocks)
        seeds = iter(
            torch.randint(
                2**63 - 1, (count,), device=x.device, dtype=torch.int64
            ).unbind()
            if count
            else ()
        )
        for block in self.blocks:
            seed = next(seeds) if block.training and block.stochastic else None
            x = block(x, seed)
        return x

    def pack(self, padded):
        """Convert padded boundary input to batch-major feature columns."""
        if padded.shape != (len(self.widths), self.batch, max(self.widths)):
            raise ValueError('Unexpected padded shape')
        return torch.cat([padded[i, :, :h] for i, h in enumerate(self.widths)], dim=1)

    def unpack(self, compact):
        """Convert batch-major output to padded boundary storage."""
        if tuple(compact.shape) != (self.batch, sum(self.widths)):
            raise ValueError('Expected [batch, sum(widths)] activations')
        output = compact.new_zeros(len(self.widths), self.batch, max(self.widths))
        for member, part in enumerate(compact.split(self.widths, dim=1)):
            output[member, :, : self.widths[member]] = part
        return output


@triton.jit
def fixed_slot_mm(
    X,
    W,
    B,
    Y,
    D,
    T,
    STRIDE: tl.constexpr,
    BATCH: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    task = tl.program_id(0)
    member = tl.load(T + task * 3)
    mi = tl.load(T + task * 3 + 1)
    ni = tl.load(T + task * 3 + 2)
    d = D + member * 3
    col = tl.load(d)
    wo = tl.load(d + 1)
    width = tl.load(d + 2)
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
    bias = tl.load(B + col + cols, cols < width, 0)
    tl.store(
        Y + rows[:, None] * STRIDE + col + cols[None, :],
        acc + bias[None, :],
        (rows[:, None] < BATCH) & (cols[None, :] < width),
    )


@triton.jit
def fixed_slot_dx(
    DY,
    W,
    DX,
    D,
    T,
    STRIDE: tl.constexpr,
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
        tl.load(D + member * 3),
        tl.load(D + member * 3 + 1),
        tl.load(D + member * 3 + 2),
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
def fixed_slot_dw(
    DY,
    X,
    DW,
    D,
    T,
    STRIDE: tl.constexpr,
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
        tl.load(D + member * 3),
        tl.load(D + member * 3 + 1),
        tl.load(D + member * 3 + 2),
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
def fixed_slot_db(
    DY,
    DB,
    D,
    T,
    STRIDE: tl.constexpr,
    BATCH: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    task = tl.program_id(0)
    member, ni = tl.load(T + task * 3), tl.load(T + task * 3 + 2)
    col, width = tl.load(D + member * 3), tl.load(D + member * 3 + 2)
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
    tl.store(DB + col + cols, acc, cols < width)


class FixedSlotPlan:
    """Prepared active-set plan. Retain it unchanged until backward completes."""

    def __init__(self, widths, batch, active, *, device='cuda'):
        self.widths = tuple(int(w) for w in widths)
        self.batch = int(batch)
        self.active = tuple(int(i) for i in active)
        if (
            not self.widths
            or min(self.widths) < 1
            or self.batch < 1
            or len(set(self.active)) != len(self.active)
            or any(i < 0 or i >= len(self.widths) for i in self.active)
        ):
            raise ValueError('Invalid widths, batch or active members')
        self.stride = sum(self.widths)
        self.weight_size = sum(w * w for w in self.widths)
        descriptors, forward_tasks, weight_tasks, bias_tasks = [], [], [], []
        col = weight = 0
        for width in self.widths:
            descriptors.append((col, weight, width))
            col += width
            weight += width * width
        for i in self.active:
            width = self.widths[i]
            forward_tasks.extend(
                (i, m, n)
                for m in range((self.batch + 31) // 32)
                for n in range((width + 31) // 32)
            )
            weight_tasks.extend(
                (i, m, n)
                for m in range((width + 31) // 32)
                for n in range((width + 31) // 32)
            )
            bias_tasks.extend((i, 0, n) for n in range((width + 31) // 32))
        forward_tasks.sort(key=lambda task: -self.widths[task[0]])
        self.descriptors = torch.tensor(descriptors, device=device, dtype=torch.int64)
        self.forward_tasks = torch.tensor(
            forward_tasks, device=device, dtype=torch.int32
        ).reshape(-1, 3)
        self.weight_tasks = torch.tensor(
            weight_tasks, device=device, dtype=torch.int32
        ).reshape(-1, 3)
        self.bias_tasks = torch.tensor(
            bias_tasks, device=device, dtype=torch.int32
        ).reshape(-1, 3)

    def __call__(self, x, weight, bias):
        if (
            x.device.type != 'cuda'
            or x.device != self.descriptors.device
            or weight.device != x.device
            or bias.device != x.device
            or x.dtype not in (torch.bfloat16, torch.float16)
            or weight.dtype != x.dtype
            or bias.dtype != x.dtype
            or x.shape != (self.batch, self.stride)
            or weight.shape != (self.weight_size,)
            or bias.shape != (self.stride,)
            or not all(t.is_contiguous() for t in (x, weight, bias))
        ):
            raise ValueError(
                'Expected contiguous fixed-slot CUDA inputs with matching FP16/BF16'
            )
        return _FixedSlotLinear.apply(x, weight, bias, self)


class _FixedSlotLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, plan):
        output = torch.zeros_like(x)
        if plan.active:
            kernel: Any = fixed_slot_mm
            kernel[(len(plan.forward_tasks),)](
                x,
                weight,
                bias,
                output,
                plan.descriptors,
                plan.forward_tasks,
                plan.stride,
                plan.batch,
                32,
                32,
                32,
                num_warps=4,
            )
        ctx.save_for_backward(x, weight)
        ctx.plan = plan
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        plan = ctx.plan
        dy = grad_output.contiguous()
        dx = torch.zeros_like(x) if ctx.needs_input_grad[0] else None
        dw = torch.zeros_like(weight) if ctx.needs_input_grad[1] else None
        db = (
            torch.zeros((plan.stride,), device=x.device, dtype=x.dtype)
            if ctx.needs_input_grad[2]
            else None
        )
        if plan.active:
            if dx is not None:
                kernel_dx: Any = fixed_slot_dx
                kernel_dx[(len(plan.forward_tasks),)](
                    dy,
                    weight,
                    dx,
                    plan.descriptors,
                    plan.forward_tasks,
                    plan.stride,
                    plan.batch,
                    32,
                    32,
                    32,
                    num_warps=4,
                )
            if dw is not None:
                kernel_dw: Any = fixed_slot_dw
                kernel_dw[(len(plan.weight_tasks),)](
                    dy,
                    x,
                    dw,
                    plan.descriptors,
                    plan.weight_tasks,
                    plan.stride,
                    plan.batch,
                    32,
                    32,
                    32,
                    num_warps=4,
                )
            if db is not None:
                kernel_db: Any = fixed_slot_db
                kernel_db[(len(plan.bias_tasks),)](
                    dy,
                    db,
                    plan.descriptors,
                    plan.bias_tasks,
                    plan.stride,
                    plan.batch,
                    128,
                    32,
                    num_warps=4,
                )
        return dx, dw, db, None



@triton.jit
def fixed_slot_fused_mm(
    X,
    W,
    B,
    Y,
    D,
    T,
    KeepProbabilities,
    Seed,
    DROPOUT: tl.constexpr,
    RANDOM: tl.constexpr,
    STRIDE: tl.constexpr,
    BATCH: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    task = tl.program_id(0)
    member = tl.load(T + task * 3)
    mi = tl.load(T + task * 3 + 1)
    ni = tl.load(T + task * 3 + 2)
    d = D + member * 3
    col = tl.load(d)
    wo = tl.load(d + 1)
    width = tl.load(d + 2)
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
    bias = tl.load(B + col + cols, cols < width, 0)
    offsets = rows[:, None] * STRIDE + col + cols[None, :]
    valid = (rows[:, None] < BATCH) & (cols[None, :] < width)
    values = tl.maximum((acc + bias[None, :]).to(Y.dtype.element_ty).to(tl.float32), 0)
    if DROPOUT:
        q = tl.load(KeepProbabilities + col + cols, cols < width, 1)
        keep = q[None, :] > 0
        if RANDOM:
            seed = tl.load(Seed).to(tl.uint64)
            keep = keep & (tl.rand(seed, offsets.to(tl.uint32)) < q[None, :])
        scale = tl.where(keep, 1.0 / tl.where(q[None, :] > 0, q[None, :], 1), 0)
        values = tl.where(keep, values * scale, 0)
    tl.store(Y + offsets, values, valid)


class _FixedSlotCompactLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, seed, block, plan):
        dropout = block.training and block.has_dropout
        if block.fused:
            output = torch.empty_like(x)
            if block.has_inactive:
                block.pointwise(x, x, output, 'carry_grad')
            fixed_slot_fused_mm[(len(plan.forward_tasks),)](
                x, weight, bias, output, plan.descriptors, plan.forward_tasks,
                block.keep_probabilities, seed, dropout, dropout and block.stochastic,
                plan.stride, plan.batch, 32, 32, 32, num_warps=4)
        else:
            linear = plan(x, weight, bias)
            output = torch.empty_like(x)
            block.pointwise(x, linear, output, 'forward', seed, dropout)
        ctx.save_for_backward(x, weight, output, seed)
        ctx.block, ctx.plan, ctx.dropout = block, plan, dropout
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        x, weight, output, seed = ctx.saved_tensors
        block, plan = ctx.block, ctx.plan
        grad_output = grad_output.contiguous()
        dz = torch.empty_like(grad_output)
        block.pointwise(grad_output, output, dz, 'activation_grad', seed, ctx.dropout)
        dx = torch.zeros_like(x) if ctx.needs_input_grad[0] else None
        dw = torch.zeros_like(weight) if ctx.needs_input_grad[1] else None
        db = torch.zeros(plan.stride, device=x.device, dtype=x.dtype) if ctx.needs_input_grad[2] else None
        if dx is not None:
            fixed_slot_dx[(len(plan.forward_tasks),)](
                dz, weight, dx, plan.descriptors, plan.forward_tasks,
                plan.stride, plan.batch, 32, 32, 32, num_warps=4)
            if block.has_inactive:
                block.pointwise(grad_output, output, dx, 'carry_grad')
        if dw is not None:
            fixed_slot_dw[(len(plan.weight_tasks),)](
                dz, x, dw, plan.descriptors, plan.weight_tasks,
                plan.stride, plan.batch, 32, 32, 32, num_warps=4)
        if db is not None:
            fixed_slot_db[(len(plan.bias_tasks),)](
                dz, db, plan.descriptors, plan.bias_tasks,
                plan.stride, plan.batch, 128, 32, num_warps=4)
        return dx, dw, db, None, None, None


class _FixedSlotCompactBlock(nn.Module):
    pointwise = CompactBlock.pointwise

    def __init__(self, widths, batch, dropout, *, device, dtype, fused):
        super().__init__()
        self.widths, self.batch = widths, batch
        self.total_width = sum(widths)
        self.fused = fused
        self.has_dropout = any(p > 0 for p in dropout)
        self.stochastic = any(0 < p < 1 for p in dropout)
        self.weight = nn.Parameter(torch.empty(sum(w * w for w in widths), device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(self.total_width, device=device, dtype=dtype))
        with torch.no_grad():
            offset = 0
            for width in widths:
                self.weight[offset:offset + width * width].normal_(std=math.sqrt(2 / width))
                offset += width * width
        self.register_buffer('keep_probabilities', torch.tensor(
            [1 - p for p, width in zip(dropout, widths) for _ in range(width)],
            device=device, dtype=torch.float32), persistent=False)
        self._plans = {}

    def forward(self, x, selected, seed):
        key = (selected, x.device)
        if key not in self._plans:
            plan = FixedSlotPlan(self.widths, self.batch, selected, device=x.device)
            mask = torch.tensor([int(i in selected) for i, w in enumerate(self.widths)
                                 for _ in range(w)], device=x.device, dtype=torch.uint8)
            self._plans[key] = plan, mask
        plan, mask = self._plans[key]
        # A call-specific view keeps metadata stable if another active set is used before backward.
        class CallBlock:
            pointwise = CompactBlock.pointwise
        block = CallBlock()
        block.training, block.fused = self.training, self.fused
        block.has_dropout, block.stochastic = self.has_dropout, self.stochastic
        block.has_inactive = len(selected) != len(self.widths)
        block.total_width, block.active_columns = self.total_width, mask
        block.keep_probabilities = self.keep_probabilities
        return _FixedSlotCompactLayer.apply(
            x, self.weight.to(x.dtype), self.bias.to(x.dtype), seed, block, plan
        )


class CompactFixedSlotMLP(nn.Module):
    """CompactMLP-style hidden stack backed by stable fixed-slot linear kernels.

    Shapes and batch are fixed, while the optional active set can change between
    calls. Weights are compact by width but reserve a slot for every member at
    each depth. The fused path combines linear, bias, ReLU and dropout.
    """

    def __init__(self, widths, batch, depths=3, *, device='cuda:0',
                 dtype=torch.bfloat16, dropout=0.0, fused=False):
        super().__init__()
        self.widths, self.batch = tuple(map(int, widths)), int(batch)
        self.depths = ((depths,) * len(self.widths) if isinstance(depths, int)
                       else tuple(map(int, depths)))
        self.dropout = ((float(dropout),) * len(self.widths) if isinstance(dropout, (int, float))
                        else tuple(map(float, dropout)))
        if (not self.widths or min(self.widths) < 1 or self.batch < 1
                or len(self.depths) != len(self.widths) or min(self.depths) < 1
                or len(self.dropout) != len(self.widths)
                or any(not 0 <= p <= 1 for p in self.dropout)):
            raise ValueError('Expected positive widths, batch, depths and valid dropout')
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError('Expected FP16, BF16 or FP32 parameters')
        self.blocks = nn.ModuleList(_FixedSlotCompactBlock(
            self.widths, self.batch, self.dropout, device=device, dtype=dtype, fused=fused
        ) for _ in range(max(self.depths)))

    def forward(self, x, active=None):
        if (tuple(x.shape) != (self.batch, sum(self.widths)) or not x.is_contiguous()
                or x.device != self.blocks[0].weight.device
                or x.dtype not in (torch.float16, torch.bfloat16)):
            raise ValueError('Expected contiguous FP16/BF16 input on the model device')
        active = tuple(range(len(self.widths))) if active is None else tuple(map(int, active))
        if len(set(active)) != len(active) or any(i < 0 or i >= len(self.widths) for i in active):
            raise ValueError('Invalid active members')
        selected = [tuple(i for i in active if self.depths[i] > layer)
                    for layer in range(len(self.blocks))]
        count = sum(bool(ids) and block.training and block.stochastic
                    for block, ids in zip(self.blocks, selected))
        seeds = iter(torch.randint(2**63 - 1, (count,), device=x.device, dtype=torch.int64).unbind()
                     if count else ())
        for block, ids in zip(self.blocks, selected):
            if ids:
                seed = next(seeds) if block.training and block.stochastic else None
                x = block(x, ids, seed)
        return x

    pack = CompactMLP.pack
    unpack = CompactMLP.unpack
