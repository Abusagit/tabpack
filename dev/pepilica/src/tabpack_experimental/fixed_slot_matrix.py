"""Runtime rectangular GEMM plans with stable compact parameter slots."""

import math
from collections import OrderedDict

import torch
import triton
import triton.language as tl
from torch import nn
from torch.autograd.function import once_differentiable

from tabpack_experimental.ragged import _grouped_dot


def matrix_shapes(values):
    shapes = tuple(
        (int(v), int(v)) if isinstance(v, int) else tuple(map(int, v)) for v in values
    )
    if not shapes or any(len(s) != 2 or min(s) < 1 for s in shapes):
        raise ValueError('Expected positive widths or (output, input) matrix shapes')
    return shapes


def active_members(active, count):
    ids = tuple(range(count)) if active is None else tuple(map(int, active))
    if len(set(ids)) != len(ids) or any(i < 0 or i >= count for i in ids):
        raise ValueError('Expected unique valid candidate IDs')
    return ids


class FixedSlotGemm:
    """Reuse the generic ragged dot kernel; dimensions/offsets stay runtime data.

    Descriptors are (Aoff, Boff, Coff, M, N, K, Am, Ak, Bk, Bn, Cm, 0, biasoff).
    Output columns are contiguous. Inputs may have arbitrary nonnegative strides.
    Descriptor rows are permanent candidate IDs; tasks select the active rows.
    """

    def __init__(self, descriptors, active, *, device, tile=(64, 64, 32)):
        self.active = active_members(active, len(descriptors))
        self.tile = tuple(tile)
        if any(v < 16 or v & (v - 1) for v in self.tile):
            raise ValueError('Expected power-of-two tiles >= 16')
        if any(
            len(d) != 13
            or min(d[:3]) < 0
            or min(d[3:6]) < 1
            or min(d[6:11]) < 0
            or d[12] < 0
            for d in descriptors
        ):
            raise ValueError('Invalid GEMM descriptor')
        self.descriptors = torch.tensor(descriptors, dtype=torch.int64, device=device)
        bm, bn, _ = self.tile
        tasks = [
            (i, m, n)
            for i in self.active
            for m in range((descriptors[i][3] + bm - 1) // bm)
            for n in range((descriptors[i][4] + bn - 1) // bn)
        ]
        tasks.sort(key=lambda t: -descriptors[t[0]][5])
        self.tasks = torch.tensor(tasks, dtype=torch.int32, device=device).reshape(
            -1, 3
        )

    def run(self, a, b, out, *, bias=None, addend=None, alpha=1.0, beta=0.0):
        if (
            a.dtype not in (torch.float16, torch.bfloat16)
            or b.dtype != a.dtype
            or out.dtype != a.dtype
            or any(t.device != self.descriptors.device for t in (a, b, out))
        ):
            raise ValueError('Expected matching CUDA FP16/BF16 GEMM buffers')
        if beta and addend is None:
            raise ValueError('A nonzero beta requires an addend')
        if len(self.tasks):
            _grouped_dot[(len(self.tasks),)](
                a,
                b,
                bias,
                out,
                self.descriptors,
                self.tasks,
                None,
                None,
                None,
                *self.tile,
                bias is not None,
                False,
                False,
                False,
                False,
                False,
                0,
                0,
                0,
                addend,
                alpha,
                beta,
                num_warps=4,
            )
        return out


class RectangularFixedSlotPlan:
    """Batch-major linear plans with fixed parameters and optional active-only columns."""

    def __init__(self, shapes, batch, active=None, *, device='cuda', active_only=False, tile=(64, 64, 32)):
        self.shapes = matrix_shapes(shapes)
        self.batch = int(batch)
        if self.batch < 1:
            raise ValueError('Expected a positive batch size')
        self.active = active_members(active, len(self.shapes))
        self.active_only = bool(active_only)
        layout = self.active if self.active_only else range(len(self.shapes))
        offsets = {}
        self.inputs = self.outputs = 0
        for i in layout:
            offsets[i] = self.inputs, self.outputs
            h, d = self.shapes[i]
            self.inputs += d
            self.outputs += h
        self.bias_size = sum(h for h, d in self.shapes)
        self.weight_size = sum(h * d for h, d in self.shapes)
        fw, dx, dw, bt = [], [], [], []
        bo = wo = 0
        for i, (h, d) in enumerate(self.shapes):
            xo, yo = offsets.get(i, (0, 0))
            fw.append(
                (xo, wo, yo, batch, h, d, self.inputs, 1, 1, d, self.outputs, 0, bo)
            )
            dx.append(
                (yo, wo, xo, batch, d, h, self.outputs, 1, d, 1, self.inputs, 0, 0)
            )
            dw.append(
                (yo, xo, wo, h, d, batch, 1, self.outputs, self.inputs, 1, d, 0, 0)
            )
            if i in self.active:
                bt.extend((i, n) for n in range((h + 31) // 32))
            bo += h
            wo += h * d
        self.forward = FixedSlotGemm(fw, self.active, device=device, tile=tile)
        self.input_grad = FixedSlotGemm(dx, self.active, device=device, tile=tile)
        self.weight_grad = FixedSlotGemm(dw, self.active, device=device, tile=tile)
        self.bias_tasks = torch.tensor(bt, dtype=torch.int32, device=device).reshape(
            -1, 2
        )

    def __call__(self, x, weight, bias):
        if (
            x.shape != (self.batch, self.inputs)
            or weight.shape != (self.weight_size,)
            or bias.shape != (self.bias_size,)
            or any(not t.is_contiguous() for t in (x, weight, bias))
            or bias.dtype != x.dtype
            or bias.device != x.device
        ):
            raise ValueError(
                'Input, weights, and bias must match the fixed-slot layout'
            )
        return _RectangularLinear.apply(x, weight, bias, self)


class _RectangularLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, plan):
        out = (x.new_empty if plan.active_only else x.new_zeros)(
            plan.batch, plan.outputs
        )
        plan.forward.run(x, weight, out, bias=bias)
        ctx.save_for_backward(x, weight)
        ctx.plan = plan
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        plan = ctx.plan
        grad = grad.contiguous()
        dx = (
            (torch.empty_like if plan.active_only else torch.zeros_like)(x)
            if ctx.needs_input_grad[0] else None
        )
        dw = torch.zeros_like(weight) if ctx.needs_input_grad[1] else None
        db = grad.new_zeros(plan.bias_size) if ctx.needs_input_grad[2] else None
        if dx is not None:
            plan.input_grad.run(grad, weight, dx)
        if dw is not None:
            plan.weight_grad.run(grad, x, dw)
        if db is not None and len(plan.bias_tasks):
            from tabpack_experimental.compact_mlp import _bias_gradient

            _bias_gradient[(len(plan.bias_tasks),)](
                grad,
                db,
                plan.forward.descriptors,
                plan.bias_tasks,
                128,
                32,
                num_warps=4,
            )
        return dx, dw, db, None


class CompactFixedSlotLinear(nn.Module):
    """Stable parameter slots; active_only compacts activations in routed order."""

    def __init__(self, shapes, *, device='cuda', dtype=torch.float32, active_only=False):
        super().__init__()
        self.shapes = matrix_shapes(shapes)
        self.active_only = bool(active_only)
        self.weight = nn.Parameter(
            torch.empty(sum(h * d for h, d in self.shapes), device=device, dtype=dtype)
        )
        self.bias = nn.Parameter(
            torch.zeros(sum(h for h, d in self.shapes), device=device, dtype=dtype)
        )
        with torch.no_grad():
            offset = 0
            for h, d in self.shapes:
                self.weight[offset : offset + h * d].uniform_(
                    -1 / math.sqrt(d), 1 / math.sqrt(d)
                )
                offset += h * d
        self._plans = OrderedDict()

    def forward(self, x, active=None):
        ids = active_members(active, len(self.shapes))
        if x.device != self.weight.device or x.dtype not in (
            torch.float16,
            torch.bfloat16,
        ):
            raise ValueError('Expected FP16/BF16 activations on the parameter device')
        key = (x.shape[0], ids, self.active_only)
        if key not in self._plans:
            with torch.inference_mode(False):
                self._plans[key] = RectangularFixedSlotPlan(
                    self.shapes, x.shape[0], ids, device=x.device,
                    active_only=self.active_only,
                )
            if len(self._plans) > 32:
                self._plans.popitem(last=False)
        self._plans.move_to_end(key)
        return self._plans[key](x, self.weight.to(x.dtype), self.bias.to(x.dtype))


@triton.jit(
    do_not_specialize=['WIDTH', 'BATCH', 'SM', 'SB', 'SF'],
    do_not_specialize_on_alignment=['WIDTH', 'BATCH', 'SM', 'SB', 'SF'],
)
def _feature_columns(
    SRC,
    DST,
    MEMBERS,
    COLUMNS,
    WIDTH,
    BATCH,
    SM,
    SB,
    SF,
    SCATTER: tl.constexpr,
    BLOCK: tl.constexpr,
):
    j = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = j // WIDTH, j % WIDTH
    member = tl.load(MEMBERS + col)
    feature = tl.load(COLUMNS + col)
    valid = (row < BATCH) & (member >= 0)
    physical = member * SM + row * SB + feature * SF
    if SCATTER:
        value = tl.load(SRC + j, valid, 0)
        tl.store(DST + physical, value, valid)
    else:
        value = tl.load(SRC + physical, valid, 0)
        tl.store(DST + j, value, row < BATCH)


class FeatureColumnTransfer:
    """Gather feature columns into full-slot or active-only batch-major storage.

    Source is [local active member, batch, padded features]. Maps are explicit
    because flattened numerical embeddings contain per-feature padding gaps.
    """

    def __init__(self, columns, active, *, device, active_only=False):
        active = active_members(active, len(columns))
        if any(not c or min(c) < 0 or len(set(c)) != len(c) for c in columns):
            raise ValueError('Expected distinct nonnegative feature columns per member')
        local = {member: i for i, member in enumerate(active)}
        self.active = active
        self.active_only = bool(active_only)
        layout = active if self.active_only else range(len(columns))
        self.width = sum(len(columns[i]) for i in layout)
        self.max_column = max(max(columns[i]) for i in active) if active else -1
        self.members = torch.tensor(
            [local.get(i, -1) for i in layout for _ in columns[i]],
            device=device,
            dtype=torch.int64,
        )
        self.columns = torch.tensor(
            [j for i in layout for j in columns[i]], device=device, dtype=torch.int64
        )

    def pack(self, source, dtype):
        if (
            source.ndim != 3
            or source.shape[0] != len(self.active)
            or source.shape[2] <= self.max_column
            or source.device != self.members.device
            or dtype not in (torch.float16, torch.bfloat16)
        ):
            raise ValueError('Feature input does not match the prepared column map')
        return _GatherFeatureColumns.apply(source, self, dtype)


class _GatherFeatureColumns(torch.autograd.Function):
    @staticmethod
    def forward(ctx, source, plan, dtype):
        out = torch.empty(
            (source.shape[1], plan.width), device=source.device, dtype=dtype
        )
        if out.numel():
            _feature_columns[(triton.cdiv(out.numel(), 1024),)](
                source,
                out,
                plan.members,
                plan.columns,
                plan.width,
                source.shape[1],
                *source.stride(),
                False,
                1024,
            )
        ctx.plan, ctx.shape, ctx.dtype = plan, source.shape, source.dtype
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        plan = ctx.plan
        out = torch.zeros(ctx.shape, device=grad.device, dtype=ctx.dtype)
        grad = grad.contiguous()
        if grad.numel():
            _feature_columns[(triton.cdiv(grad.numel(), 1024),)](
                grad,
                out,
                plan.members,
                plan.columns,
                plan.width,
                out.shape[1],
                *out.stride(),
                True,
                1024,
            )
        return out, None, None
