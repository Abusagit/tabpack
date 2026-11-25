"""Rectangular plans and feature transfer for active-only fixed-slot layers."""

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

from tabpack_experimental.grouped_gemm import grouped_dot


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
    """Dimensions and offsets are runtime descriptors shared by all GEMMs.

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
            grouped_dot[(len(self.tasks),)](
                a,
                b,
                bias,
                out,
                self.descriptors,
                self.tasks,
                *self.tile,
                bias is not None,
                addend,
                alpha,
                beta,
                num_warps=4,
            )
        return out


class RectangularFixedSlotPlan:
    """Batch-major active-only linear plans with fixed parameter slots."""

    def __init__(self, shapes, batch, active=None, *, device='cuda', tile=(64, 64, 32)):
        self.shapes = matrix_shapes(shapes)
        self.batch = int(batch)
        if self.batch < 1:
            raise ValueError('Expected a positive batch size')
        self.active = active_members(active, len(self.shapes))
        layout = self.active
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
    """Gather selected feature columns into active-only batch-major storage.

    Source is [local active member, batch, padded features]. Maps are explicit
    because flattened numerical embeddings contain per-feature padding gaps.
    """

    def __init__(self, columns, active, *, device):
        active = active_members(active, len(columns))
        if any(not c or min(c) < 0 or len(set(c)) != len(c) for c in columns):
            raise ValueError('Expected distinct nonnegative feature columns per member')
        local = {member: i for i, member in enumerate(active)}
        self.active = active
        layout = active
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
