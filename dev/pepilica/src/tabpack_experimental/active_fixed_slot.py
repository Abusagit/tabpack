"""Active-only activations with compact, permanent candidate parameter slots."""

from collections import OrderedDict

import torch
import triton
import triton.language as tl
from torch import nn

from tabpack_experimental.grouped_mlp_block import GroupedMLPBlock


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
        forward = [
            (i, m, n)
            for i in participating
            for m in range(triton.cdiv(batch, tile))
            for n in range(triton.cdiv(widths[i], tile))
        ]
        forward.sort(key=lambda t: -widths[t[0]])
        weight_tasks = [
            (i, m, n)
            for i in participating
            for m in range(triton.cdiv(widths[i], tile))
            for n in range(triton.cdiv(widths[i], tile))
        ]
        bias_tasks = [
            (i, 0, n) for i in participating for n in range(triton.cdiv(widths[i], 32))
        ]
        self.descriptors = torch.tensor(desc, device=device, dtype=torch.int64)
        self.forward_tasks = torch.tensor(
            forward, device=device, dtype=torch.int32
        ).reshape(-1, 3)
        self.weight_tasks = torch.tensor(
            weight_tasks, device=device, dtype=torch.int32
        ).reshape(-1, 3)
        self.bias_tasks = torch.tensor(
            bias_tasks, device=device, dtype=torch.int32
        ).reshape(-1, 3)


class ActiveFixedSlotMLP(nn.Module):
    """Routed [batch, sum(active widths)] MLP with optional grouped input layer.

    Depths count square hidden blocks. With input_widths, a rectangular block
    precedes them. Only active candidates execute; parameter slots remain fixed.
    """

    def __init__(
        self,
        widths,
        batch,
        depths=3,
        *,
        device='cuda:0',
        dtype=torch.bfloat16,
        dropout=0.0,
        tile=64,
        cache_size=32,
        input_widths=None,
    ):
        super().__init__()
        self.widths, self.batch = tuple(map(int, widths)), int(batch)
        n = len(self.widths)
        self.depths = (
            (int(depths),) * n if isinstance(depths, int) else tuple(map(int, depths))
        )
        self.dropout = (
            (float(dropout),) * n
            if isinstance(dropout, (int, float))
            else tuple(map(float, dropout))
        )
        self.input_widths = (
            None if input_widths is None else tuple(map(int, input_widths))
        )
        if (
            not n
            or min(self.widths) < 1
            or self.batch < 1
            or len(self.depths) != n
            or min(self.depths) < 0
            or len(self.dropout) != n
            or any(not 0 <= p <= 1 for p in self.dropout)
        ):
            raise ValueError(
                'Expected positive widths/batch, nonnegative depths and valid dropout'
            )
        if self.input_widths is not None and (
            len(self.input_widths) != n or min(self.input_widths) < 1
        ):
            raise ValueError('Expected one positive input width per candidate')
        if tile not in (32, 64) or cache_size < 1:
            raise ValueError('Expected tile 32/64 and positive cache size')
        self.tile, self.cache_size = tile, cache_size
        options = {
            'device': device,
            'dtype': dtype,
            'dropout': self.dropout,
            'tile': tile,
            'cache_size': cache_size,
        }
        self.input_layer = (
            None
            if self.input_widths is None
            else GroupedMLPBlock(self.input_widths, self.widths, **options)
        )
        count = (
            max(self.depths)
            if self.input_layer is not None
            else max(1, max(self.depths))
        )
        self.blocks = nn.ModuleList(
            GroupedMLPBlock(self.widths, self.widths, **options) for _ in range(count)
        )
        self._repack_indices = OrderedDict()

    def _parameter(self):
        return (
            self.input_layer.weight
            if self.input_layer is not None
            else self.blocks[0].weight
        )

    def _ids(self, active):
        ids = tuple(map(int, active))
        if (
            not ids
            or len(set(ids)) != len(ids)
            or any(i < 0 or i >= len(self.widths) for i in ids)
        ):
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
        if device is not None and torch.device(device) != self._parameter().device:
            raise ValueError('Plans must use the parameter device')
        return [
            module.prepare(
                self.batch, active, tuple(i for i in active if self.depths[i] > layer)
            )
            if any(self.depths[i] > layer for i in active)
            else None
            for layer, module in enumerate(self.blocks)
        ]

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
            indices = [
                j
                for i in target
                for j in range(offsets[i], offsets[i] + self.widths[i])
            ]
            with torch.inference_mode(False):
                indices = torch.tensor(indices, dtype=torch.int64, device=x.device)
            self._put(self._repack_indices, key, indices)
        else:
            self._repack_indices.move_to_end(key)
        return x.index_select(1, self._repack_indices[key])

    def forward(self, x, active=None):
        active = self._ids(range(len(self.widths)) if active is None else active)
        inputs = self.widths if self.input_widths is None else self.input_widths
        layout = active
        expected_width = sum(inputs[i] for i in layout)
        parameter = (
            self.input_layer.weight
            if self.input_layer is not None
            else self.blocks[0].weight
        )
        if (
            tuple(x.shape) != (self.batch, expected_width)
            or not x.is_contiguous()
            or x.device != parameter.device
            or x.dtype not in (torch.float16, torch.bfloat16)
        ):
            raise ValueError(
                'Expected contiguous FP16/BF16 input matching the routed layout'
            )
        selected = [
            tuple(i for i in active if self.depths[i] > layer)
            for layer in range(len(self.blocks))
        ]
        calls = [] if self.input_layer is None else [(self.input_layer, active)]
        calls += [(block, ids) for block, ids in zip(self.blocks, selected) if ids]
        count = sum(
            self.training and any(0 < block.dropout[i] < 1 for i in ids)
            for block, ids in calls
        )
        seeds = iter(
            torch.randint(
                2**63 - 1, (count,), device=x.device, dtype=torch.int64
            ).unbind()
            if count
            else ()
        )
        for block, ids in calls:
            stochastic = self.training and any(0 < block.dropout[i] < 1 for i in ids)
            seed = next(seeds) if stochastic else None
            x = block(x, active, seed, participating=ids)
        return x

    def pack(self, padded, active=None):
        ids = self._ids(range(len(self.widths)) if active is None else active)
        widths = self.widths if self.input_widths is None else self.input_widths
        if tuple(padded.shape) != (len(ids), self.batch, max(widths[i] for i in ids)):
            raise ValueError('Unexpected padded input shape')
        return torch.cat([padded[j, :, : widths[i]] for j, i in enumerate(ids)], dim=1)

    def unpack(self, compact, active=None):
        ids = self._ids(range(len(self.widths)) if active is None else active)
        widths = tuple(self.widths[i] for i in ids)
        if tuple(compact.shape) != (self.batch, sum(widths)):
            raise ValueError('Unexpected compact output shape')
        output = compact.new_zeros(len(ids), self.batch, max(widths))
        for j, part in enumerate(compact.split(widths, dim=1)):
            output[j, :, : widths[j]] = part
        return output
