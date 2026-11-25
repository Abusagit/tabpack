"""Transfers at the native TabPack / active-only activation boundaries."""

from typing import Any

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


@triton.jit(
    do_not_specialize=['S0', 'S1', 'S2', 'ROW_STRIDE'],
    do_not_specialize_on_alignment=['S0', 'S1', 'S2', 'ROW_STRIDE'],
)
def _pack_matrix(
    S,
    D,
    Descriptors,
    Tasks,
    S0,
    S1,
    S2,
    ROW_STRIDE,
    BLOCK: tl.constexpr,
):
    task = tl.program_id(0)
    member = tl.load(Tasks + task * 2)
    chunk = tl.load(Tasks + task * 2 + 1)
    desc = Descriptors + member * 3
    base = tl.load(desc)
    rows = tl.load(desc + 1)
    columns = tl.load(desc + 2)
    index = chunk * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(
        S + member * S0 + (index // columns) * S1 + (index % columns) * S2,
        index < rows * columns,
        0,
    )
    stride = tl.where(ROW_STRIDE != 0, ROW_STRIDE, columns)
    destination = base + (index // columns) * stride + index % columns
    tl.store(D + destination, values, index < rows * columns)


@triton.jit(
    do_not_specialize=['ROWS', 'COLS', 'ROW_STRIDE'],
    do_not_specialize_on_alignment=['ROWS', 'COLS', 'ROW_STRIDE'],
)
def _unpack_matrix(
    S,
    D,
    Descriptors,
    ROWS,
    COLS,
    ROW_STRIDE,
    BLOCK: tl.constexpr,
):
    member = tl.program_id(1)
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    desc = Descriptors + member * 3
    base = tl.load(desc)
    rows = tl.load(desc + 1)
    columns = tl.load(desc + 2)
    r, c = index // COLS, index % COLS
    stride = tl.where(ROW_STRIDE != 0, ROW_STRIDE, columns)
    values = tl.load(
        S + base + r * stride + c,
        (index < ROWS * COLS) & (r < rows) & (c < columns),
        0,
    )
    tl.store(D + member * ROWS * COLS + index, values, index < ROWS * COLS)


class MatrixTransfer:
    def __init__(self, shapes, padded_shape, device):
        self.padded_shape = tuple(padded_shape)
        descs, tasks = [], []
        total = 0
        for member, (rows, columns) in enumerate(shapes):
            descs.append((total, rows, columns))
            tasks.extend(
                (member, chunk) for chunk in range(triton.cdiv(rows * columns, 1024))
            )
            total += rows * columns
        self.size = total
        self.compact_shape = (total,)
        self.row_stride = 0
        self.descriptors = torch.tensor(descs, dtype=torch.int64, device=device)
        self.tasks = torch.tensor(tasks, dtype=torch.int32, device=device)

    def pack(self, source, dtype):
        if tuple(source.shape) != self.padded_shape:
            raise ValueError('Packed-linear operand has an unexpected shape')
        output = torch.empty(self.compact_shape, dtype=dtype, device=source.device)
        kernel: Any = _pack_matrix
        kernel[(self.tasks.shape[0],)](
            source,
            output,
            self.descriptors,
            self.tasks,
            *source.stride(),
            self.row_stride,
            1024,
        )
        return output

    def unpack(self, source, dtype):
        output = torch.empty(self.padded_shape, dtype=dtype, device=source.device)
        _, rows, columns = self.padded_shape
        kernel: Any = _unpack_matrix
        kernel[(triton.cdiv(rows * columns, 1024), self.padded_shape[0])](
            source, output, self.descriptors, rows, columns, self.row_stride, 1024
        )
        return output


class ActivationTransfer(MatrixTransfer):
    """Transfer padded member matrices to/from batch-major feature columns."""

    def __init__(self, shapes, padded_shape, device):
        self.padded_shape = tuple(padded_shape)
        self.row_stride = sum(columns for _, columns in shapes)
        self.compact_shape = (padded_shape[1], self.row_stride)
        self.size = self.compact_shape[0] * self.row_stride
        descriptors, tasks = [], []
        offset = 0
        for member, (rows, columns) in enumerate(shapes):
            descriptors.append((offset, rows, columns))
            tasks.extend(
                (member, chunk) for chunk in range(triton.cdiv(rows * columns, 1024))
            )
            offset += columns
        self.descriptors = torch.tensor(descriptors, dtype=torch.int64, device=device)
        self.tasks = torch.tensor(tasks, dtype=torch.int32, device=device)


class _Pack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, source, transfer, dtype):
        ctx.transfer, ctx.dtype = transfer, source.dtype
        return transfer.pack(source, dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        return ctx.transfer.unpack(gradient.contiguous(), ctx.dtype), None, None


class _Unpack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, source, transfer):
        ctx.transfer = transfer
        return transfer.unpack(source, source.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        return ctx.transfer.pack(gradient, gradient.dtype), None
