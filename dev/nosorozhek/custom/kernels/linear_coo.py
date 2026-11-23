"""Grouped LinearPack GEMM with explicit COO-like tile coordinates."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd import Function

from .utils import resolve_matmul_dtypes


def _morton(row: int, col: int) -> int:
    key = 0
    for bit in range(max(row.bit_length(), col.bit_length())):
        key |= ((row >> bit) & 1) << (2 * bit)
        key |= ((col >> bit) & 1) << (2 * bit + 1)
    return key


def _hilbert(row: int, col: int, size: int) -> int:
    distance = 0
    step = size // 2
    while step:
        rx = int(bool(row & step))
        ry = int(bool(col & step))
        distance += step * step * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1:
                row, col = step - 1 - row, step - 1 - col
            row, col = col, row
        step //= 2
    return distance


def _tile_coordinates(
    rows: list[int],
    columns: list[int],
    block_m: int,
    block_n: int,
    swizzle: str,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    matrices = []
    row_tiles = []
    column_tiles = []
    for matrix, (row_count, column_count) in enumerate(zip(rows, columns, strict=True)):
        n_rows = math.ceil(row_count / block_m)
        n_columns = math.ceil(column_count / block_n)
        coordinates = [
            (row, column) for row in range(n_rows) for column in range(n_columns)
        ]
        if swizzle == 'z':
            coordinates.sort(key=lambda pair: _morton(*pair))
        elif swizzle == 'hilbert':
            size = 1 << (max(n_rows, n_columns, 1) - 1).bit_length()
            coordinates.sort(key=lambda pair: _hilbert(*pair, size))
        elif swizzle != 'row':
            raise ValueError(f'Unknown tile swizzle: {swizzle}')
        matrices.extend([matrix] * len(coordinates))
        row_tiles.extend(row for row, _ in coordinates)
        column_tiles.extend(column for _, column in coordinates)
    return tuple(
        torch.tensor(values, dtype=torch.int32, device=device)
        for values in (matrices, row_tiles, column_tiles)
    )


@triton.jit
def _coo_matmul(
    lhs_ptr,
    rhs_ptr,
    output_ptr,
    row_sizes_ptr,
    col_sizes_ptr,
    reduction_sizes_ptr,
    matrix_ids_ptr,
    row_tiles_ptr,
    column_tiles_ptr,
    padded_m,
    padded_n,
    padded_k,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    MAX_K: tl.constexpr,
    TRANSPOSE_LHS: tl.constexpr,
    TRANSPOSE_RHS: tl.constexpr,
    bias_ptr,
    HAS_BIAS: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    pid = tl.program_id(0)
    matrix = tl.load(matrix_ids_ptr + pid)
    tile_row = tl.load(row_tiles_ptr + pid)
    tile_column = tl.load(column_tiles_ptr + pid)
    m = tl.load(row_sizes_ptr + matrix)
    n = tl.load(col_sizes_ptr + matrix)
    k_size = tl.load(reduction_sizes_ptr + matrix)
    rows = tile_row * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = tile_column * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
    for k_block in tl.static_range(0, tl.cdiv(MAX_K, BLOCK_K)):
        k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        if TRANSPOSE_LHS:
            lhs = (
                lhs_ptr
                + matrix * padded_k * padded_m
                + k[None, :] * padded_m
                + rows[:, None]
            )
            lhs_mask = (k[None, :] < k_size) & (rows[:, None] < m)
        else:
            lhs = (
                lhs_ptr
                + matrix * padded_m * padded_k
                + rows[:, None] * padded_k
                + k[None, :]
            )
            lhs_mask = (rows[:, None] < m) & (k[None, :] < k_size)
        lhs = tl.load(lhs, mask=lhs_mask, other=0)
        if TRANSPOSE_RHS:
            rhs = tl.load(
                rhs_ptr
                + matrix * padded_n * padded_k
                + columns[:, None] * padded_k
                + k[None, :],
                mask=(columns[:, None] < n) & (k[None, :] < k_size),
                other=0,
            )
            rhs = tl.trans(rhs)
        else:
            rhs = tl.load(
                rhs_ptr
                + matrix * padded_k * padded_n
                + k[:, None] * padded_n
                + columns[None, :],
                mask=(k[:, None] < k_size) & (columns[None, :] < n),
                other=0,
            )
        acc += tl.dot(lhs, rhs, input_precision=INPUT_PRECISION)
    if HAS_BIAS:
        bias = tl.load(
            bias_ptr + matrix * padded_n + columns, mask=columns < n, other=0.0
        )
        acc += bias[None, :]
    tl.store(
        output_ptr
        + matrix * padded_m * padded_n
        + rows[:, None] * padded_n
        + columns[None, :],
        acc,
        mask=(rows[:, None] < m) & (columns[None, :] < n),
    )


def _launch(
    spec, lhs, rhs, output, bias, options, transpose_lhs, transpose_rhs, input_precision
):
    (
        row_sizes,
        col_sizes,
        reduction_sizes,
        matrix_ids,
        row_tiles,
        column_tiles,
    ) = spec
    block_m, block_n, block_k, num_warps, num_stages = options
    padded_m, padded_n = output.shape[-2:]
    padded_k = lhs.shape[-1] if not transpose_lhs else lhs.shape[-2]
    _coo_matmul[(len(matrix_ids),)](
        lhs,
        rhs,
        output,
        row_sizes,
        col_sizes,
        reduction_sizes,
        matrix_ids,
        row_tiles,
        column_tiles,
        padded_m,
        padded_n,
        padded_k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        MAX_K=padded_k,
        TRANSPOSE_LHS=transpose_lhs,
        TRANSPOSE_RHS=transpose_rhs,
        bias_ptr=bias,
        HAS_BIAS=bias.numel() > 0,
        ACC_DTYPE=tl.float32,
        INPUT_PRECISION=input_precision,
        num_warps=num_warps,
        num_stages=num_stages,
    )


class _COOLinearFunction(Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(ctx, x, weight, bias, specs, options):
        _, _, input_precision = resolve_matmul_dtypes(x, weight)
        x, weight, bias = x.contiguous(), weight.contiguous(), bias.contiguous()
        output = torch.zeros(
            x.shape[0], x.shape[1], weight.shape[-2], dtype=x.dtype, device=x.device
        )
        _launch(
            specs[0], x, weight, output, bias, options, False, True, input_precision
        )
        ctx.save_for_backward(x, weight, bias)
        ctx.specs = specs
        ctx.options = options
        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        _, grad_x_spec, grad_weight_spec = ctx.specs
        grad_output = grad_output.contiguous()
        grad_x = torch.zeros_like(x)
        grad_weight = torch.zeros_like(weight)
        _launch(
            grad_x_spec,
            grad_output,
            weight,
            grad_x,
            grad_output.new_empty(0),
            ctx.options,
            False,
            False,
            resolve_matmul_dtypes(grad_output, weight)[2],
        )
        _launch(
            grad_weight_spec,
            grad_output,
            x,
            grad_weight,
            grad_output.new_empty(0),
            ctx.options,
            True,
            False,
            resolve_matmul_dtypes(grad_output, x)[2],
        )
        if bias.numel():
            out_sizes = ctx.specs[0][1]
            mask = (
                torch.arange(weight.shape[-2], device=x.device)[None, None, :]
                < out_sizes[:, None, None]
            )
            grad_bias = (grad_output * mask.to(grad_output.dtype)).sum(1)
        else:
            grad_bias = None
        return grad_x, grad_weight, grad_bias, None, None


def grouped_linear_coo(x, weight, bias, specs, options):
    dtype, _, _ = resolve_matmul_dtypes(x, weight, autocast=True)
    if bias.numel():
        resolve_matmul_dtypes(x, bias, autocast=True)
    return _COOLinearFunction.apply(
        x.to(dtype), weight.to(dtype), bias.to(dtype), specs, options
    )


def make_specs(
    in_sizes: Tensor,
    out_sizes: Tensor,
    batch_size: int,
    block_m: int,
    block_n: int,
    swizzle: str,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...], tuple[Tensor, ...]]:
    device = in_sizes.device
    in_values = in_sizes.detach().cpu().tolist()
    out_values = out_sizes.detach().cpu().tolist()

    def sizes(rows: Iterable[int], columns: Iterable[int], reduction: Iterable[int]):
        rows, columns, reduction = map(list, (rows, columns, reduction))
        matrices, row_tiles, column_tiles = _tile_coordinates(
            rows, columns, block_m, block_n, swizzle, device
        )
        return (
            torch.tensor(rows, dtype=torch.int32, device=device),
            torch.tensor(columns, dtype=torch.int32, device=device),
            torch.tensor(reduction, dtype=torch.int32, device=device),
            matrices,
            row_tiles,
            column_tiles,
        )

    return (
        sizes([batch_size] * len(out_values), out_values, in_values),
        sizes([batch_size] * len(in_values), in_values, out_values),
        sizes(out_values, in_values, [batch_size] * len(out_values)),
    )
