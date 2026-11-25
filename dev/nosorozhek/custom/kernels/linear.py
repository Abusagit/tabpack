"""Grouped LinearPack CUDA kernels and autograd launcher."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd import Function

from .utils import _locate_matrix, resolve_matmul_dtypes


@triton.jit
def _packed_matmul(
    lhs_ptr,
    rhs_ptr,
    output_ptr,
    row_sizes_ptr,
    col_sizes_ptr,
    reduction_sizes_ptr,
    prefix_ptr,
    n_matrices,
    padded_m,
    padded_n,
    padded_k,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TRANSPOSE_LHS: tl.constexpr,
    TRANSPOSE_RHS: tl.constexpr,
    bias_ptr,
    HAS_BIAS: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    """Multiply packed matrices and optionally add column bias from bias_ptr.

    Let P be n_matrices and M, N, K be the padded dimensions. Each buffer
    is contiguous and row-major, with shapes and strides in elements as follows.

    - lhs_ptr: (P, M, K), strides (M*K, K, 1). With TRANSPOSE_LHS,
      (P, K, M), strides (K*M, M, 1).
    - rhs_ptr: (P, K, N), strides (K*N, N, 1). With TRANSPOSE_RHS,
      (P, N, K), strides (N*K, K, 1).
    - output_ptr: (P, M, N), strides (M*N, N, 1).
    - bias_ptr: (P, N), strides (N, 1), broadcast across output rows.

    Size arrays give valid M, N, and K. Missing row or reduction sizes use
    the padded extent. Each program finds its tile via prefix_ptr and
    accumulates over each matrix's valid K in BLOCK_K steps with padding masked.
    """
    pid = tl.program_id(0)
    matrix = _locate_matrix(prefix_ptr, n_matrices, pid, SEARCH_STEPS)
    local_id = pid - tl.load(prefix_ptr + matrix)
    n = tl.load(col_sizes_ptr + matrix)
    column_tiles = tl.cdiv(n, BLOCK_N)
    rows = (local_id // column_tiles) * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = (local_id % column_tiles) * BLOCK_N + tl.arange(0, BLOCK_N)
    m = padded_m if row_sizes_ptr is None else tl.load(row_sizes_ptr + matrix)
    k_size = (
        padded_k
        if reduction_sizes_ptr is None
        else tl.load(reduction_sizes_ptr + matrix)
    )
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
    for k_block in range(tl.cdiv(k_size, BLOCK_K)):
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
                other=0.0,
            )
            rhs = tl.trans(rhs)
        else:
            rhs = tl.load(
                rhs_ptr
                + matrix * padded_k * padded_n
                + k[:, None] * padded_n
                + columns[None, :],
                mask=(k[:, None] < k_size) & (columns[None, :] < n),
                other=0.0,
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


class _TritonLinearFunction(Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(
        ctx,
        x,
        weight,
        bias,
        layout: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, int, int, int],
        options: tuple[int, int, int, int, int],
    ) -> Tensor:
        (
            in_sizes,
            out_sizes,
            prefix_out,
            prefix_x,
            prefix_weight,
            total_out,
            total_x,
            total_weight,
        ) = layout
        block_m, block_n, block_k, num_warps, num_stages = options
        max_out, max_in = weight.shape[-2:]
        _, acc_dtype, input_precision = resolve_matmul_dtypes(x, weight)
        # The Triton kernel computes offsets for contiguous row-major buffers.
        x, weight, bias = x.contiguous(), weight.contiguous(), bias.contiguous()
        output = torch.zeros(
            x.shape[0], x.shape[1], max_out, dtype=x.dtype, device=x.device
        )
        _packed_matmul[(total_out,)](
            x,
            weight,
            output,
            None,
            out_sizes,
            in_sizes,
            prefix_out,
            x.shape[0],
            x.shape[1],
            max_out,
            max_in,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            SEARCH_STEPS=x.shape[0].bit_length(),
            INPUT_PRECISION=input_precision,
            ACC_DTYPE=acc_dtype,
            TRANSPOSE_LHS=False,
            TRANSPOSE_RHS=True,
            bias_ptr=bias,
            HAS_BIAS=bias.numel() > 0,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        ctx.save_for_backward(
            x, weight, bias, in_sizes, out_sizes, prefix_x, prefix_weight
        )
        ctx.max_in, ctx.max_out = max_in, max_out
        ctx.block_m, ctx.block_n, ctx.block_k = block_m, block_n, block_k
        ctx.num_warps, ctx.num_stages = num_warps, num_stages
        ctx.total_x, ctx.total_weight = total_x, total_weight
        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, grad_output):
        x, weight, bias, in_sizes, out_sizes, prefix_x, prefix_weight = (
            ctx.saved_tensors
        )
        _, acc_dtype, input_precision = resolve_matmul_dtypes(grad_output, weight)
        grad_output = grad_output.contiguous()
        grad_x = torch.zeros_like(x)
        grad_weight = torch.zeros_like(weight)
        _packed_matmul[(ctx.total_x,)](
            grad_output,
            weight,
            grad_x,
            None,
            in_sizes,
            out_sizes,
            prefix_x,
            x.shape[0],
            x.shape[1],
            ctx.max_in,
            ctx.max_out,
            BLOCK_M=ctx.block_m,
            BLOCK_N=ctx.block_n,
            BLOCK_K=ctx.block_k,
            SEARCH_STEPS=x.shape[0].bit_length(),
            INPUT_PRECISION=input_precision,
            ACC_DTYPE=acc_dtype,
            TRANSPOSE_LHS=False,
            TRANSPOSE_RHS=False,
            bias_ptr=None,
            HAS_BIAS=False,
            num_warps=ctx.num_warps,
            num_stages=ctx.num_stages,
        )
        _packed_matmul[(ctx.total_weight,)](
            grad_output,
            x,
            grad_weight,
            out_sizes,
            in_sizes,
            None,
            prefix_weight,
            x.shape[0],
            ctx.max_out,
            ctx.max_in,
            x.shape[1],
            BLOCK_M=ctx.block_m,
            BLOCK_N=ctx.block_n,
            BLOCK_K=ctx.block_k,
            SEARCH_STEPS=x.shape[0].bit_length(),
            INPUT_PRECISION=input_precision,
            ACC_DTYPE=acc_dtype,
            TRANSPOSE_LHS=True,
            TRANSPOSE_RHS=False,
            bias_ptr=None,
            HAS_BIAS=False,
            num_warps=ctx.num_warps,
            num_stages=ctx.num_stages,
        )
        if bias.numel():
            output_mask = (
                torch.arange(ctx.max_out, device=x.device)[None, None, :]
                < out_sizes[:, None, None]
            )
            grad_bias = (grad_output * output_mask.to(grad_output.dtype)).sum(1)
        else:
            grad_bias = None
        return grad_x, grad_weight, grad_bias, None, None


def grouped_linear(x, weight, bias, layout, options):
    dtype, _, _ = resolve_matmul_dtypes(x, weight, autocast=True)
    if bias.numel():
        resolve_matmul_dtypes(x, bias, autocast=True)
    # Cast outside the autograd Function so gradients return to the input dtypes.
    return _TritonLinearFunction.apply(
        x.to(dtype), weight.to(dtype), bias.to(dtype), layout, options
    )
