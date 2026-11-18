"""Shared grouped-kernel layout and dtype utilities."""

import torch
import triton
import triton.language as tl


def resolve_matmul_dtypes(
    lhs: torch.Tensor, rhs: torch.Tensor, *, autocast: bool = False
) -> tuple[torch.dtype, tl.dtype, str]:
    """Resolve operand/output dtype, accumulator dtype, and FP32 dot precision.

    Match matmul's equal-dtype contract outside autocast. The caller performs
    any required casts; this function only reads metadata and backend settings.
    """
    if lhs.device != rhs.device or lhs.device.type != 'cuda':
        raise ValueError('Triton matmul operands must be on the same CUDA device')
    supported = (torch.float16, torch.bfloat16, torch.float32)
    if lhs.dtype not in supported or rhs.dtype not in supported:
        raise TypeError('Triton matmul supports only FP16, BF16, and FP32')
    if autocast and torch.is_autocast_enabled('cuda'):
        dtype = torch.get_autocast_dtype('cuda')
        if dtype not in (torch.float16, torch.bfloat16):
            raise TypeError('CUDA autocast must use FP16 or BF16')
    else:
        if lhs.dtype != rhs.dtype:
            raise TypeError('Triton matmul operands must have the same dtype')
        dtype = lhs.dtype
    precision = (
        'tf32'
        if dtype == torch.float32 and torch.backends.cuda.matmul.allow_tf32
        else 'ieee'
    )
    return dtype, tl.float32, precision


@triton.jit
def _locate_matrix(prefix_ptr, n_matrices, program_id, SEARCH_STEPS: tl.constexpr):
    """Find the matrix that owns program_id using a specialized search depth.

    prefix_ptr holds cumulative tile counts for n_matrices matrices, starting
    at zero. program_id is the flat tile index across the pack.
    The launcher supplies n_matrices.bit_length() as SEARCH_STEPS.
    """
    low, high = 0, n_matrices
    for _ in tl.static_range(0, SEARCH_STEPS):
        middle = (low + high) // 2
        take_right = tl.load(prefix_ptr + middle) <= program_id
        low = tl.where(take_right, middle + 1, low)
        high = tl.where(take_right, high, middle)
    return tl.cast(low - 1, tl.int32)
