"""Map grouped tile IDs to packed matrices."""

import triton
import triton.language as tl


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
