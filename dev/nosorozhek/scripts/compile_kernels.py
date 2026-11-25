"""Compile PTX/SASS on the current CUDA device."""

from pathlib import Path

import torch
import triton.language as tl

from dev.nosorozhek.custom.kernels.linear import _packed_matmul
from dev.nosorozhek.custom.kernels.newton_schulz import _grouped_gemm


def main() -> None:
    output_dir = Path(__file__).resolve().parents[1] / 'compiled'
    output_dir.mkdir(exist_ok=True)
    for name, dtype in (('fp32', torch.float32), ('bf16', torch.bfloat16)):
        for precision in ('ieee', 'tf32'):
            options = {
                'SEARCH_STEPS': 2,
                'ACC_DTYPE': tl.float32,
                'INPUT_PRECISION': precision,
                'num_warps': 4,
                'num_stages': 2,
                'grid': (2,),
            }
            gemm = _grouped_gemm.warmup(
                dtype,
                dtype,
                dtype,
                dtype,
                torch.int32,
                torch.int32,
                2,
                # A matrix, row, column strides
                4096,
                64,
                1,
                # B matrix, row, column strides
                4096,
                64,
                1,
                # C matrix, row, column strides
                4096,
                64,
                1,
                # D matrix, row, column strides
                4096,
                64,
                1,
                ALPHA=1.0,
                BETA=0.0,
                BLOCK_M=64,
                BLOCK_N=64,
                BLOCK_K=64,
                **options,
            )
            packed = _packed_matmul.warmup(
                dtype,
                dtype,
                dtype,
                None,
                torch.int32,
                torch.int32,
                torch.int32,
                2,
                # Padded M, N, K
                64,
                64,
                64,
                # Block M, N, K
                64,
                64,
                64,
                False,  # TRANSPOSE_LHS
                True,  # TRANSPOSE_RHS
                dtype,  # bias
                HAS_BIAS=True,
                **options,
            )
            for kernel_name, kernel in (
                ('_grouped_gemm', gemm),
                ('_packed_matmul', packed),
            ):
                for extension in ('ptx', 'sass'):
                    assembly = kernel.asm[extension]
                    assert assembly.strip(), (kernel_name, name, precision, extension)
                    path = output_dir / f'{kernel_name}_{name}_{precision}.{extension}'
                    path.write_text(assembly)
                    print(path)


if __name__ == '__main__':
    main()
