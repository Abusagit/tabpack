"""Record compiled forward kernel resources without timing or profiling."""

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import torch
import triton

import tabpack_experimental.compact_mlp as compact
from tabpack_experimental import ragged


class Recorder:
    def __init__(self, function: Any, kind, records, label):
        self.function = function
        self.kind = kind
        self.records = records
        self.label = label

    def __getitem__(self, grid):
        launch = self.function[grid]

        def call(*args, **kwargs):
            kernel = launch(*args, **kwargs)
            self.records.append(
                {
                    **self.label,
                    'kind': self.kind,
                    'registers_per_thread': kernel.n_regs,
                    'spills': kernel.n_spills,
                    'shared_memory_bytes': kernel.metadata.shared,
                    'num_warps': kernel.metadata.num_warps,
                }
            )
            return kernel

        return call


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists')
    records, label = [], {}
    original_gemm = ragged._specialized_group
    original_pointwise = compact._pointwise
    ragged._specialized_group = Recorder(original_gemm, 'gemm', records, label)
    compact._pointwise = Recorder(original_pointwise, 'pointwise', records, label)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    try:
        for workload, widths in (
            ('uniform', (256,) * 16),
            ('vocabulary', ragged.HIDDEN_SIZES),
        ):
            for variant, fused, scalar, p in (
                ('current_no_dropout', False, False, 0.0),
                ('epilogue_no_dropout', True, False, 0.0),
                ('current_dropout', False, False, 0.2),
                ('epilogue_dropout', True, False, 0.2),
                ('epilogue_scalar_dropout', True, True, 0.2),
            ):
                label.update(workload=workload, variant=variant)
                model = compact.CompactMLP(
                    widths,
                    256,
                    1,
                    dropout=p,
                    fused=fused,
                    scalar_dropout=scalar,
                    tiled=True,
                    schedule='k_desc',
                )
                x = torch.ones(256, sum(widths), device='cuda', dtype=torch.bfloat16)
                with torch.no_grad():
                    model(x)
        torch.cuda.synchronize()
    finally:
        ragged._specialized_group = original_gemm
        compact._pointwise = original_pointwise
    report = {
        'scope': (
            'Compiled forward kernel resources for batch 256, one BF16 block, '
            'fixed hidden-width dispatch vocabulary, width-aware tiles and k_desc '
            'scheduling. Counts describe the entire compiled function, including '
            'unused vocabulary branches. Dropout probabilities are 0.2 for each '
            'member; their values are runtime data. No timing or measured '
            'occupancy is inferred from these counts.'
        ),
        'environment': {
            'python': platform.python_version(),
            'torch': torch.__version__,
            'triton': triton.__version__,
            'gpu': torch.cuda.get_device_name(),
        },
        'records': records,
    }
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
