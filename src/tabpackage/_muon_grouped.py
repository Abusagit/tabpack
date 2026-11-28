"""Experimental BF16 backend; consume output before the next workspace call.

The output can be strided when internal columns are aligned.
Wide first layers use an out-of-place cuBLASLt BX epilogue when available;
Gram and polynomial products continue to use grouped GEMM.
"""

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_extension = None


def extension():
    global _extension
    if _extension is None:
        import ninja

        os.environ['PATH'] = ninja.BIN_DIR + os.pathsep + os.environ.get('PATH', '')
        # A stopped compiler must not stall another experiment's cache lock.
        build = Path(
            os.environ.get('TABPACK_GROUPED_BUILD_ROOT', '../output/extensions')
        ) / str(os.getpid())
        build.mkdir(parents=True, exist_ok=True)
        cuda = Path(os.environ.get('CUDA_HOME', '/usr/local/cuda'))
        _extension = load(
            name='tabpack_grouped_bf16',
            build_directory=str(build.resolve()),
            sources=[str(Path(__file__).with_name('csrc') / 'muon_grouped.cpp')],
            extra_include_paths=[str(cuda / 'include')],
            extra_cflags=['-O2'],
            extra_ldflags=['-ldl'],
            with_cuda=True,
            verbose=True,
        )
    return _extension


class GroupedNewtonSchulz:
    def __init__(self):
        self.key = None
        self.plan = None
        self.padding_mask = None
        self.widths = None
        self.members = None
        self.mask_key = None

    def validate_padding(self, update, widths, *, rectangular=False):
        key = (
            id(widths),
            widths._version,
            tuple(update.shape),
            update.device,
            rectangular,
        )
        if key != self.mask_key:
            rows = torch.arange(update.shape[-2], device=update.device)
            self.padding_mask = rows[None, :, None] >= widths[:, None, None]
            if not rectangular:
                cols = torch.arange(update.shape[-1], device=update.device)
                self.padding_mask = self.padding_mask | (
                    cols[None, None, :] >= widths[:, None, None]
                )
            self.mask_key = key
            self.widths = widths
        # This synchronizing guard must be included in performance measurements.
        if torch.any((update != 0) & self.padding_mask).item():
            raise ValueError(
                'grouped Muon requires zero update padding; '
                'refusing to discard nonzero values'
            )

    def __call__(
        self,
        update,
        widths,
        steps=5,
        active_pack_idx=None,
        *,
        rectangular=False,
        _padding_checked=False,
    ):
        if not _padding_checked:
            self.validate_padding(update, widths, rectangular=rectangular)
        # Tensor identity/version handles replacement, removal, and load_state_dict.
        # Workspaces own their addresses; changing grad/momentum pointers is harmless.
        members = None if active_pack_idx is None else tuple(active_pack_idx.tolist())
        wide_strategy = os.environ.get('TABPACK_MUON_BX_BACKEND', 'auto')
        key = (
            wide_strategy,
            rectangular,
            id(widths),
            widths._version,
            tuple(update.shape),
            update.device,
            torch.cuda.current_stream(update.device).cuda_stream,
        )
        if key != self.key:
            values = widths.tolist()
            # Initialize Lt cases for every width, so later member changes only
            # rebuild small pointer lists and retain all large GPU allocations.
            indices = list(range(len(values)))
            self.plan = extension().Plan(
                update,
                values,
                indices,
                rectangular,
                wide_strategy,
            )
            self.key = key
            self.members = None
            self.widths = widths
        if members != self.members:
            indices = list(range(len(widths))) if members is None else list(members)
            values = widths.tolist()
            self.plan.set_members([values[i] for i in indices], indices)
            self.members = members
        x = update.bfloat16()
        # Normalize the original matrix before C++ adds internal zero columns.
        x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        return self.plan.run(x, steps)
