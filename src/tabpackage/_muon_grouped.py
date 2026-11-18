"""Experimental BF16 backend; consume output before the next workspace call."""

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_extension = None


def extension():
    global _extension
    if _extension is None:
        import ninja

        os.environ["PATH"] = ninja.BIN_DIR + os.pathsep + os.environ.get("PATH", "")
        cuda = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
        _extension = load(
            name="tabpack_grouped_bf16",
            sources=[
                str(
                    Path(__file__).resolve().parents[2]
                    / "dev/veleboks/grouped_gemm/grouped.cpp"
                )
            ],
            extra_include_paths=[str(cuda / "include")],
            extra_cflags=["-O2"],
            extra_ldflags=["-ldl"],
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

    def __call__(self, update, widths, steps=5, active_pack_idx=None):
        # Tensor identity/version handles replacement, removal, and load_state_dict.
        # Workspaces own their addresses; changing grad/momentum pointers is harmless.
        members = None if active_pack_idx is None else tuple(active_pack_idx.tolist())
        key = (
            id(widths),
            widths._version,
            tuple(update.shape),
            update.device,
            torch.cuda.current_stream(update.device).cuda_stream,
            members,
        )
        if key != self.key:
            values = widths.tolist()
            indices = list(range(len(values))) if members is None else list(members)
            self.plan = extension().Plan(update, [values[i] for i in indices], indices)
            axis = torch.arange(update.shape[-1], device=update.device)
            self.padding_mask = (axis[None, :, None] >= widths[:, None, None]) | (
                axis[None, None, :] >= widths[:, None, None]
            )
            self.key = key
            self.widths = widths  # Keep identity alive until the plan is invalidated.
        # Never silently discard nonzero padding. This strict experimental guard
        # synchronizes once per invocation and its cost belongs in all benchmarks.
        if torch.any((update != 0) & self.padding_mask).item():
            raise ValueError(
                "grouped Muon requires zero update padding; "
                "refusing to discard nonzero values"
            )
        x = update.bfloat16()
        x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        return self.plan.run(x, steps)
