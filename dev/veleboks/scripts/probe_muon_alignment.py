"""Isolate column-stride alignment in rectangular Muon; no model training."""

import argparse
import gc
import json
from functools import partial
from pathlib import Path

import torch

from tabpackage._muon_grouped import GroupedNewtonSchulz, extension
from tabpackage.optim import _zeropower_via_newtonschulz5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--widths", type=int, nargs="+", required=True)
    parser.add_argument("--columns", type=int, nargs="+", default=[20061, 31506])
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    out = Path("../output")
    out.mkdir(exist_ok=True)
    extension()
    torch.manual_seed(0)
    widths = torch.tensor(args.widths, device="cuda")
    rows = max(args.widths)
    report = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "widths": args.widths,
        "scope": (
            "Synthetic updates, real first-layer shapes; "
            "aligned case adds zero columns."
        ),
        "cases": [],
    }
    for columns in args.columns:
        base = torch.randn(len(args.widths), rows, columns, device="cuda")
        for i, width in enumerate(args.widths):
            base[i, width:] = 0
        reference = _zeropower_via_newtonschulz5(base, 5)
        for aligned in (False, True):
            n = ((columns + 7) // 8) * 8 if aligned else columns
            update = torch.zeros(len(args.widths), rows, n, device="cuda")
            update[:, :, :columns] = base
            runtime = GroupedNewtonSchulz()
            functions = {
                "native": partial(_zeropower_via_newtonschulz5, update, 5),
                "grouped": partial(runtime, update, widths, rectangular=True),
            }
            actual = functions["grouped"]().clone()
            error = (actual[:, :, :columns].float() - reference.float()).norm()
            relative = float(error / reference.float().norm().clamp_min(1e-12))
            assert relative < 0.03
            assert actual[:, :, columns:].count_nonzero() == 0
            samples = {v: [] for v in functions}
            for variant in ("native", "grouped", "grouped", "native"):
                fn = functions[variant]
                for _ in range(3):
                    fn()
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
                for _ in range(args.steps):
                    fn()
                end.record()
                end.synchronize()
                samples[variant].append(start.elapsed_time(end) / args.steps)
            label = f"n{columns}-input{n}-ld{runtime.plan.workspace_columns}"
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
            ) as profiler:
                for variant, fn in functions.items():
                    for _ in range(3):
                        with torch.profiler.record_function(f"probe/{variant}"):
                            fn()
                torch.cuda.synchronize()
            profiler.export_chrome_trace(str(out / f"{label}-trace.json"))
            case = {
                "input_columns": columns,
                "call_input_columns": n,
                "workspace_columns": runtime.plan.workspace_columns,
                "relative_l2_to_original_native": relative,
                "ms_per_ns_call": samples,
                "trace": f"{label}-trace.json",
            }
            report["cases"].append(case)
            print(case, flush=True)
            del profiler, actual, functions, runtime, update, fn
            gc.collect()
        del base, reference
        gc.collect()
    (out / "alignment.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
