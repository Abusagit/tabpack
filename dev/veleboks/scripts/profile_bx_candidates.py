"""Compare actual-width BX layouts/APIs, including an out-of-place Lt epilogue."""

import argparse
import gc
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

WIDTHS = [384, 512, 512, 64, 256, 256, 128, 256, 384, 512, 512, 384, 128, 384, 512, 256]


def extension(out):
    import ninja

    os.environ["PATH"] = ninja.BIN_DIR + os.pathsep + os.environ.get("PATH", "")
    build = Path(os.environ.setdefault("BX_PROBE_BUILD_DIR", str(out / "extension")))
    build.mkdir(parents=True, exist_ok=True)
    return load(
        name="tabpack_bx_probe",
        sources=[
            str(Path(__file__).resolve().parents[1] / "grouped_gemm/bx_probe.cpp")
        ],
        build_directory=str(build),
        extra_include_paths=[
            str(Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "include")
        ],
        extra_cflags=["-O2"],
        with_cuda=True,
        verbose=True,
    )


def tensors(columns, layout):
    torch.manual_seed(0)
    x = torch.randn(16, 512, columns, dtype=torch.bfloat16, device="cuda") * 0.05
    b = torch.randn(16, 512, 512, dtype=torch.bfloat16, device="cuda") * 0.01
    logical = 20061 if columns == 20064 else 31506
    x[:, :, logical:] = 0
    for i, w in enumerate(WIDTHS):
        x[i, w:] = 0
        b[i, w:] = 0
        b[i, :, w:] = 0
    if layout == "column":
        x = x.transpose(1, 2).contiguous().transpose(1, 2)
    return x, b


def measure(plan, mode, algo, steps):
    values = []
    for _ in range(2):
        for _ in range(3):
            plan.run(mode, algo)
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(steps):
            plan.run(mode, algo)
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) / steps)
    return values


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--columns", type=int, default=31512)
    p.add_argument("--layout", default="row", choices=["row", "column"])
    p.add_argument("--mode", default="grouped")
    p.add_argument("--algo", type=int, default=-1)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--counters", action="store_true")
    args = p.parse_args()
    out = Path("../output").resolve()
    out.mkdir(exist_ok=True)
    mod = extension(out)
    if args.worker:
        x, b = tensors(args.columns, args.layout)
        plan = mod.BXProbe(x, b, WIDTHS, args.layout == "column")
        plan.run(args.mode, args.algo)
        torch.cuda.synchronize()
        return
    result = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "widths": WIDTHS,
        "scope": (
            "Synthetic BF16 aX+BX using real first-layer widths. "
            "Ordinary APIs include Y.copy_(X); Lt reads C=X and writes D=Y directly."
        ),
        "cases": [],
    }
    for columns in [31506, 31512, 20064]:
        for layout in ["row", "column"]:
            x, b = tensors(columns, layout)
            plan = mod.BXProbe(x, b, WIDTHS, layout == "column")
            reference = torch.baddbmm(x, b, x, beta=3.4445)
            candidates = [
                (m, -1) for m in ["grouped", "coalesced", "batched", "single", "lt"]
            ]
            if columns == 31512:
                candidates += [("lt", i) for i in range(1, 8)]
                candidates += [("batched", i) for i in [99, 100, 101, 102, 103]]
            valid = []
            for mode, algo in candidates:
                item = {
                    "columns": columns,
                    "layout": layout,
                    "mode": mode,
                    "algo": algo,
                }
                try:
                    actual = plan.run(mode, algo)
                    relative = float(
                        (actual.float() - reference.float()).norm()
                        / reference.float().norm()
                    )
                    assert relative < 0.02
                    torch.testing.assert_close(actual, reference, rtol=0.03, atol=0.01)
                    item.update(
                        relative_l2=relative, ms=measure(plan, mode, algo, args.steps)
                    )
                    valid.append((mode, algo))
                except RuntimeError as e:
                    item["error"] = str(e).splitlines()[0]
                result["cases"].append(item)
                print(item, flush=True)
                (out / "bx-candidates.json").write_text(json.dumps(result, indent=2))
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
            ) as profiler:
                for mode, algo in valid:
                    with torch.profiler.record_function(f"bx/{mode}/{algo}"):
                        plan.run(mode, algo)
                torch.cuda.synchronize()
            profiler.export_chrome_trace(str(out / f"bx-{columns}-{layout}-trace.json"))
            del plan, x, b, reference, actual, profiler
            gc.collect()
    if args.counters:
        maps_cases = [
            c for c in result["cases"] if c["columns"] in (31506, 31512) and "ms" in c
        ]
        best = min(maps_cases, key=lambda c: sum(c["ms"]))
        configs = [
            {"columns": 31512, "layout": "row", "mode": "grouped", "algo": -1},
            best,
        ]
        result["counters"] = []
        for i, c in enumerate(configs):
            cmd = [
                "/usr/local/cuda/bin/ncu",
                "--target-processes",
                "all",
                "--clock-control",
                "none",
                "--cache-control",
                "none",
                "--section",
                "SpeedOfLight",
                "--section",
                "LaunchStats",
                "--section",
                "Occupancy",
                "--section",
                "MemoryWorkloadAnalysis",
                "--kernel-name",
                "regex:.*gemm.*",
                "--kernel-name-base",
                "demangled",
                "--launch-count",
                "1",
                "--export",
                str(out / f"ncu-{i}"),
                "--force-overwrite",
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--columns",
                str(c["columns"]),
                "--layout",
                c["layout"],
                "--mode",
                c["mode"],
                "--algo",
                str(c["algo"]),
            ]
            log = out / f"ncu-{i}.log"
            with log.open("w") as f:
                code = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
            result["counters"].append({"case": c, "exit_code": code, "log": str(log)})
            print("ncu", i, code, log.read_text()[-600:], flush=True)
        (out / "bx-candidates.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
