"""Test one padded Lt batch, exact-shape stride runs, sorted batches and graphs."""

import argparse
import gc
import json
import subprocess
import time
from functools import partial
from pathlib import Path

import torch
from profile_bx_candidates import WIDTHS, extension, tensors


def timed(fn, steps):
    samples = []
    for _ in range(2):
        for _ in range(3):
            fn()
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(steps):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / steps)
    return samples


def wait_free(timeout):
    uuid = str(torch.cuda.get_device_properties(0).uuid)
    if not uuid.startswith("GPU-"):
        uuid = "GPU-" + uuid
    deadline = time.monotonic() + timeout
    stable = 0
    while True:
        text = subprocess.check_output(
            [
                "nvidia-smi",
                "--id=" + uuid,
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        memory, utilization = map(int, text.strip().split(","))
        stable = stable + 1 if memory < 1600 and utilization < 5 else 0
        if stable >= 3:
            return uuid
        if time.monotonic() >= deadline:
            raise RuntimeError(f"GPU remained busy: {text.strip()}")
        print("Waiting for idle", uuid, text.strip(), flush=True)
        time.sleep(5)


def check(actual, expected):
    relative = float(
        (actual.float() - expected.float()).norm() / expected.float().norm()
    )
    assert relative < 0.02
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.01)
    return relative


def compute_and_unsort(plan, inverse):
    return plan.run("lt_runs").index_select(0, inverse)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-seconds", type=int, default=600)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    out = Path("../output").resolve()
    out.mkdir(exist_ok=True)
    mod = extension(out)  # CPU compilation can proceed while another job uses the GPU.
    uuid = wait_free(args.wait_seconds)
    result = {
        "gpu": torch.cuda.get_device_name(),
        "uuid": uuid,
        "torch": torch.__version__,
        "widths": WIDTHS,
        "steps_per_window": args.steps,
        "cases": [],
        "scope": (
            "Isolated BF16 aX+BX; same data and Lt separate C/D epilogue. "
            "Two timed windows per case."
        ),
    }
    order = torch.tensor(
        sorted(range(len(WIDTHS)), key=WIDTHS.__getitem__), device="cuda"
    )
    inverse = order.argsort()
    for columns in [31512, 20064]:
        x, b = tensors(columns, "row")
        expected = torch.baddbmm(x, b, x, beta=3.4445)
        plan = mod.BXProbe(x, b, WIDTHS, False)
        for mode, calls in [("lt", 16), ("lt_full", 1), ("lt_runs", 9)]:
            for algo in [-1] if mode == "lt" else [-1, 1, 2, 3]:
                item = {
                    "columns": columns,
                    "mode": mode,
                    "algo": algo,
                    "lt_calls": calls,
                    "packing_included": False,
                }
                try:
                    item["relative_l2"] = check(plan.run(mode, algo), expected)
                    item["ms"] = timed(partial(plan.run, mode, algo), args.steps)
                except RuntimeError as exc:
                    item["error"] = str(exc).splitlines()[0]
                result["cases"].append(item)
                print(item, flush=True)
        for mode in ["lt", "lt_full", "lt_runs"]:
            item = {"columns": columns, "mode": mode + "_graph", "algo": -1}
            try:
                for _ in range(3):
                    plan.run(mode)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = plan.run(mode)
                graph.replay()
                item["relative_l2"] = check(actual, expected)
                item["ms"] = timed(graph.replay, args.steps)
                del actual, graph
            except RuntimeError as exc:
                item["error"] = str(exc).splitlines()[0]
            result["cases"].append(item)
            print(item, flush=True)
        sx, sb = x.index_select(0, order), b.index_select(0, order)
        sorted_plan = mod.BXProbe(sx, sb, sorted(WIDTHS), False)
        sorted_ref = expected.index_select(0, order)
        for algo in [-1, 1, 2, 3]:
            item = {
                "columns": columns,
                "mode": "lt_sorted_runs",
                "algo": algo,
                "lt_calls": 5,
                "packing_included": False,
            }
            try:
                item["relative_l2"] = check(
                    sorted_plan.run("lt_runs", algo), sorted_ref
                )
                item["ms"] = timed(
                    partial(sorted_plan.run, "lt_runs", algo), args.steps
                )
            except RuntimeError as exc:
                item["error"] = str(exc).splitlines()[0]
            result["cases"].append(item)
            print(item, flush=True)
        result.setdefault("packing", []).append(
            {
                "columns": columns,
                "x_sort_ms": timed(partial(x.index_select, 0, order), args.steps),
                "b_sort_ms": timed(partial(b.index_select, 0, order), args.steps),
                "output_unsort_ms": timed(
                    partial(compute_and_unsort, sorted_plan, inverse),
                    args.steps,
                ),
                "output_unsort_scope": (
                    "Includes five-run Lt compute; subtract corresponding raw Lt "
                    "timing only approximately."
                ),
            }
        )
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        ) as profiler:
            for name, p, mode in [
                ("individual", plan, "lt"),
                ("full", plan, "lt_full"),
                ("runs", plan, "lt_runs"),
                ("sorted", sorted_plan, "lt_runs"),
            ]:
                with torch.profiler.record_function("batching/" + name):
                    p.run(mode)
            torch.cuda.synchronize()
        profiler.export_chrome_trace(str(out / f"lt-batching-{columns}-trace.json"))
        (out / "lt-batching.json").write_text(json.dumps(result, indent=2))
        del plan, sorted_plan, x, b, sx, sb, expected, sorted_ref, profiler, p
        gc.collect()
    print("Batching comparison complete", flush=True)


if __name__ == "__main__":
    main()
