"""Summarize bounded phase profiles and GPU trace ranges without double counting."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def analyze_trace(path):
    events = json.loads(path.read_text())["traceEvents"]
    ranges = [
        x
        for x in events
        if x.get("cat") == "gpu_user_annotation"
        and x.get("name", "").startswith("tabpack/")
    ]
    by_scope = defaultdict(list)
    for x in ranges:
        by_scope[x["name"]].append(x["dur"] / 1000)
    kernels = [x for x in events if x.get("cat") == "kernel"]
    # Link kernels to CPU launch sites. GPU annotation ranges alone miss
    # backward kernels launched from autograd's worker thread.
    cpu_ranges = [
        x
        for x in events
        if x.get("cat") == "user_annotation"
        and x.get("name", "").startswith("tabpack/")
    ]
    runtimes = {
        x.get("args", {}).get("correlation"): x
        for x in events
        if x.get("cat") in ("cuda_runtime", "cuda_driver")
    }

    def scope_for_launch(launch):
        if launch is None:
            return "unattributed"
        candidates = [
            a
            for a in cpu_ranges
            if a["ts"] <= launch["ts"] and a["ts"] + a["dur"] >= launch["ts"]
        ]
        same_thread = [a for a in candidates if a["tid"] == launch["tid"]]
        candidates = same_thread or candidates
        return (
            min(candidates, key=lambda a: a["dur"])["name"]
            if candidates
            else "unattributed"
        )

    totals = defaultdict(lambda: {"kernel_count": 0, "kernel_ms": 0.0})
    first_muon_kernels = defaultdict(lambda: {"count": 0, "gpu_ms": 0.0})
    first_muon_cpu_ops = defaultdict(lambda: {"count": 0, "gpu_ms": 0.0})
    cpu_ops = [x for x in events if x.get("cat") == "cpu_op"]
    for x in kernels:
        launch = runtimes.get(x.get("args", {}).get("correlation"))
        scope = scope_for_launch(launch)
        totals[scope]["kernel_count"] += 1
        totals[scope]["kernel_ms"] += x["dur"] / 1000
        if scope.startswith("tabpack/muon/block0/"):
            first_muon_kernels[x["name"]]["count"] += 1
            first_muon_kernels[x["name"]]["gpu_ms"] += x["dur"] / 1000
            parents = [
                op
                for op in cpu_ops
                if op["tid"] == launch["tid"]
                and op["ts"] <= launch["ts"] <= op["ts"] + op["dur"]
            ]
            name = (
                min(parents, key=lambda op: op["dur"])["name"]
                if parents
                else "direct_cuBLAS_or_CUDA"
            )
            first_muon_cpu_ops[name]["count"] += 1
            first_muon_cpu_ops[name]["gpu_ms"] += x["dur"] / 1000
    runtime_totals = defaultdict(lambda: {"calls": 0, "cpu_ms": 0.0})
    copies = defaultdict(lambda: {"count": 0, "gpu_ms": 0.0, "bytes": 0})
    for x in events:
        if x.get("cat") == "gpu_memcpy":
            scope = scope_for_launch(runtimes.get(x.get("args", {}).get("correlation")))
            key = scope + ":" + x["name"]
            copies[key]["count"] += 1
            copies[key]["gpu_ms"] += x["dur"] / 1000
            copies[key]["bytes"] += x.get("args", {}).get("bytes", 0)
    for x in runtimes.values():
        if any(t in x["name"].lower() for t in ("synchron", "memcpy")):
            key = scope_for_launch(x) + ":" + x["name"]
            runtime_totals[key]["calls"] += 1
            runtime_totals[key]["cpu_ms"] += x["dur"] / 1000
    top = defaultdict(lambda: [0, 0.0])
    for x in kernels:
        if (
            scope_for_launch(runtimes.get(x.get("args", {}).get("correlation")))
            == "unattributed"
        ):
            continue
        top[x["name"]][0] += 1
        top[x["name"]][1] += x["dur"] / 1000
    return {
        "gpu_annotation_range_means_incomplete_for_backward_ms": {
            k: statistics.fmean(v) for k, v in by_scope.items()
        },
        "exclusive_kernel_scopes": dict(totals),
        "first_muon_kernels": dict(first_muon_kernels),
        "first_muon_kernels_by_cpu_op": dict(first_muon_cpu_ops),
        "first_muon_steps": sum(
            x["name"].startswith("tabpack/muon/block0/") for x in cpu_ranges
        ),
        "synchronization_and_copies": dict(runtime_totals),
        "gpu_copies": dict(copies),
        "top_kernels": [
            {"name": k, "count": v[0], "total_ms": v[1]}
            for k, v in sorted(top.items(), key=lambda x: -x[1][1])[:15]
        ],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    args = p.parse_args()
    result = []
    for path in sorted(args.output.glob("*/report.json")):
        r = json.loads(path.read_text())
        record = {
            "case": path.parent.name,
            "variant": r["variant"],
            "mode": r["mode"],
            "groups": r["groups"],
        }
        phases = r["profile"].get("phase_ms", {})
        if phases:
            steps = r["profile"]["schedule"]["active"]
            means = {k: v["total"] / steps for k, v in phases.items()}
            main_phases = (
                "gather",
                "forward",
                "loss",
                "backward",
                "optimizer",
                "zero_grad",
            )
            total = sum(means.get(k, 0) for k in main_phases)
            eligible = sum(
                v
                for k, v in means.items()
                if k.startswith("muon/") and not k.startswith("muon/block0/")
            )
            record.update(
                ms_per_step=means,
                total_step_ms=total,
                eligible_hidden_ms=eligible,
                eligible_step_fraction=eligible / total,
            )
        rows = list(csv.reader((path.parent / "gpu.csv").open()))
        # All samples include warmup; keep raw CSV for exact time-window alignment.
        record["gpu_telemetry"] = {
            name: {
                "median": statistics.median(float(row[i]) for row in rows),
                "min": min(float(row[i]) for row in rows),
                "max": max(float(row[i]) for row in rows),
            }
            for i, name in [
                (2, "util_percent"),
                (4, "sm_clock_mhz"),
                (6, "power_w"),
                (7, "temperature_c"),
            ]
            if rows
        }
        if (path.parent / "trace.json").exists():
            record["trace"] = analyze_trace(path.parent / "trace.json")
        result.append(record)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
