"""Run native/grouped pairs through common.run; each case has a separate process."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main():
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="../output")
    p.add_argument("--data-root", required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-epochs", type=int)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--profiles", action="store_true")
    p.add_argument("--profiles-only", action="store_true")
    p.add_argument("--n-models", type=int)
    p.add_argument("--tabred-n-models", type=int)
    p.add_argument("--after-suite", type=Path)
    p.add_argument("--wait-free-gpu", action="store_true")
    p.add_argument("--gpu-memory-limit-mib", type=int, default=1200)
    p.add_argument("--skip-values", nargs="+", type=int, choices=(0, 1), default=[0, 1])
    p.add_argument(
        "--dataset",
        nargs="+",
        default=["california", "maps-routing", "homecredit-default"],
    )
    args = p.parse_args()
    if args.after_suite:
        while not args.after_suite.is_file():
            time.sleep(10)
        previous = json.loads(args.after_suite.read_text())
        if previous["failures"]:
            raise RuntimeError(f"Prerequisite smoke failed: {previous['failures']}")
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    failures = []
    for dataset in args.dataset:
        cfg = json.loads(
            (
                ROOT / f"dev/veleboks/configs/{dataset}-diverse-embeddings.json"
            ).read_text()
        )
        cfg["data_root"] = os.path.expanduser(args.data_root)
        if args.n_models:
            cfg["n_models"] = args.n_models
        elif args.tabred_n_models and dataset != "california":
            cfg["n_models"] = args.tabred_n_models
        if args.max_epochs is not None:
            if args.max_epochs < 1:
                p.error("--max-epochs must be positive")
            cfg["max_epochs"] = args.max_epochs
        for seed in [0] if args.smoke else args.seeds:
            cfg["seed"] = seed
            for skip_value in args.skip_values:
                skip = bool(skip_value)
                cfg["muon_skip_zero_updates"] = skip
                for variant in ("native", "grouped"):
                    if args.wait_free_gpu:
                        import torch

                        uuid = str(torch.cuda.get_device_properties(0).uuid)
                        if not uuid.startswith("GPU-"):
                            uuid = "GPU-" + uuid
                        while True:
                            status = subprocess.check_output(
                                [
                                    "nvidia-smi",
                                    "--id=" + uuid,
                                    "--query-gpu=memory.used,utilization.gpu",
                                    "--format=csv,noheader,nounits",
                                ],
                                text=True,
                            )
                            memory, utilization = map(int, status.strip().split(","))
                            if memory < args.gpu_memory_limit_mib and utilization < 5:
                                break
                            print(
                                f"Waiting for {uuid}: {memory} MiB, {utilization}%",
                                flush=True,
                            )
                            time.sleep(30)
                    name = f"{dataset}-seed{seed}-skip{int(skip)}-{variant}"
                    case = out / name
                    config = out / f"{name}.json"
                    config.write_text(json.dumps(cfg, indent=2))
                    cmd = [
                        sys.executable,
                        "-m",
                        "common.run",
                        str(config),
                        "--variant",
                        variant,
                        "--profile",
                        "none",
                        "--out",
                        str(case),
                    ]
                    if args.smoke:
                        cmd += ["--max-epochs", "2"]
                    with (out / f"{name}.log").open("w") as log:
                        import torch

                        uuid = str(torch.cuda.get_device_properties(0).uuid)
                        uuid = uuid if uuid.startswith("GPU-") else "GPU-" + uuid
                        telemetry = (out / f"{name}-gpu.csv").open("w")
                        monitor = subprocess.Popen(
                            [
                                "nvidia-smi",
                                "--id=" + uuid,
                                "--query-gpu=timestamp,uuid,utilization.gpu,memory.used,clocks.sm,power.draw,temperature.gpu",
                                "--format=csv,noheader,nounits",
                                "--loop-ms=500",
                            ],
                            stdout=telemetry,
                        )
                        try:
                            code = (
                                0
                                if args.profiles_only
                                else subprocess.call(
                                    cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
                                )
                            )
                        finally:
                            monitor.terminate()
                            monitor.wait()
                            telemetry.close()
                    (out / f"{name}.exit-code").write_text(str(code))
                    print(name, code, flush=True)
                    if (
                        not code
                        and (args.profiles or args.profiles_only)
                        and seed == args.seeds[0]
                    ):
                        for mode in ("timing", "torch"):
                            profile_cmd = [
                                sys.executable,
                                "-m",
                                "common.run",
                                str(config),
                                "--variant",
                                variant,
                                "--profile",
                                mode,
                                "--epoch",
                                "3",
                                "--steps",
                                "10",
                                "--out",
                                str(out / (name + "-" + mode)),
                            ]
                            with (out / f"{name}-{mode}.log").open("w") as log:
                                profile_code = subprocess.call(
                                    profile_cmd,
                                    cwd=ROOT,
                                    stdout=log,
                                    stderr=subprocess.STDOUT,
                                )
                            if profile_code:
                                failures.append(name + "-" + mode)
                    if code:
                        failures.append(name)
                        print((out / f"{name}.log").read_text()[-5000:], flush=True)
    (out / "suite.json").write_text(
        json.dumps({"failures": failures, "smoke": args.smoke}, indent=2)
    )
    return bool(failures)


if __name__ == "__main__":
    sys.exit(main())
