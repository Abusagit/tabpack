"""Bounded real-training phase profiles; exits after a warmed step window."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def worker(args):
    import torch

    import tabpackage
    import tabpackage._instrument as hooks
    from common import data, run
    from common.profiling import Instrumenter, Schedule
    from tabpackage import optim

    os.environ["TABPACK_MUON_BX_BACKEND"] = args.bx_backend

    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(args.config).read_text())
    run.setup_torch(cfg)
    arrays, meta = data.load(cfg)
    datasets = {
        s: {k: torch.as_tensor(v, device="cuda") for k, v in arrays[s].items()}
        for s in ("train", "val")
    }
    kwargs = run.VARIANTS[args.variant](run.build_train_kwargs(cfg, meta, datasets, 1))
    kwargs["print_progress"] = False
    if args.variant == "grouped":
        from tabpackage._muon_grouped import extension

        extension()  # build/load outside measurement and without a shared build lock

    class WindowFinished(Exception):
        pass

    class Bounded(Instrumenter):
        def step_end(self):
            super().step_end()
            if self.done:
                raise WindowFinished()

    inst = Bounded(
        args.mode,
        out,
        Schedule(20, args.steps, warmup=3),
        torch_options={
            "record_shapes": True,
            "profile_memory": False,
            "with_stack": False,
        },
    )
    groups = {}
    original_muon = optim.MuonAdamWPack._step_muon
    original_adamw = optim.MuonAdamWPack._step_adamw

    def muon(self, group):
        p = group["params"][0]
        key = id(p)
        if key not in groups:
            widths = group.get("muon_grouped_widths")
            shape = list(p.shape)
            groups[key] = {
                "index": len(groups),
                "shape": shape,
                "actual_widths": None if widths is None else widths.tolist(),
            }
        info = groups[key]
        label = (
            "muon/block" + str(info["index"]) + "/" + "x".join(map(str, info["shape"]))
        )
        with inst.phase(label):
            result = original_muon(self, group)
        if "wide_backend" not in info:
            runtime = self._grouped_runtimes.get(id(p))
            info["wide_backend"] = runtime.plan.wide_backend if runtime else "padded"
        return result

    def adamw(self, group):
        with inst.phase("adamw"):
            return original_adamw(self, group)

    optim.MuonAdamWPack._step_muon = muon
    optim.MuonAdamWPack._step_adamw = adamw
    uuid = str(torch.cuda.get_device_properties(0).uuid)
    uuid = uuid if uuid.startswith("GPU-") else "GPU-" + uuid
    with (out / "gpu.csv").open("w") as telemetry:
        monitor = subprocess.Popen(
            [
                "nvidia-smi",
                "--id=" + uuid,
                "--query-gpu=timestamp,uuid,utilization.gpu,memory.used,clocks.sm,clocks.mem,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
                "--loop-ms=200",
            ],
            stdout=telemetry,
        )
        hooks.set_instrumenter(inst)
        try:
            tabpackage.train(**kwargs)
            raise RuntimeError("Training ended before profile window")
        except WindowFinished:
            pass
        finally:
            hooks.set_instrumenter(None)
            monitor.terminate()
            monitor.wait()
    torch.cuda.synchronize()
    report = {
        "mode": args.mode,
        "variant": args.variant,
        "bx_backend": args.bx_backend,
        "environment": run.environment(),
        "config": cfg,
        "metadata": meta,
        "groups": list(groups.values()),
        "sampled_hyperparameters": run.sample_hyperparameters(cfg),
        "data_hashes": {
            s: {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in part.items()}
            for s, part in arrays.items()
        },
        "profile": inst.summary(),
        "limitations": (
            "Early first-epoch steady window; excludes validation/ensemble "
            "selection and convergence."
        ),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(args.variant, args.mode, report["profile"], flush=True)


def main():
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--config")
    p.add_argument("--variant", choices=["native", "grouped"])
    p.add_argument("--mode", choices=["timing", "torch"])
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--output", default="../output")
    p.add_argument("--data-root", default="~/tabpack-data/tabred-v0.1.2")
    p.add_argument(
        "--datasets", nargs="+", default=["maps-routing", "homecredit-default"]
    )
    p.add_argument("--timing-only", action="store_true")
    p.add_argument("--bx-backend", choices=["auto", "grouped", "lt"], default="auto")
    args = p.parse_args()
    if args.worker:
        worker(args)
        return
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    failures = []
    for dataset in args.datasets:
        cfg = json.loads(
            (
                ROOT / f"dev/veleboks/configs/{dataset}-diverse-embeddings.json"
            ).read_text()
        )
        cfg.update(
            n_models=cfg["n_models"] if dataset == "california" else 16,
            seed=0,
            muon_skip_zero_updates=False,
            data_root=os.path.expanduser(args.data_root),
        )
        config = out / f"{dataset}.json"
        config.write_text(json.dumps(cfg, indent=2))
        variants = ("native", "grouped", "grouped", "native")
        if not args.timing_only:
            variants += ("native", "grouped")
        for i, variant in enumerate(variants):
            mode = "timing" if i < 4 else "torch"
            name = f"{dataset}-{i}-{variant}-{mode}"
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--config",
                str(config),
                "--variant",
                variant,
                "--mode",
                mode,
                "--steps",
                "20" if mode == "timing" else "5",
                "--bx-backend",
                args.bx_backend,
                "--output",
                str(out / name),
            ]
            with (out / f"{name}.log").open("w") as log:
                code = subprocess.call(
                    cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
                )
            print(name, code, flush=True)
            if code:
                failures.append(name)
                print((out / f"{name}.log").read_text()[-4000:], flush=True)
                (out / "suite.json").write_text(
                    json.dumps({"failures": failures, "incomplete": True}, indent=2)
                )
                raise RuntimeError(f"Profile case failed: {name}")
    (out / "suite.json").write_text(json.dumps({"failures": failures}, indent=2))
    if failures:
        raise RuntimeError(failures)


if __name__ == "__main__":
    main()
