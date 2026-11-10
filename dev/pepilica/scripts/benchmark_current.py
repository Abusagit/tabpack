"""Original, compiled and ragged TabPack on a highly skewed width distribution."""

import argparse
import dataclasses
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import torch
from dev.pepilica.scripts.benchmark_compile import (
    StableRestore,
    capture_profile,
    counter_delta,
    counter_snapshot,
    deterministic_check,
    validate_replay,
)
from dev.pepilica.scripts.benchmark_ragged_plugin import measure_paired

from tabpack_experimental.benchmark import (
    Config,
    environment,
    gpu_status,
    make_runner,
    padding_report,
)
from tabpack_experimental.ragged_linear import install_ragged_linears
from tabpack_experimental.ragged_muon import install_ragged_muon


class SkewVariants:
    def __init__(self, runner):
        self.runner = runner
        self.forward = runner.forward
        self.optimizer_step = runner.optimizer.step
        self.linears = install_ragged_linears(
            runner.model, scope='all', schedule='k_desc', tiled=True
        )
        self.muon = install_ragged_muon(runner.optimizer, runner.model)
        self.compiled_forward = torch.compile(self.forward)
        self.compiled_optimizer_step = torch.compile(self.optimizer_step)
        self.calls = dict.fromkeys(
            ('native', 'compile_both', 'ragged_linear', 'ragged_muon', 'ragged_both')
        )
        self.select(None, 'native')

    def select(self, _, name):
        for backend in self.linears.values():
            backend.module.forward = (
                backend.forward
                if name in ('ragged_linear', 'ragged_both')
                else backend.original_forward
            )
        if name in ('ragged_muon', 'ragged_both'):
            self.muon.enable()
        else:
            self.muon.disable()
        self.runner.forward = (
            self.compiled_forward if name == 'compile_both' else self.forward
        )
        self.runner.optimizer.step = (
            self.compiled_optimizer_step
            if name == 'compile_both'
            else self.optimizer_step
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--no-profile', action='store_true')
    parser.add_argument('--memory-fraction', type=float)
    args = parser.parse_args()
    if args.memory_fraction is not None and not 0 < args.memory_fraction <= 1:
        parser.error('memory-fraction must be in (0, 1]')
    args.output.mkdir(parents=True, exist_ok=False)
    raw = json.loads(args.config.read_text())
    for key in ('widths', 'cat_cardinalities'):
        raw[key] = tuple(raw[key])
    config = dataclasses.replace(Config(**raw), steps=args.steps, repeats=args.repeats)
    torch.cuda.set_device(0)
    if args.memory_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch._dynamo.config.suppress_errors = False
    report = {
        'environment': environment(args.output),
        'affinity': sorted(os.sched_getaffinity(0)),
        'config': dataclasses.asdict(config),
        'padding': padding_report(config),
        'protocol': {
            'allocator_memory_fraction_limit': args.memory_fraction,
            'variants': (
                'Original padded model/optimizer, default compilation of original '
                'forward and optimizer.step, all-linear ragged adapter, Muon adapter, '
                'and both adapters. All retain padded FP32 parameters and state.'
            ),
            'timing': (
                'Randomized paired complete-step wall blocks and separate phase '
                'event blocks; same post-warmup model, optimizer, batches and RNG '
                'restored outside timing. All plans and compilation warmed.'
            ),
            'scope': (
                'Resident Churn training batches including gathers, original loss, '
                'backward and optimizer. No preprocessing/evaluation/convergence run. '
                'This is the real-model compatibility path, not CompactMLP.'
            ),
            'correctness': (
                'Three deterministic updates with only dropout disabled, followed '
                'by exact state-restored replay within every training-mode variant. '
                'Original sampled per-member dropout is enabled during timing.'
            ),
        },
    }
    for name in (
        'dev/pepilica/scripts/benchmark_current.py',
        'dev/pepilica/scripts/benchmark_compile.py',
        'dev/pepilica/scripts/benchmark_ragged_plugin.py',
        'dev/pepilica/src/tabpack_experimental/ragged.py',
        'dev/pepilica/src/tabpack_experimental/ragged_linear.py',
        'dev/pepilica/src/tabpack_experimental/ragged_muon.py',
        'dev/pepilica/src/tabpack_experimental/matrix_gemm.py',
        'dev/pepilica/src/tabpack_experimental/compact_mlp.py',
    ):
        path = Path(name)
        target = args.output / 'sources' / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        report['environment']['source_sha256'][name] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    shutil.copy2(args.config, args.output / 'config.json')

    def save():
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')

    save()
    runner = make_runner(config)
    for _ in range(config.warmup):
        runner.step()
    runner.check_finite()
    runner.sync()
    torch.cuda.reset_peak_memory_stats()
    runner.step()
    runner.sync()
    report['clean_native_peak_bytes'] = torch.cuda.max_memory_allocated()
    snapshot = runner.snapshot()
    runner.restore = StableRestore(runner)
    variants = SkewVariants(runner)
    report['dataset'] = runner.dataset_metadata
    report['resolved_model'] = runner.model_config
    report['resolved_optimizer'] = runner.optimizer_config
    report['muon_shapes'] = [p.shapes for p in variants.muon.plans.values()]
    report['warmup'] = {}
    for name in variants.calls:
        variants.select(None, name)
        runner.restore(snapshot)
        start = time.perf_counter()
        before = counter_snapshot()
        for _ in range(3):
            runner.step()
        runner.sync()
        report['warmup'][name] = {
            'seconds': time.perf_counter() - start,
            'compiler_counters': counter_delta(before, counter_snapshot()),
        }
        print(f'Prepared {name}: {report["warmup"][name]["seconds"]:.3f}s', flush=True)
        save()
    report['correctness'] = deterministic_check(runner, snapshot, variants)
    save()
    report['replay'] = validate_replay(runner, snapshot, variants)
    for name in variants.calls:
        variants.select(None, name)
        runner.restore(snapshot)
        for _ in range(3):
            runner.step()
    runner.sync()
    before = counter_snapshot()
    report['timings'] = measure_paired(
        runner,
        snapshot,
        None,
        list(variants.calls),
        repeats=args.repeats,
        steps=args.steps,
        selector=variants.select,
    )
    report['measurement_compiler_counters'] = counter_delta(before, counter_snapshot())
    if report['measurement_compiler_counters']:
        save()
        raise RuntimeError('Compiler activity occurred inside timed measurement blocks')
    report['compiler_counters'] = counter_snapshot()
    if any(backend.fallback_calls for backend in variants.linears.values()):
        raise RuntimeError('A ragged linear used its eager fallback')
    save()
    if not args.no_profile:
        report['profiles'] = {}
        for name in variants.calls:
            variants.select(None, name)
            output = args.output / name
            output.mkdir()
            report['profiles'][name] = capture_profile(runner, snapshot, output)
            save()
    report['gpu_after'] = gpu_status()
    save()
    for name, values in report['timings']['variants'].items():
        print(
            f'{name}: {values["wall_ms_per_step"]["median"]:.3f} ms, '
            f'paired speedup {values["paired_speedup"]["median"]:.3f}x',
            flush=True,
        )


if __name__ == '__main__':
    main()
