"""Paired torch.compile baselines for the unmodified resident-data TabPack runner."""

import argparse
import dataclasses
import hashlib
import json
import os
import shutil
import time
import traceback
from pathlib import Path

import torch
from dev.pepilica.scripts.benchmark_ragged_plugin import collect, error, group_error, measure_paired
from torch._dynamo.utils import counters

from tabpack_experimental.benchmark import (
    Config,
    environment,
    gpu_status,
    make_runner,
    summarize_trace,
)
from tabpackage.nn import DropoutPack


def counter_snapshot():
    return {group: dict(values) for group, values in counters.items()}


def counter_delta(before, after):
    return {
        group: {
            key: value - before.get(group, {}).get(key, 0)
            for key, value in values.items()
            if value != before.get(group, {}).get(key, 0)
        }
        for group, values in after.items()
        if any(
            value != before.get(group, {}).get(key, 0) for key, value in values.items()
        )
    }


def copy_tree(destination, source):
    if isinstance(destination, torch.Tensor):
        if not isinstance(source, torch.Tensor):
            raise TypeError('Optimizer state type changed')
        destination.copy_(source)
    elif isinstance(destination, dict):
        if destination.keys() != source.keys():
            raise AssertionError('Optimizer state keys changed')
        for key in destination:
            destination[key] = copy_tree(destination[key], source[key])
    elif isinstance(destination, list | tuple):
        if len(destination) != len(source):
            raise AssertionError('Optimizer group length changed')
        values = [
            copy_tree(current, saved) for current, saved in zip(destination, source)
        ]
        if isinstance(destination, tuple):
            return tuple(values)
        destination[:] = values
    else:
        return source
    return destination


class StableRestore:
    """Restore tensor values without replacing optimizer objects guarded by Dynamo."""

    def __init__(self, runner):
        self.runner = runner
        self.original = runner.restore

    @torch.no_grad()
    def __call__(self, snapshot):
        runner = self.runner
        runner.optimizer.zero_grad()
        parameters = dict(runner.model.named_parameters())
        for name, current in runner.model.state_dict().items():
            saved = snapshot['model'][name].to(current.device)
            if name in parameters:
                current.copy_(saved)
            elif not torch.equal(current, saved):
                raise RuntimeError('Static benchmark buffers changed between samples')
        indices = runner.model.backbone._get_active_local_pack_idx_by_block(None)
        for block, selection in zip(runner.model.backbone.blocks, indices):
            backend = getattr(block.linear, '_ragged_linear_backend', None)
            if backend is not None and (selection is None or selection.numel()):
                backend.prepare(runner.config.batch_size, selection)
        backend = getattr(runner.model.output, '_ragged_linear_backend', None)
        if backend is not None:
            backend.prepare(runner.config.batch_size)
        adapter = getattr(runner.optimizer, '_ragged_muon_adapter', None)
        if adapter is not None:
            adapter.refresh_metadata()
        copy_tree(runner.optimizer.state_dict(), snapshot['optimizer'])
        torch.set_rng_state(snapshot['rng_cpu'])
        torch.cuda.set_rng_state(snapshot['rng_cuda'], runner.device)
        runner.batch_cursor = snapshot['batch_cursor']
        runner.batch_generator.set_state(snapshot['batch_rng'])
        runner.sync()


class Variants:
    def __init__(self, runner, include_reduce_overhead):
        self.runner = runner
        self.forward = runner.forward
        self.optimizer_step = runner.optimizer.step
        self.step = runner.step
        compiled_forward = torch.compile(self.forward)
        compiled_optimizer = torch.compile(self.optimizer_step)
        self.calls = {
            'native': (self.forward, self.optimizer_step),
            'compile_model': (compiled_forward, self.optimizer_step),
            'compile_optimizer': (self.forward, compiled_optimizer),
            'compile_both': (compiled_forward, compiled_optimizer),
        }
        if include_reduce_overhead:
            self.calls['compile_both_reduce_overhead'] = (
                torch.compile(self.forward, mode='reduce-overhead'),
                torch.compile(self.optimizer_step, mode='reduce-overhead'),
            )

    def select(self, _, name):
        self.runner.forward, self.runner.optimizer.step = self.calls[name]
        self.runner.step = (
            self.marked_step if name.endswith('reduce_overhead') else self.step
        )

    def marked_step(self, scope=None):
        torch.compiler.cudagraph_mark_step_begin()
        self.step(scope)


def deterministic_check(runner, snapshot, variants):
    dropouts = [m for m in runner.model.modules() if isinstance(m, DropoutPack)]
    training = [module.training for module in dropouts]
    for module in dropouts:
        module.train(False)
    variants.select(None, 'native')
    runner.restore(snapshot)
    expected = []
    for _ in range(3):
        runner.step()
        runner.check_finite()
        expected.append(collect(runner))
    result = {}
    try:
        for name in variants.calls:
            if name == 'native':
                continue
            variants.select(None, name)
            runner.restore(snapshot)
            records = []
            for wanted in expected:
                runner.step()
                runner.check_finite()
                actual = collect(runner)
                torch.testing.assert_close(
                    actual['loss'], wanted['loss'], rtol=0.003, atol=0.003
                )
                gradients = group_error(actual['gradients'], wanted['gradients'])
                parameters = group_error(actual['parameters'], wanted['parameters'])
                if gradients['relative_l2'] > 0.05 or parameters['relative_l2'] > 0.01:
                    raise AssertionError((name, gradients, parameters))
                records.append(
                    {
                        'loss': error(actual['loss'], wanted['loss']),
                        'gradient': gradients,
                        'parameter': parameters,
                    }
                )
            result[name] = records
            print(f'Deterministic three-update check passed: {name}', flush=True)
    finally:
        for module, enabled in zip(dropouts, training):
            module.train(enabled)
    return result


def validate_replay(runner, snapshot, variants):
    result = {}
    for name in variants.calls:
        variants.select(None, name)
        records = []
        for _ in range(2):
            runner.restore(snapshot)
            runner.step()
            runner.check_finite()
            records.append(collect(runner))
        for key in ('parameters', 'gradients'):
            for tensor in records[0][key]:
                torch.testing.assert_close(
                    records[0][key][tensor], records[1][key][tensor], rtol=0, atol=0
                )
        torch.testing.assert_close(
            records[0]['loss'], records[1]['loss'], rtol=0, atol=0
        )
        torch.testing.assert_close(records[0]['rng'], records[1]['rng'], rtol=0, atol=0)
        result[name] = {'exact_state_restored_replay': True}
        print(f'Training-mode replay check passed: {name}', flush=True)
    return result


def capture_profile(runner, snapshot, output):
    runner.restore(snapshot)
    for _ in range(3):
        runner.step()
    runner.sync()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as profiler:
        with torch.profiler.record_function('profiler/warmup'):
            runner.step()
        runner.sync()
        for _ in range(runner.config.profile_steps):
            runner.step(
                lambda phase: torch.profiler.record_function(f'tabpack/{phase}')
            )
        runner.sync()
    path = output / 'trace.json'
    profiler.export_chrome_trace(str(path))
    (output / 'operators.txt').write_text(
        profiler.key_averages().table(sort_by='self_cuda_time_total', row_limit=100)
    )
    report = summarize_trace(
        json.loads(path.read_text()), runner.config.profile_steps, True, runner.phases
    )
    (output / 'trace-summary.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('dev/pepilica/configs/churn.json'),
    )
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--reduce-overhead', action='store_true')
    parser.add_argument('--no-profile', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    raw = json.loads(args.config.read_text())
    for key in ('widths', 'cat_cardinalities'):
        if raw.get(key) is not None:
            raw[key] = tuple(raw[key])
    config = dataclasses.replace(
        Config(**raw), steps=args.steps, repeats=args.repeats, warmup=args.warmup
    )
    torch.set_num_threads(2)
    torch._dynamo.config.suppress_errors = False
    runner = make_runner(config)
    for _ in range(config.warmup):
        runner.step()
    runner.sync()
    snapshot = runner.snapshot()
    runner.restore = StableRestore(runner)
    variants = Variants(runner, args.reduce_overhead)
    report = {
        'environment': environment(args.output),
        'affinity': sorted(os.sched_getaffinity(0)),
        'config': dataclasses.asdict(config),
        'resolved_model': runner.model_config,
        'resolved_optimizer': runner.optimizer_config,
        'dataset': runner.dataset_metadata,
        'scope': runner.scope,
        'protocol': {
            'model_compile_boundary': (
                'Runner.forward: model, autocast, output conversion'
            ),
            'optimizer_compile_boundary': (
                'Original optimizer.step, including Muon and AdamW'
            ),
            'loss_and_batch_gather': (
                'Eager in every variant, included in complete-step times'
            ),
            'custom_kernels': 'None; no ragged adapters installed',
            'correctness': (
                'Three updates with only dropout disabled; original BF16 tolerances'
            ),
            'dropout': (
                'Original heterogeneous training dropout enabled for all timing. '
                'Compiler RNG need not match eager masks; '
                'replay checked within each variant.'
            ),
            'state_restoration': (
                'In-place tensor copies preserve optimizer objects; model, optimizer, '
                'batch cursor and CPU/CUDA RNG restored outside each timed block.'
            ),
            'compilation': (
                'First use and warmup excluded from steady-state measurements'
            ),
            'limits': (
                'Static pack; no evaluation, preprocessing, '
                'convergence or quality comparison'
            ),
        },
    }
    source = Path(__file__)
    target = args.output / 'sources/scripts' / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    report['benchmark_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()

    def save():
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')

    save()
    report['first_use'] = {}
    report['unsupported_variants'] = {}
    for name in list(variants.calls):
        variants.select(None, name)
        runner.restore(snapshot)
        before = counter_snapshot()
        start = time.perf_counter()
        try:
            runner.step()
            runner.sync()
            first = time.perf_counter() - start
            for _ in range(config.warmup):
                runner.step()
            runner.sync()
        except RuntimeError:
            if not name.endswith('reduce_overhead'):
                raise
            report['unsupported_variants'][name] = traceback.format_exc()
            del variants.calls[name]
            print(f'Unsupported optional mode: {name}', flush=True)
            save()
            continue
        report['first_use'][name] = {
            'first_complete_step_seconds': first,
            'compiler_counters': counter_delta(before, counter_snapshot()),
        }
        print(f'First use: {name} {first:.3f}s', flush=True)
        save()

    report['correctness'] = deterministic_check(runner, snapshot, variants)
    save()
    report['training_replay'] = validate_replay(runner, snapshot, variants)
    for name in variants.calls:
        variants.select(None, name)
        runner.restore(snapshot)
        for _ in range(config.warmup):
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
    report['compiler_counters'] = counter_snapshot()
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
    for name, timing in report['timings']['variants'].items():
        print(
            f'{name}: {timing["wall_ms_per_step"]["median"]:.3f} ms, '
            f'paired speedup {timing["paired_speedup"]["median"]:.3f}x',
            flush=True,
        )


if __name__ == '__main__':
    main()
