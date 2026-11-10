"""Paired real-TabPack benchmarks for the opt-in LinearPack adapter."""

import argparse
import contextlib
import dataclasses
import gc
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path

import torch

from tabpack_experimental.benchmark import (
    Config,
    describe,
    environment,
    gpu_status,
    make_runner,
    padding_report,
    profile,
)
from tabpack_experimental.ragged_linear import (
    install_ragged_linears,
    remove_ragged_linears,
)


def is_hidden(name):
    return name.startswith('backbone.blocks.') and int(name.split('.')[2]) > 0


def select(backends, variant):
    for name, backend in backends.items():
        enabled = variant == 'ragged_all' or (
            variant == 'ragged_hidden' and is_hidden(name)
        )
        backend.module.forward = (
            backend.forward if enabled else backend.original_forward
        )


def error(actual, expected):
    a, b = actual.double(), expected.double()
    difference = a - b
    return {
        'relative_l2': (difference.norm() / b.norm().clamp_min(1e-20)).item(),
        'max_abs': difference.abs().max().item(),
    }


def collect(runner):
    assert runner.last_loss is not None
    parameters, gradients = {}, {}
    for name, parameter in runner.model.named_parameters():
        parameters[name] = parameter.detach().cpu().clone()
        if parameter.grad is not None:
            gradients[name] = parameter.grad.detach().cpu().clone()
    return {
        'loss': runner.last_loss.cpu().clone(),
        'parameters': parameters,
        'gradients': gradients,
        'rng': torch.cuda.get_rng_state(runner.device).clone(),
    }


def group_error(actual, expected):
    if actual.keys() != expected.keys():
        raise AssertionError('Different parameter/gradient keys')
    errors = {name: error(actual[name], expected[name]) for name in expected}
    difference = sum(
        ((actual[name].double() - expected[name].double()) ** 2).sum().item()
        for name in expected
    )
    norm = sum((value.double() ** 2).sum().item() for value in expected.values())
    worst = max(errors, key=lambda name: errors[name]['relative_l2'])
    return {
        'relative_l2': (difference / max(norm, 1e-40)) ** 0.5,
        'max_abs': max(value['max_abs'] for value in errors.values()),
        'worst_tensor': {'name': worst, **errors[worst]},
        'tensors': errors,
    }


def correctness(runner, snapshot, backends, variants, *, selector=select):
    parameter_ids = [id(p) for p in runner.model.parameters()]
    keys = tuple(runner.model.state_dict())
    selector(backends, 'native')
    runner.restore(snapshot)
    expected = []
    for _ in range(3):
        runner.step()
        runner.check_finite()
        expected.append(collect(runner))
    report = {}
    for variant in variants:
        if variant == 'native':
            continue
        selector(backends, variant)
        runner.restore(snapshot)
        records = []
        for wanted in expected:
            runner.step()
            runner.check_finite()
            actual = collect(runner)
            torch.testing.assert_close(
                actual['loss'], wanted['loss'], rtol=0.003, atol=0.003
            )
            torch.testing.assert_close(actual['rng'], wanted['rng'], rtol=0, atol=0)
            gradients = group_error(actual['gradients'], wanted['gradients'])
            parameters = group_error(actual['parameters'], wanted['parameters'])
            if gradients['relative_l2'] > 0.05 or parameters['relative_l2'] > 0.01:
                raise AssertionError((variant, gradients, parameters))
            records.append(
                {
                    'loss': error(actual['loss'], wanted['loss']),
                    'gradient': gradients,
                    'parameter': parameters,
                    'identical_dropout_rng': True,
                }
            )
        report[variant] = records
    assert [id(p) for p in runner.model.parameters()] == parameter_ids
    assert tuple(runner.model.state_dict()) == keys
    assert {
        id(p) for group in runner.optimizer.param_groups for p in group['params']
    } == set(parameter_ids)
    return report


def measure_paired(
    runner, snapshot, backends, variants, *, repeats, steps, selector=select
):
    wall = {v: [] for v in variants}
    phases = {v: {p: [] for p in runner.phases} for v in variants}
    memory = {v: [] for v in variants}
    rng = random.Random(641)
    orders = []
    for repetition in range(repeats):
        order = list(variants)
        rng.shuffle(order)
        orders.append(order)
        for variant in order:
            selector(backends, variant)
            runner.restore(snapshot)
            torch.cuda.reset_peak_memory_stats(runner.device)
            baseline = torch.cuda.memory_allocated(runner.device)
            begin = time.perf_counter()
            for _ in range(steps):
                runner.step()
            runner.sync()
            wall[variant].append((time.perf_counter() - begin) * 1000 / steps)
            memory[variant].append(
                {
                    'baseline_allocated_bytes': baseline,
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated(
                        runner.device
                    ),
                }
            )
            runner.check_finite()
        print(
            f'wall repetition {repetition + 1}/{repeats}: '
            + ', '.join(f'{v}={wall[v][-1]:.3f}ms' for v in order),
            flush=True,
        )
    events = [
        {
            p: (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for p in runner.phases
        }
        for _ in range(steps)
    ]
    for event_set in events:
        for begin, end in event_set.values():
            begin.record()
            end.record()
    runner.sync()
    for _ in range(repeats):
        order = list(variants)
        rng.shuffle(order)
        for variant in order:
            selector(backends, variant)
            runner.restore(snapshot)
            for event_set in events:

                @contextlib.contextmanager
                def scope(phase, event_set=event_set):
                    begin, end = event_set[phase]
                    begin.record()
                    yield
                    end.record()

                runner.step(scope)
            runner.sync()
            for event_set in events:
                for phase, (begin, end) in event_set.items():
                    phases[variant][phase].append(begin.elapsed_time(end))
            runner.check_finite()
    native = wall['native']
    return {
        'orders': orders,
        'paired_blocks': [
            {
                'order': order,
                'wall_ms_per_step': {v: wall[v][i] for v in variants},
            }
            for i, order in enumerate(orders)
        ],
        'steps_per_sample': steps,
        'repeats': repeats,
        'variants': {
            v: {
                'wall_ms_per_step': describe(wall[v]),
                'paired_speedup': describe([a / b for a, b in zip(native, wall[v])]),
                'phases_ms': {p: describe(values) for p, values in phases[v].items()},
                'memory': memory[v],
            }
            for v in variants
        },
    }


def run_case(config, checkpoint, output, args):
    output.mkdir(parents=True, exist_ok=False)
    print(f'case: {output.name}', flush=True)
    runner = make_runner(config)
    if checkpoint:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        runner.model.load_state_dict(saved['model'])
        checkpoint_metadata = {
            'path': str(checkpoint.resolve()),
            'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'epoch': saved['epoch'],
            'optimizer': 'Fresh state, initialized by native warmup',
        }
        del saved
    else:
        checkpoint_metadata = None
    for _ in range(config.warmup):
        runner.step()
    runner.check_finite()
    snapshot = runner.snapshot()
    backends = install_ragged_linears(
        runner.model, scope='all', schedule=args.schedule, tiled=args.tiled
    )
    variants = (
        ['native', 'ragged_hidden', 'ragged_all']
        if args.all_linears
        else ['native', 'ragged_hidden']
    )
    for variant in variants:
        select(backends, variant)
        runner.restore(snapshot)
        for _ in range(3):
            runner.step()
        runner.sync()
    checks = correctness(runner, snapshot, backends, variants)
    print('three-step correctness checks passed', flush=True)
    timings = measure_paired(
        runner, snapshot, backends, variants, repeats=args.repeats, steps=args.steps
    )
    traces = {}
    if not args.no_profile:
        for variant in variants:
            select(backends, variant)
            target = output / variant
            target.mkdir()
            traces[variant] = profile(runner, snapshot, target)
            (target / 'trace-summary.json').write_text(
                json.dumps(traces[variant], indent=2) + '\n'
            )
    layers = {
        name: {
            'hidden_scope': is_hidden(name),
            'calls': backend.calls,
            'fallback_calls': backend.fallback_calls,
            'prepared_selections': len(backend.cache),
            'shapes': [
                state.plans['forward'].shapes for state in backend.cache.values()
            ],
        }
        for name, backend in backends.items()
    }
    if any(value['fallback_calls'] for value in layers.values()):
        raise RuntimeError('A purported ragged operation used the native fallback')
    report = {
        'config': dataclasses.asdict(runner.config),
        'resolved_model': runner.model_config,
        'resolved_optimizer': runner.optimizer_config,
        'checkpoint': checkpoint_metadata,
        'dataset': runner.dataset_metadata,
        'original_padding': padding_report(runner.config),
        'correctness': checks,
        'timings': timings,
        'traces': traces,
        'layers': layers,
        'gpu_after': gpu_status(),
    }
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    for variant, timing in timings['variants'].items():
        print(
            f'{variant}: {timing["wall_ms_per_step"]["median"]:.3f} ms, '
            f'paired speedup {timing["paired_speedup"]["median"]:.3f}x',
            flush=True,
        )
    remove_ragged_linears(backends)
    del runner, backends, snapshot
    gc.collect()
    torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('dev/pepilica/configs/churn.json'),
    )
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=None,
    )
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument(
        '--cases',
        nargs='+',
        choices=['churn_muon', 'churn_adamw', 'churn_pow2_muon'],
        default=['churn_muon', 'churn_adamw', 'churn_pow2_muon'],
    )
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument(
        '--schedule', choices=['member', 'k_desc', 'round_robin'], default='member'
    )
    parser.add_argument('--tiled', action='store_true')
    parser.add_argument('--all-linears', action='store_true')
    parser.add_argument('--no-profile', action='store_true')
    args = parser.parse_args()
    if args.steps < 1 or args.repeats < 1:
        parser.error('steps and repeats must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_float32_matmul_precision('highest')
    metadata = environment(args.output)
    metadata['cpu_affinity'] = sorted(os.sched_getaffinity(0))
    props = torch.cuda.get_device_properties(0)
    metadata['gpu'] = {
        'name': props.name,
        'total_memory': props.total_memory,
        'compute_capability': [props.major, props.minor],
    }
    for name in (
        'dev/pepilica/scripts/benchmark_ragged_plugin.py',
        'dev/pepilica/src/tabpack_experimental/ragged_linear.py',
        'dev/pepilica/src/tabpack_experimental/ragged.py',
        'dev/pepilica/src/tabpack_experimental/compact_mlp.py',
        'dev/pepilica/tests/test_ragged_linear.py',
    ):
        source = Path(name)
        if 'tests' in source.parts and not source.is_file():
            continue
        target = args.output / 'sources' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        metadata['source_sha256'][name] = hashlib.sha256(
            source.read_bytes()
        ).hexdigest()
    (args.output / 'environment.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (args.output / 'protocol.json').write_text(
        json.dumps(
            {
                'arguments': {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                'scope': (
                    'Actual ModelPack and RealDataRunner, actual loss/backward/'
                    'optimizer and resident Churn train batches. Hidden scope '
                    'replaces intermediate linears only; all scope also replaces '
                    'input projection and output head. Original parameter '
                    'objects, activation/dropout/masks, optimizer and padded '
                    'storage are retained.'
                ),
                'timing': (
                    'Randomized interleaved blocks. Each block restores the same '
                    'post-warmup weights, optimizer, batch cursor and RNG outside '
                    'timing. Compilation and cached metadata preparation excluded; '
                    'every input/weight/bias/gradient pack and output/gradient '
                    'unpack included. Uninstrumented synchronized wall time is '
                    'primary. CUDA event phase samples are collected separately.'
                ),
                'correctness': (
                    'Three optimizer steps from an identical checkpoint/RNG/'
                    'batches; same dropout RNG; loss rtol/atol .003, global '
                    'gradient relative L2 <=.05 and parameter relative L2 <=.01. '
                    'BF16 arithmetic is not bitwise identical. Numerical and '
                    'performance validation, not a new accuracy experiment.'
                ),
                'limitations': (
                    'Shared GPU host; no clock locking. Original optimizer still '
                    'operates on padded parameters. Raw unused-input gradients '
                    'are zeroed by the adapter, equivalent after native upstream '
                    'masks. No timing claim for full data loading, preprocessing, '
                    'evaluation or epoch shuffling.'
                ),
            },
            indent=2,
        )
        + '\n'
    )
    config = Config(**json.loads(args.config.read_text()))
    for case in args.cases:
        actual = dataclasses.replace(config, steps=args.steps, repeats=args.repeats)
        checkpoint = args.checkpoint
        if case == 'churn_adamw':
            actual = dataclasses.replace(actual, optimizer='AdamWPack')
        elif case == 'churn_pow2_muon':
            actual = dataclasses.replace(actual, width=512, widths=(128, 256, 512))
            checkpoint = None
        run_case(actual, checkpoint, args.output / case, args)


if __name__ == '__main__':
    main()
