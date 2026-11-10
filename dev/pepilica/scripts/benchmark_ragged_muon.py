"""Compare compact Newton-Schulz and its opt-in real TabPack optimizer adapter."""

import argparse
import dataclasses
import gc
import json
import os
import random
import shutil
from pathlib import Path

import torch
from dev.pepilica.scripts.benchmark_ragged_plugin import correctness, error, measure_paired, select

from tabpack_experimental.benchmark import (
    Config,
    describe,
    environment,
    gpu_status,
    make_runner,
    summarize_trace,
)
from tabpack_experimental.ragged import HIDDEN_SIZES
from tabpack_experimental.ragged_linear import (
    install_ragged_linears,
    remove_ragged_linears,
)
from tabpack_experimental.ragged_muon import MuonPlan, install_ragged_muon
from tabpackage.optim import _zeropower_via_newtonschulz5


def graph_sample(function, *, warm_replays, replays):
    for _ in range(3):
        function()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=torch.cuda.current_stream()):
        output = function()
    for _ in range(warm_replays):
        graph.replay()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    del output
    return start.elapsed_time(end) / replays


def save_profile(output, functions, phases, steps=3):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        with torch.profiler.record_function('profiler/warmup'):
            for function in functions.values():
                function()
        torch.cuda.synchronize()
        for _ in range(steps):
            for name, function in functions.items():
                with torch.profiler.record_function(f'tabpack/{name}'):
                    function()
        torch.cuda.synchronize()
    path = output / 'trace.json'
    prof.export_chrome_trace(str(path))
    (output / 'operators.txt').write_text(
        prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=80)
    )
    summary = summarize_trace(json.loads(path.read_text()), steps, True, phases)
    if any(summary['phases'][p]['kernel_launches_per_step'] == 0 for p in phases):
        raise RuntimeError('Profiler missed a labelled GPU phase')
    summary['notes'].append(
        'Unlabelled profiler warm-up appears as unassigned and is excluded '
        'from labelled counts.'
    )
    (output / 'trace-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def ns_case(shapes, output, repeats, profile):
    torch.manual_seed(612)
    max_rows, max_cols = max(r for r, _ in shapes), max(c for _, c in shapes)
    source = torch.zeros(
        len(shapes), max_rows, max_cols, device='cuda', dtype=torch.float32
    )
    for member, (rows, cols) in enumerate(shapes):
        source[member, :rows, :cols].normal_()
    plan = MuonPlan(shapes)
    functions = {
        'padded': lambda: _zeropower_via_newtonschulz5(source, 5),
        'ragged': lambda: plan.run_padded(source),
    }
    state = torch.cuda.get_rng_state()
    expected, actual = functions['padded'](), functions['ragged']()
    errors = []
    for member, (rows, cols) in enumerate(shapes):
        metrics = error(actual[member, :rows, :cols], expected[member, :rows, :cols])
        if metrics['relative_l2'] > 0.04:
            raise AssertionError(metrics)
        assert actual[member, rows:, :].count_nonzero() == 0
        assert actual[member, :rows, cols:].count_nonzero() == 0
        errors.append(metrics)
    assert torch.equal(state, torch.cuda.get_rng_state())
    del expected, actual
    large = max(max_rows, max_cols) >= 2048
    warm, replays = (3, 3) if large else (10, 20)
    timings = {name: [] for name in functions}
    orders = []
    rng = random.Random(592)
    for _ in range(repeats):
        order = list(functions)
        rng.shuffle(order)
        orders.append(order)
        for name in order:
            timings[name].append(
                graph_sample(functions[name], warm_replays=warm, replays=replays)
            )
            if large:
                gc.collect()
                torch.cuda.empty_cache()
    memory = {}
    for name, function in functions.items():
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        result = function()
        torch.cuda.synchronize()
        memory[name] = torch.cuda.max_memory_allocated() - before
        del result
    workspace = sum(
        t.numel() * t.element_size()
        for t in (plan.x, plan.y, plan.a, plan.b, plan.partials, plan.norms)
    )
    summary = save_profile(output, functions, tuple(functions)) if profile else None
    r, s = min(max_rows, max_cols), max(max_rows, max_cols)
    padded_flops = len(shapes) * 5 * (4 * r * r * s + 2 * r**3)
    useful = sum(
        5 * (4 * min(r, c) ** 2 * max(r, c) + 2 * min(r, c) ** 3) for r, c in shapes
    )
    scheduled = 5 * sum(
        p.scheduled_flops for p in (plan.gram, plan.square, plan.product)
    )
    report = {
        'shapes': shapes,
        'padded_shape': list(source.shape),
        'steps': 5,
        'correctness': {
            'member_errors': errors,
            'padding_zero': True,
            'rng_unchanged': True,
        },
        'timing_ms': {name: describe(values) for name, values in timings.items()},
        'paired_orders': orders,
        'graph_inner': 1,
        'warm_replays': warm,
        'timed_replays': replays,
        'flops': {
            'padded': padded_flops,
            'useful': useful,
            'ragged_scheduled': scheduled,
        },
        'incremental_peak_bytes': memory,
        'compact_persistent_workspace_bytes': workspace,
        'trace': summary,
    }
    print(
        json.dumps(
            {
                'ns_shapes': shapes[:4],
                'timing_ms': {n: d['median'] for n, d in report['timing_ms'].items()},
            }
        ),
        flush=True,
    )
    return report


def model_case(args, output):
    settings = json.loads(args.config.read_text())
    settings['widths'] = tuple(settings['widths']) if settings.get('widths') else None
    settings['cat_cardinalities'] = tuple(settings['cat_cardinalities'])
    config = dataclasses.replace(
        Config(**settings), repeats=args.repeats, steps=args.steps
    )
    runner = make_runner(config)
    for _ in range(config.warmup):
        runner.step()
    runner.check_finite()
    snapshot = runner.snapshot()
    torch.cuda.reset_peak_memory_stats()
    runner.step()
    runner.sync()
    clean_native_peak = torch.cuda.max_memory_allocated()
    runner.restore(snapshot)
    linears = install_ragged_linears(
        runner.model, scope='all', schedule='k_desc', tiled=True
    )
    muon = install_ragged_muon(runner.optimizer, runner.model)
    variants = ['native', 'ragged_muon', 'ragged_hidden', 'ragged_both']

    def selector(backends, variant):
        select(
            backends,
            'ragged_hidden'
            if variant in ('ragged_hidden', 'ragged_both')
            else 'native',
        )
        if variant in ('ragged_muon', 'ragged_both'):
            muon.enable()
        else:
            muon.disable()

    for variant in variants:
        selector(linears, variant)
        runner.restore(snapshot)
        for _ in range(3):
            runner.step()
        runner.sync()
    checks = correctness(runner, snapshot, linears, variants, selector=selector)
    print('Real TabPack three-step checks passed', flush=True)
    timings = measure_paired(
        runner,
        snapshot,
        linears,
        variants,
        repeats=args.repeats,
        steps=args.steps,
        selector=selector,
    )
    traces = {}
    if not args.no_profile:
        for variant in variants:
            selector(linears, variant)
            runner.restore(snapshot)
            target = output / variant
            target.mkdir()
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                with torch.profiler.record_function('profiler/warmup'):
                    runner.step()
                runner.sync()
                for _ in range(3):
                    runner.step(
                        lambda phase: torch.profiler.record_function(f'tabpack/{phase}')
                    )
                runner.sync()
            path = target / 'trace.json'
            prof.export_chrome_trace(str(path))
            (target / 'operators.txt').write_text(
                prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=80)
            )
            summary = summarize_trace(
                json.loads(path.read_text()), 3, True, runner.phases
            )
            if summary['phases']['optimizer']['kernel_launches_per_step'] == 0:
                raise RuntimeError('Profiler missed optimizer kernels')
            summary['notes'].append(
                'Unlabelled profiler warm-up is excluded from labelled counts.'
            )
            (target / 'trace-summary.json').write_text(
                json.dumps(summary, indent=2) + '\n'
            )
            traces[variant] = summary
    if any(b.fallback_calls for b in linears.values()):
        raise RuntimeError('Linear adapter used a native fallback')
    report = {
        'config': dataclasses.asdict(config),
        'dataset': runner.dataset_metadata,
        'resolved_model': runner.model_config,
        'resolved_optimizer': runner.optimizer_config,
        'correctness': checks,
        'timings': timings,
        'traces': traces,
        'muon_matrix_shapes': [p.shapes for p in muon.plans.values()],
        'native_without_adapters_peak_bytes': clean_native_peak,
        'compact_muon_scratch_bytes': sum(
            t.numel() * t.element_size()
            for p in muon.plans.values()
            for t in (p.x, p.y, p.a, p.ca, p.b, p.partials, p.norms)
        ),
        'scope': (
            'Real resident Churn train batches, full original TabPack model and '
            'optimizer. Padded parameters/momentum remain. Muon-only leaves '
            'model linears native; both uses the earlier hidden-linear adapter. '
            'This is not the fully compact MLP or a convergence experiment.'
        ),
    }
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    muon.disable()
    remove_ragged_linears(linears)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('all', 'ns', 'model'), default='all')
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('dev/pepilica/configs/churn.json'),
    )
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--no-profile', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    report = {
        'environment': environment(args.output),
        'affinity': sorted(os.sched_getaffinity(0)),
        'scope': (
            'Isolated five-step Newton-Schulz, then full TabPack integration. '
            'Same optimizer algorithm, BF16 intermediate rounding and '
            'coefficients; no optimizer replacement or iteration reduction. '
            'GPU is shared; clocks are not locked.'
        ),
        'ns': {},
    }
    for path in (
        'dev/pepilica/src/tabpack_experimental/ragged.py',
        'dev/pepilica/src/tabpack_experimental/matrix_gemm.py',
        'dev/pepilica/src/tabpack_experimental/ragged_muon.py',
        'dev/pepilica/src/tabpack_experimental/ragged_linear.py',
        'dev/pepilica/src/tabpack_experimental/compact_mlp.py',
        'dev/pepilica/scripts/benchmark_ragged_muon.py',
        'dev/pepilica/scripts/benchmark_ragged_plugin.py',
        'dev/pepilica/tests/test_ragged_muon.py',
    ):
        if 'tests' in Path(path).parts and not Path(path).is_file():
            continue
        target = args.output / 'sources' / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    if args.mode in ('all', 'ns'):
        for name, shapes in (
            ('uniform', [(256, 256)] * 16),
            ('mixed_widths', [((128, 256, 384)[i % 3],) * 2 for i in range(64)]),
            ('input_projection', [((128, 256, 384)[i % 3], 16) for i in range(64)]),
            ('vocabulary', [(h, h) for h in HIDDEN_SIZES]),
        ):
            print(f'NS case: {name}', flush=True)
            output = args.output / name
            output.mkdir()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                report['ns'][name] = ns_case(
                    shapes, output, args.repeats, not args.no_profile
                )
            torch.cuda.current_stream().wait_stream(stream)
            (args.output / 'report.json').write_text(
                json.dumps(report, indent=2) + '\n'
            )
            gc.collect()
            torch.cuda.empty_cache()
    if args.mode in ('all', 'model'):
        output = args.output / 'model'
        output.mkdir()
        report['model'] = model_case(args, output)
    report['gpu_after'] = gpu_status()
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
