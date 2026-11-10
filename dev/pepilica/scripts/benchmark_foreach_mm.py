"""Pure GEMM comparison with torch.foreach.mm in an isolated newer environment."""

import argparse
import dataclasses
import hashlib
import importlib.metadata
import inspect
import json
import random
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import triton

from tabpack_experimental.compact_mlp import width_tiles
from tabpack_experimental.ragged import LinearPlan


def graph_time(function):
    for _ in range(3):
        function()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=torch.cuda.current_stream()):
        for _ in range(3):
            output = function()
    for _ in range(20):
        graph.replay()
    begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    begin.record()
    for _ in range(10):
        graph.replay()
    end.record()
    end.synchronize()
    del output
    return begin.elapsed_time(end) * 1000 / 30


def profile_function(function, output):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        for _ in range(3):
            result = function()
        torch.cuda.synchronize()
    del result
    profiler.export_chrome_trace(str(output))
    trace = json.loads(output.read_text())
    kernels = [e for e in trace['traceEvents'] if e.get('cat') == 'kernel']
    names = {}
    for event in kernels:
        names[event['name']] = names.get(event['name'], 0) + 1
    return {
        'total_kernel_events': len(kernels),
        'calls': 3,
        'kernels_per_call': len(kernels) / 3,
        'kernel_names': names,
    }


def case(widths, batch, output, repeats):
    output.mkdir()
    maximum, count = max(widths), len(widths)
    xs = [torch.randn(batch, h, device='cuda', dtype=torch.bfloat16) for h in widths]
    dys = [torch.randn_like(x) for x in xs]
    ws = [torch.randn(h, h, device='cuda', dtype=torch.bfloat16) for h in widths]
    x, dy = torch.cat(xs, dim=1), torch.cat(dys, dim=1)
    w = torch.cat([v.flatten() for v in ws])
    xv, dyv = x.split(widths, dim=1), dy.split(widths, dim=1)
    xp = torch.zeros(count, batch, maximum, device='cuda', dtype=torch.bfloat16)
    dyp = torch.zeros_like(xp)
    wp = torch.zeros(count, maximum, maximum, device='cuda', dtype=torch.bfloat16)
    for i, h in enumerate(widths):
        xp[i, :, :h], dyp[i, :, :h], wp[i, :h, :h] = xs[i], dys[i], ws[i]
    report, rng = {}, random.Random(809)
    for mode in ('forward', 'input_grad', 'weight_grad'):
        if mode == 'forward':
            a, b, av, bv = xs, [v.mT for v in ws], xv, [v.mT for v in ws]
            ca, cb = x, w

            def padded():
                return torch.bmm(xp, wp.mT)
        elif mode == 'input_grad':
            a, b, av, bv = dys, ws, dyv, ws
            ca, cb = dy, w

            def padded():
                return torch.bmm(dyp, wp)
        else:
            a, b, av, bv = [v.mT for v in dys], xs, [v.mT for v in dyv], xv
            ca, cb = dy, x

            def padded():
                return torch.bmm(dyp.mT, xp)

        def pack(values, mode=mode):
            return (
                torch.cat([v.flatten() for v in values])
                if mode == 'weight_grad'
                else torch.cat(tuple(values), dim=1)
            )

        reference = pack(
            [torch.mm(ai.float(), bi.float()).bfloat16() for ai, bi in zip(a, b)]
        )
        functions = {
            'padded': padded,
            'foreach_contiguous_lists': lambda a=a, b=b: torch.foreach.mm(a, b),
            'foreach_views_to_packed': lambda av=av, bv=bv, pack=pack: pack(
                torch.foreach.mm(av, bv)
            ),
            'loop_contiguous_lists': lambda a=a, b=b: tuple(
                torch.mm(ai, bi) for ai, bi in zip(a, b)
            ),
        }
        plans = {}
        for name, tiles in (
            ('ragged_default', None),
            ('ragged_width_aware', width_tiles(widths, batch, mode)),
        ):
            plan = LinearPlan(
                [(batch, h, h) for h in widths],
                mode,
                schedule='k_desc',
                tiles_by_width=tiles,
            )
            plans[name] = plan

            def run(plan=plan, ca=ca, cb=cb):
                return plan.run(ca, cb)

            functions[name] = run
        checks = {}
        for name, function in functions.items():
            actual = function()
            if isinstance(actual, tuple | list):
                actual = pack(actual)
            elif name == 'padded':
                actual = pack(
                    [
                        actual[i, :h, :h] if mode == 'weight_grad' else actual[i, :, :h]
                        for i, h in enumerate(widths)
                    ]
                )
            delta = actual.float() - reference.float()
            relative = (delta.norm() / reference.float().norm().clamp_min(1e-20)).item()
            if relative > 0.015 or not torch.isfinite(actual).all().item():
                raise AssertionError((mode, name, relative))
            checks[name] = {
                'relative_l2': relative,
                'max_abs': delta.abs().max().item(),
            }
        times = {name: [] for name in functions}
        orders = []
        for _ in range(repeats):
            order = list(functions)
            rng.shuffle(order)
            orders.append(order)
            for name in order:
                times[name].append(graph_time(functions[name]))
        wall = {name: [] for name in functions}
        for _ in range(3):
            order = list(functions)
            rng.shuffle(order)
            for name in order:
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(20):
                    result = functions[name]()
                torch.cuda.synchronize()
                del result
                wall[name].append((time.perf_counter() - start) * 1e6 / 20)
        profiles = {}
        for name, function in functions.items():
            profiles[name] = profile_function(
                function, output / f'{mode}-{name}-trace.json'
            )
        report[mode] = {
            'device_us': {
                name: {'median': statistics.median(v), 'samples': v}
                for name, v in times.items()
            },
            'eager_wall_us': {
                name: {'median': statistics.median(v), 'samples': v}
                for name, v in wall.items()
            },
            'correctness': checks,
            'orders': orders,
            'profiles': profiles,
            'plans': {
                name: {
                    'tiles': [dataclasses.asdict(t) for t in plan.problem_tiles],
                    'useful_flops': plan.useful_flops,
                    'scheduled_flops': plan.scheduled_flops,
                }
                for name, plan in plans.items()
            },
        }
        print(
            mode,
            {n: round(statistics.median(v), 3) for n, v in times.items()},
            flush=True,
        )
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=7)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.manual_seed(612)
    if not hasattr(torch, 'foreach') or not hasattr(torch.foreach, 'mm'):
        raise RuntimeError('This environment does not provide torch.foreach.mm')
    report: dict[str, Any] = {
        'torch': torch.__version__,
        'triton': triton.__version__,
        'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(),
        'capability': torch.cuda.get_device_capability(),
        'batch': 256,
        'packages': sorted(
            f'{d.metadata["Name"]}=={d.version}'
            for d in importlib.metadata.distributions()
        ),
        'gpu_status': subprocess.check_output(['nvidia-smi'], text=True),
        'scope': (
            'Pure BF16 GEMMs with FP32 accumulation; no bias/activation advantage. '
            'All calls allocate outputs. Contiguous-list foreach is an intrinsic '
            'API baseline; views-to-packed includes output concatenation and matches '
            'the ragged activation/gradient storage contract. Preparation, input '
            'conversion, compilation and graph capture excluded. Same environment '
            'and GPU for all methods; original 2.7 environment unchanged.'
        ),
        'cases': {},
    }
    for name in (
        'dev/pepilica/scripts/benchmark_foreach_mm.py',
        'dev/pepilica/src/tabpack_experimental/ragged.py',
        'dev/pepilica/src/tabpack_experimental/compact_mlp.py',
    ):
        source = Path(
            inspect.getfile(LinearPlan)
            if name.endswith('ragged.py')
            else inspect.getfile(width_tiles)
            if name.endswith('compact_mlp.py')
            else name
        )
        target = args.output / 'sources' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        report.setdefault('source_sha256', {})[name] = hashlib.sha256(
            source.read_bytes()
        ).hexdigest()
    compatibility = (
        Path(inspect.getfile(LinearPlan)).parent.parent / 'compatibility.json'
    )
    if compatibility.exists():
        report['kernel_compatibility'] = json.loads(compatibility.read_text())
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ):
        torch.ones(16, device='cuda').sin()
        torch.cuda.synchronize()
    for name, widths in (
        ('skew_99x64_1x1024', (64,) * 99 + (1024,)),
        ('uniform_16x256', (256,) * 16),
        ('mixed_64', tuple((128, 256, 384)[i % 3] for i in range(64))),
    ):
        print('Case', name, flush=True)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            report['cases'][name] = case(widths, 256, args.output / name, args.repeats)
        torch.cuda.current_stream().wait_stream(stream)
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
