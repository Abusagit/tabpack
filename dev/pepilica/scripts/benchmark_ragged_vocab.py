"""Compare vocabulary, generic, and padded hidden-layer GEMMs."""

import argparse
import dataclasses
import datetime
import json
import os
import platform
import random
import shutil
import statistics
import time
from pathlib import Path

import torch
import triton
from dev.pepilica.scripts.benchmark_ragged import baseline, check, compact_reference, graph_time

from tabpack_experimental.ragged import HIDDEN_SIZES, LinearPlan, Tile


def make_record(widths, batch, members):
    widths = tuple(widths) * members
    maximum = max(widths)
    count = len(widths)
    x = torch.randn(count, batch, maximum, device='cuda', dtype=torch.bfloat16)
    w = torch.randn(count, maximum, maximum, device='cuda', dtype=torch.bfloat16)
    dy = torch.randn_like(x)
    bias = torch.randn(count, maximum, device='cuda', dtype=torch.bfloat16)
    for i, width in enumerate(widths):
        x[i, :, width:] = 0
        dy[i, :, width:] = 0
    return {
        'name': 'hidden',
        'shapes': [(batch, width, width) for width in widths],
        'x': x,
        'w': w,
        'dy': dy,
        'bias': bias,
        'compact_x': torch.cat([x[i, :, :h] for i, h in enumerate(widths)], dim=1),
        'compact_w': torch.cat([w[i, :h, :h].flatten() for i, h in enumerate(widths)]),
        'compact_dy': torch.cat([dy[i, :, :h] for i, h in enumerate(widths)], dim=1),
        'compact_bias': torch.cat([bias[i, :h] for i, h in enumerate(widths)]),
    }


def measure(record, operation, repeats):
    mode = 'forward' if operation == 'forward_mm' else operation
    a, b = {
        'forward': (record['compact_x'], record['compact_w']),
        'input_grad': (record['compact_dy'], record['compact_w']),
        'weight_grad': (record['compact_dy'], record['compact_x']),
    }[mode]
    bias = record['compact_bias'] if operation == 'forward' else None
    padded, padded_out = baseline(record, operation)
    padded()
    expected = compact_reference(record, operation, padded_out)
    functions = {'padded': padded}
    outputs = {}
    details = {}
    constructors = [
        ('vocabulary', LinearPlan, {}),
        ('generic', LinearPlan, {'specialize': False}),
    ]
    for name, constructor, kwargs in constructors:
        start = time.perf_counter()
        plan = constructor(record['shapes'], mode, tile=Tile(), **kwargs)
        torch.cuda.synchronize()
        preparation_ms = (time.perf_counter() - start) * 1000
        out = plan.empty_output(a.dtype)

        def run(plan=plan, out=out):
            return plan.run(a, b, bias, out=out)

        start = time.perf_counter()
        run()
        torch.cuda.synchronize()
        first_launch_ms = (time.perf_counter() - start) * 1000
        outputs[name] = out
        functions[name] = run
        details[name] = {
            'kernel_kind': getattr(
                plan, 'kernel_kind', 'specialized' if plan.specialized else 'generic'
            ),
            'compiled_shape_count': len(
                getattr(plan, 'kernel_signatures', plan.signatures)
            ),
            'preparation_ms': preparation_ms,
            'first_launch_ms_including_possible_compilation': first_launch_ms,
            'tile': dataclasses.asdict(plan.tile),
            'useful_flops': plan.useful_flops,
            'scheduled_flops': plan.scheduled_flops,
            'correctness': check(record, operation, out, expected),
        }
    samples = {name: [] for name in functions}
    rng = random.Random(97)
    for _ in range(repeats):
        order = list(functions)
        rng.shuffle(order)
        for name in order:
            timing = graph_time(functions[name], inner=4, replays=3, repeats=1)
            samples[name].append(timing['median_us'])
    medians = {name: statistics.median(values) for name, values in samples.items()}
    for name, out in outputs.items():
        details[name]['correctness'] = check(record, operation, out, expected)
    return {
        'operation': operation,
        'median_us': medians,
        'samples_us': samples,
        'padded_flops': 2 * record['w'].numel() * record['x'].shape[1],
        'speedup_vs_padded': medians['padded'] / medians['vocabulary'],
        'speedup_vs_generic': medians['generic'] / medians['vocabulary'],
        'details': details,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--members-per-width', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=7)
    args = parser.parse_args()
    if min(args.batch_size, args.members_per_width, args.repeats) < 1:
        parser.error('Batch size, members, and repeats must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_device(0)
    torch.manual_seed(83)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = {
        'created_utc': datetime.datetime.now(datetime.UTC).isoformat(),
        'scope': (
            'Synthetic square hidden layers; isolated BF16 GEMMs, '
            'not end-to-end training. Fixed default tile; no autotuning. '
            'Compilation, packing and metadata preparation excluded from device timing.'
        ),
        'hidden_sizes': HIDDEN_SIZES,
        'batch_size': args.batch_size,
        'members_per_width': args.members_per_width,
        'environment': {
            'torch': torch.__version__,
            'triton': triton.__version__,
            'python': platform.python_version(),
            'gpu': torch.cuda.get_device_name(),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        },
        'results': [],
    }
    sources = args.output / 'sources'
    sources.mkdir()
    for source in (Path(__file__), Path('dev/pepilica/src/tabpack_experimental/ragged.py')):
        shutil.copy2(source, sources / source.name)
    for case, widths in (
        ('original_widths', (128, 256, 384)),
        ('vocabulary_three', (128, 256, 512)),
        ('vocabulary_four', (64, 128, 256, 512)),
        ('vocabulary_all', HIDDEN_SIZES),
    ):
        record = make_record(widths, args.batch_size, args.members_per_width)
        for operation in ('forward_mm', 'forward', 'input_grad', 'weight_grad'):
            row = measure(record, operation, args.repeats)
            row.update(case=case, widths=widths)
            report['results'].append(row)
            (args.output / 'report.json').write_text(
                json.dumps(report, indent=2) + '\n'
            )
            print(
                json.dumps({'case': case, 'operation': operation, **row['median_us']}),
                flush=True,
            )
        del record


if __name__ == '__main__':
    main()
