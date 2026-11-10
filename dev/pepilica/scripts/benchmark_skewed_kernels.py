"""Isolated linear and Muon kernels for 99 narrow members and one wide member."""

import argparse
import dataclasses
import hashlib
import json
import random
import shutil
from pathlib import Path

import torch
from dev.pepilica.scripts.benchmark_ragged import baseline, check, compact_reference, graph_time
from dev.pepilica.scripts.benchmark_ragged_muon import ns_case
from dev.pepilica.scripts.benchmark_ragged_vocab import make_record

from tabpack_experimental.benchmark import describe, environment, gpu_status
from tabpack_experimental.compact_mlp import width_tiles
from tabpack_experimental.ragged import LinearPlan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--repeats', type=int, default=7)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.manual_seed(612)
    widths = (64,) * 99 + (1024,)
    report = {
        'environment': environment(args.output),
        'widths': widths,
        'batch': args.batch,
        'scope': (
            'Synthetic BF16 square hidden-layer operations, not dataset throughput. '
            'Linear timing excludes packing, casts, allocation and metadata. '
            'Complete five-step Muon backend timing includes normalization, '
            'pointwise work, casts, packing and output restoration.'
        ),
        'linear': {},
    }
    for name in (
        'dev/pepilica/scripts/benchmark_skewed_kernels.py',
        'dev/pepilica/scripts/benchmark_ragged.py',
        'dev/pepilica/scripts/benchmark_ragged_vocab.py',
        'dev/pepilica/scripts/benchmark_ragged_muon.py',
        'dev/pepilica/scripts/benchmark_ragged_plugin.py',
        'dev/pepilica/src/tabpack_experimental/ragged.py',
        'dev/pepilica/src/tabpack_experimental/compact_mlp.py',
        'dev/pepilica/src/tabpack_experimental/matrix_gemm.py',
        'dev/pepilica/src/tabpack_experimental/ragged_muon.py',
        'dev/pepilica/src/tabpack_experimental/ragged_linear.py',
    ):
        path = Path(name)
        target = args.output / 'sources' / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        report['environment']['source_sha256'][name] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()

    def save():
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')

    record = make_record(widths, args.batch, 1)
    rng = random.Random(907)
    for operation in ('forward_mm', 'forward', 'input_grad', 'weight_grad'):
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
        functions, plans, outputs = {'padded': padded}, {}, {}
        details = {}
        for name, tiles in (
            ('ragged_default', None),
            ('ragged_width_aware', width_tiles(widths, args.batch, mode)),
        ):
            plan = LinearPlan(
                record['shapes'], mode, schedule='k_desc', tiles_by_width=tiles
            )
            out = plan.empty_output(a.dtype)
            plans[name], outputs[name] = plan, out

            def run(plan=plan, out=out, a=a, b=b, bias=bias):
                return plan.run(a, b, bias, out=out)

            functions[name] = run
            run()
            details[name] = {
                'correctness': check(record, operation, out, expected),
                'useful_flops': plan.useful_flops,
                'scheduled_flops': plan.scheduled_flops,
                'tiles': [dataclasses.asdict(t) for t in plan.problem_tiles],
            }
        samples, orders = {name: [] for name in functions}, []
        for _ in range(args.repeats):
            order = list(functions)
            rng.shuffle(order)
            orders.append(order)
            for name in order:
                result = graph_time(functions[name], inner=4, replays=5, repeats=1)
                samples[name].append(result['median_us'])
        report['linear'][operation] = {
            'timing_us': {n: describe(s) for n, s in samples.items()},
            'orders': orders,
            'details': details,
            'padded_flops': 2 * len(widths) * args.batch * max(widths) ** 2,
        }
        print(
            operation,
            {n: describe(s)['median'] for n, s in samples.items()},
            flush=True,
        )
        save()
    del record, functions, outputs, plans, expected, a, b, bias, padded, padded_out
    torch.cuda.empty_cache()
    output = args.output / 'muon'
    output.mkdir()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        report['muon'] = ns_case([(w, w) for w in widths], output, args.repeats, True)
    torch.cuda.current_stream().wait_stream(stream)
    report['gpu_after'] = gpu_status()
    save()


if __name__ == '__main__':
    main()
