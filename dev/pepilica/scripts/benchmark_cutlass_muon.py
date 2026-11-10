"""Compare grouped optimizer GEMMs on the actual California width distribution."""

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch

from experiments.cutlass_grouped.backend import CutlassGroupedMuon
from tabpack_experimental.grouped_fixed_optimizer import GroupedFixedMuonAdamW, _mm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hyperparameters', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_per_process_memory_fraction(0.15)
    torch.set_num_threads(2)
    torch.manual_seed(7)
    h = json.loads(args.hyperparameters.read_text())
    widths = tuple(v['model']['d_block'] for v in h)
    depths = tuple(v['model']['n_blocks'] - 1 for v in h)
    n = sum(w * w for w in widths)
    w = torch.nn.Parameter(torch.zeros(n, device='cuda'))
    b = torch.nn.Parameter(torch.zeros(sum(widths), device='cuda'))
    opt = GroupedFixedMuonAdamW(
        w, b, widths, muon_lr=0.001, adamw_lr=0.001, weight_decay=0.01
    )
    backend = CutlassGroupedMuon()
    backend.register(opt)
    a = torch.randn(n, device='cuda', dtype=torch.bfloat16)
    operand = torch.randn_like(a)
    add = torch.randn_like(a)
    y1 = torch.zeros_like(a)
    y2 = torch.zeros_like(a)
    result = {
        'scope': (
            'one grouped Newton-Schulz matmul, CUDA events and synchronized host time'
        ),
        'gpu': torch.cuda.get_device_name(),
        'widths': widths,
        'depths': depths,
        'cases': [],
    }
    for layer in range(max(depths)):
        active = tuple(i for i, d in enumerate(depths) if d > layer)
        _, _, tasks = opt._tasks(active)
        for transpose, alpha, beta in [
            (True, 1.0, 0.0),
            (False, 2.0315, -4.775),
            (False, 1.0, 3.4445),
        ]:

            def triton(tasks=tasks, transpose=transpose, alpha=alpha, beta=beta):
                _mm[(len(tasks),)](
                    a,
                    operand,
                    y1,
                    add,
                    opt.desc,
                    tasks,
                    transpose,
                    alpha,
                    beta,
                    32,
                    32,
                    32,
                    num_warps=4,
                )

            def cutlass(tasks=tasks, transpose=transpose, alpha=alpha, beta=beta):
                backend[(len(tasks),)](
                    a,
                    operand,
                    y2,
                    add,
                    opt.desc,
                    tasks,
                    transpose,
                    alpha,
                    beta,
                    32,
                    32,
                    32,
                    num_warps=4,
                )

            triton()
            cutlass()
            torch.cuda.synchronize()
            torch.testing.assert_close(y1.float(), y2.float(), rtol=0.02, atol=0.02)
            for _ in range(5):
                triton()
                cutlass()
            vals = {k: {'cuda_ms': [], 'host_ms': []} for k in ('triton', 'cutlass')}
            funcs = {'triton': triton, 'cutlass': cutlass}
            rng = random.Random(123)
            for _ in range(7):
                order = list(funcs)
                rng.shuffle(order)
                for name in order:
                    torch.cuda.synchronize()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    t0 = time.perf_counter()
                    start.record()
                    for _ in range(20):
                        funcs[name]()
                    end.record()
                    end.synchronize()
                    vals[name]['host_ms'].append((time.perf_counter() - t0) * 1000 / 20)
                    vals[name]['cuda_ms'].append(start.elapsed_time(end) / 20)
            case = {
                'layer': layer,
                'active': len(active),
                'transpose': transpose,
                'alpha': alpha,
                'beta': beta,
                'times': {
                    k: {metric: statistics.median(v) for metric, v in sample.items()}
                    for k, sample in vals.items()
                },
            }
            result['cases'].append(case)
            print(json.dumps(case), flush=True)
    (args.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
