"""Compare optimizer GEMM tile sizes on the California model widths."""

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch

from tabpack_experimental.grouped_fixed_optimizer import GroupedFixedMuonAdamW, _mm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyperparameters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_per_process_memory_fraction(0.1)
    torch.set_num_threads(2)
    torch.manual_seed(7)
    hp = json.loads(args.hyperparameters.read_text())
    widths = tuple(v['model']['d_block'] for v in hp)
    depths = tuple(v['model']['n_blocks'] - 1 for v in hp)
    n = sum(w * w for w in widths)
    w = torch.nn.Parameter(torch.zeros(n, device='cuda'))
    b = torch.nn.Parameter(torch.zeros(sum(widths), device='cuda'))
    opt = GroupedFixedMuonAdamW(
        w, b, widths, muon_lr=0.001, adamw_lr=0.001, weight_decay=0.01
    )
    a = torch.randn(n, device='cuda', dtype=torch.bfloat16)
    operand, add = torch.randn_like(a), torch.randn_like(a)
    outputs = {tile: torch.zeros_like(a) for tile in (32, 64)}
    result = {
        'scope': (
            'identical Triton optimizer GEMM with 32x32 vs 64x64 output tiles; '
            'BK=32, 4 warps'
        ),
        'gpu': torch.cuda.get_device_name(),
        'uuid': str(torch.cuda.get_device_properties(0).uuid),
        'torch': torch.__version__,
        'widths': widths,
        'depths': depths,
        'cases': [],
    }
    for layer in range(max(depths)):
        active = tuple(i for i, depth in enumerate(depths) if depth > layer)
        plans = {}
        for tile in (32, 64):
            tasks = [
                (i, m, j)
                for i in active
                for m in range((widths[i] + tile - 1) // tile)
                for j in range((widths[i] + tile - 1) // tile)
            ]
            plans[tile] = torch.tensor(tasks, device='cuda', dtype=torch.int32)
        for transpose, alpha, beta in (
            (True, 1.0, 0.0),
            (False, 2.0315, -4.775),
            (False, 1.0, 3.4445),
        ):

            def run(tile, transpose=transpose, alpha=alpha, beta=beta, plans=plans):
                tasks = plans[tile]
                _mm[(len(tasks),)](
                    a,
                    operand,
                    outputs[tile],
                    add,
                    opt.desc,
                    tasks,
                    transpose,
                    alpha,
                    beta,
                    tile,
                    tile,
                    32,
                    num_warps=4,
                )

            run(32)
            run(64)
            torch.cuda.synchronize()
            torch.testing.assert_close(
                outputs[32].float(), outputs[64].float(), rtol=0.02, atol=0.02
            )
            for _ in range(5):
                run(32)
                run(64)
            samples = {tile: {'cuda_ms': [], 'host_ms': []} for tile in (32, 64)}
            rng = random.Random(123)
            for _ in range(9):
                order = [32, 64]
                rng.shuffle(order)
                for tile in order:
                    torch.cuda.synchronize()
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    t0 = time.perf_counter()
                    start.record()
                    for _ in range(30):
                        run(tile)
                    end.record()
                    end.synchronize()
                    samples[tile]['cuda_ms'].append(start.elapsed_time(end) / 30)
                    samples[tile]['host_ms'].append(
                        (time.perf_counter() - t0) * 1000 / 30
                    )
            case = {
                'layer': layer,
                'active': len(active),
                'transpose': transpose,
                'alpha': alpha,
                'beta': beta,
                'tiles': {
                    str(tile): {
                        'tasks': len(plans[tile]),
                        **{k: statistics.median(v) for k, v in samples[tile].items()},
                        'cuda_samples_ms': samples[tile]['cuda_ms'],
                    }
                    for tile in (32, 64)
                },
            }
            result['cases'].append(case)
            print(json.dumps(case), flush=True)
    (args.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
