"""Synchronized optimizer-only comparison for stable fixed slots."""

import argparse
import functools
import gc
import json
import random
import statistics
import time
from pathlib import Path

import torch
from torch import nn

from tabpack_experimental.grouped_fixed_optimizer import GroupedFixedMuonAdamW
from tabpackage.nn import ParameterPack
from tabpackage.optim import MuonAdamWPack


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_per_process_memory_fraction(0.15)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    widths = (32, 64, 128, 256, 384, 128, 64, 256, 32, 384, 128, 64)
    n = len(widths)
    size = sum(w * w for w in widths)
    bias_size = sum(widths)
    torch.manual_seed(42)
    base_w = torch.randn(size, device='cuda')
    base_b = torch.randn(bias_size, device='cuda')
    gw = torch.randn_like(base_w)
    gb = torch.randn_like(base_b)
    options = {
        'muon_lr': [0.002 + i * 0.0001 for i in range(n)],
        'adamw_lr': [0.001 + i * 0.00002 for i in range(n)],
        'weight_decay': [0.01 + i * 0.001 for i in range(n)],
    }
    result = {
        'scope': 'optimizer.step only; fixed gradients; synchronized eager blocks',
        'widths': widths,
        'gpu': torch.cuda.get_device_name(),
        'uuid': str(torch.cuda.get_device_properties(0).uuid),
        'torch': torch.__version__,
        'cases': [],
    }
    for count in (12, 8, 4):
        active = tuple(range(count))
        m = max(widths[:count])
        w2 = nn.Parameter(base_w.clone())
        b2 = nn.Parameter(base_b.clone())
        w2.grad = gw.clone()
        b2.grad = gb.clone()
        grouped = GroupedFixedMuonAdamW(w2, b2, widths, **options)
        wp = ParameterPack(torch.zeros(count, m, m, device='cuda'))
        bp = ParameterPack(torch.zeros(count, m, device='cuda'))
        wgp = torch.zeros_like(wp)
        bgp = torch.zeros_like(bp)
        wo = co = 0
        with torch.no_grad():
            for i, width in enumerate(widths[:count]):
                wp[i, :width, :width].copy_(
                    base_w[wo : wo + width * width].view(width, width)
                )
                bp[i, :width].copy_(base_b[co : co + width])
                wgp[i, :width, :width].copy_(
                    gw[wo : wo + width * width].view(width, width)
                )
                bgp[i, :width].copy_(gb[co : co + width])
                wo += width * width
                co += width
        wp.grad = wgp
        bp.grad = bgp
        native = MuonAdamWPack(
            [
                {'params': [wp], 'muon': True, 'muon_update_scale': 1.0},
                {'params': [bp], 'muon': False, 'weight_decay': 0.0},
            ],
            muon_lr=options['muon_lr'][:count],
            adamw_lr=options['adamw_lr'][:count],
            weight_decay=options['weight_decay'][:count],
            allow_inplace_grad_mutation=False,
            pack_size=count,
        )
        funcs = {
            'native': native.step,
            'grouped': functools.partial(grouped.step, active),
        }
        for fn in funcs.values():
            fn()
            fn()
        torch.cuda.synchronize()
        samples = {name: [] for name in funcs}
        rng = random.Random(91)
        for _ in range(7):
            order = list(funcs)
            rng.shuffle(order)
            for name in order:
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(5):
                    funcs[name]()
                torch.cuda.synchronize()
                samples[name].append((time.perf_counter() - start) * 1000 / 5)
        case = {
            'active': count,
            'timings': {
                k: {'median_ms': statistics.median(v), 'samples_ms': v}
                for k, v in samples.items()
            },
        }
        result['cases'].append(case)
        (a.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(case), flush=True)
        del funcs, grouped, native, wp, bp, wgp, bgp, w2, b2
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
