"""Compare complete synthetic hidden-stack training steps across backends."""

import argparse
import gc
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from dev.pepilica.scripts.fixed_slot_mlp import FixedSlotMLP, FixedSlotMLPOptimizer
from tabpackage._training import make_optimizer_pack
from tabpack_experimental.compact_backbone import CompactBackboneBackend
from tabpackage.nn import MLPBackbonePack
from tabpackage.optim import MuonAdamWPack


class Padded(nn.Module):
    def __init__(self, widths, depths, dropout, state):
        super().__init__()
        self.backbone: Any = MLPBackbonePack(
            d_in=list(widths),
            d_block=list(widths),
            n_blocks=list(depths),
            dropout=list(dropout),
            forward_impl='indexed',
            pack_size=len(widths),
        ).cuda()
        col = weight_offset = 0
        for i, width in enumerate(widths):
            for layer, block in enumerate(self.backbone.blocks):
                with torch.no_grad():
                    block.linear.weight[i, :width, :width].copy_(
                        state['weights'][layer][
                            weight_offset : weight_offset + width * width
                        ].view(width, width)
                    )
                    block.linear.bias[i, :width].copy_(
                        state['biases'][layer][col : col + width]
                    )
            col += width
            weight_offset += width * width

    @property
    def pack_size(self):
        return self.backbone.pack_size

    def forward(self, x):  # ty: ignore[invalid-mutable-override]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return self.backbone(x)


def measure(fn, *, repeats=5, steps=3):
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(steps):
            fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000 / steps)
    return {'median_ms': statistics.median(samples), 'samples_ms': samples}


def phases(forward, backward, optimizer, *, repeats=5):
    result = {'forward': [], 'backward': [], 'optimizer': []}
    for _ in range(repeats):
        for name, fn in [
            ('forward', forward),
            ('backward', backward),
            ('optimizer', optimizer),
        ]:
            torch.cuda.synchronize()
            start = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            result[name].append((time.perf_counter() - start) * 1000)
    return {
        k: {'median_ms': statistics.median(v), 'samples_ms': v}
        for k, v in result.items()
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_per_process_memory_fraction(0.15)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    widths = (32, 64, 128, 256, 384, 128, 64, 256, 32, 384, 128, 64)
    depths = (1, 2, 3, 2, 3, 1, 2, 3, 2, 1, 3, 2)
    dropout = tuple(0.0 if i % 2 == 0 else 0.3 for i in range(len(widths)))
    batch = 128
    source = FixedSlotMLP(widths, depths, dropout)
    state = {
        'weights': [p.detach().cpu().clone() for p in source.weights],
        'biases': [p.detach().cpu().clone() for p in source.biases],
    }
    del source
    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(101)
    full_x = torch.randn(batch, sum(widths), device='cuda', dtype=torch.bfloat16)
    full_dy = torch.randn_like(full_x)
    report = {
        'scope': (
            'Synthetic hidden-stack steps; no embeddings, head, loss, evaluation '
            'or early stopping; '
            'dropout enabled; separate fresh state for each case.'
        ),
        'widths': widths,
        'depths': depths,
        'dropout': dropout,
        'batch': batch,
        'gpu': torch.cuda.get_device_name(),
        'torch': torch.__version__,
        'cases': [],
    }

    def run_case(count, variant):
        active = tuple(range(count))
        offset = sum(widths[:count])
        model: Any
        opt: Any
        torch.manual_seed(42)
        if variant == 'fixed_grouped':
            model = FixedSlotMLP(widths, depths, dropout)
            with torch.no_grad():
                for dst, src in zip(model.weights, state['weights']):
                    dst.copy_(src)
                for dst, src in zip(model.biases, state['biases']):
                    dst.copy_(src)
            fixed_opt = FixedSlotMLPOptimizer(
                model,
                muon_lr=0.001,
                adamw_lr=0.001,
                weight_decay=0.01,
            )
            x = full_x.detach().clone().requires_grad_()
            dy = torch.zeros_like(x)
            dy[:, :offset] = full_dy[:, :offset]

            def run():
                return model(x, active)

            clear = fixed_opt.zero_grad

            def step():
                return fixed_opt.step(active)
        else:
            model = Padded(widths[:count], depths[:count], dropout[:count], state)
            native_model: Any = model
            opt = make_optimizer_pack(
                native_model,
                type=MuonAdamWPack,
                muon_lr=0.001,
                adamw_lr=0.001,
                weight_decay=0.01,
            )
            if variant == 'changing':
                backend = CompactBackboneBackend(model.backbone)
            pieces = []
            upstream = []
            col = 0
            for width in widths[:count]:
                pieces.append(
                    torch.nn.functional.pad(
                        full_x[:, col : col + width],
                        (0, max(widths[:count]) - width),
                    )
                )
                upstream.append(
                    torch.nn.functional.pad(
                        full_dy[:, col : col + width],
                        (0, max(widths[:count]) - width),
                    )
                )
                col += width
            x = torch.stack(pieces).detach().requires_grad_()
            dy = torch.stack(upstream)

            def run():
                return model(x)

            def clear():
                return opt.zero_grad(set_to_none=True)

            step = opt.step

        def whole():
            clear()
            x.grad = None
            y = run()
            y.backward(dy)
            step()

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        whole()
        torch.cuda.synchronize()
        first = (time.perf_counter() - start) * 1000
        for _ in range(2):
            whole()
        timing = measure(whole)
        stash = {}

        def fwd():
            clear()
            x.grad = None
            stash['y'] = run()

        def bwd():
            stash.pop('y').backward(dy)

        phase = phases(fwd, bwd, step)
        peak = torch.cuda.max_memory_allocated()
        case = {
            'active': count,
            'variant': variant,
            'first_step_ms': first,
            'steady': timing,
            'synchronized_phase_times': phase,
            'peak_allocated_bytes': peak,
        }
        if variant == 'changing':
            backend.remove()
        return case

    for count in (12, 8, 4):
        for variant in ('native', 'changing', 'fixed_grouped'):
            case = run_case(count, variant)
            report['cases'].append(case)
            (args.output / 'report.json').write_text(
                json.dumps(report, indent=2) + '\n'
            )
            print(json.dumps(case), flush=True)
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
