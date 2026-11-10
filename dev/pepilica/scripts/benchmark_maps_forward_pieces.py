"""Isolate Maps-routing forward components using matching synthetic tensors."""

import argparse
import functools
import json
import random
import statistics
from pathlib import Path

import torch

from dev.pepilica.scripts.prototype_fixed_slots import FixedSlotPlan


def measure(fn, repeats=7, iterations=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / iterations)
    return {'median_ms': statistics.median(times), 'samples_ms': times}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_per_process_memory_fraction(0.3)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.manual_seed(42)
    r = json.loads(args.report.read_text())
    widths = r['widths']
    depths = r['depths']
    batch = 1024
    n = len(widths)
    maxw = max(widths)
    stride = sum(widths)
    device = 'cuda'
    offsets = [sum(widths[:i]) for i in range(n)]
    woffset = [sum(v * v for v in widths[:i]) for i in range(n)]
    result = {
        'gpu': torch.cuda.get_device_name(),
        'scope': (
            'Synthetic tensors and real model geometry; excludes routing, '
            'native selection; GPU-event medians'
        ),
        'cases': [],
    }
    x = torch.randn(batch, stride, device=device, dtype=torch.bfloat16) * 0.1
    w = (
        torch.randn(sum(v * v for v in widths), device=device, dtype=torch.bfloat16)
        * 0.1
    )
    b = torch.randn(stride, device=device, dtype=torch.bfloat16) * 0.1
    for layer in range(1, max(depths)):
        active = tuple(i for i in range(n) if depths[i] > layer)
        plan = FixedSlotPlan(widths, batch, active, device=device)
        xpad = torch.zeros((len(active), batch, maxw), device=device, dtype=x.dtype)
        wpad = torch.zeros((len(active), maxw, maxw), device=device, dtype=x.dtype)
        bpad = torch.zeros((len(active), 1, maxw), device=device, dtype=x.dtype)
        for j, i in enumerate(active):
            z = widths[i]
            xpad[j, :, :z] = x[:, offsets[i] : offsets[i] + z]
            wpad[j, :z, :z] = w[woffset[i] : woffset[i] + z * z].view(z, z)
            bpad[j, 0, :z] = b[offsets[i] : offsets[i] + z]
        native = functools.partial(torch.baddbmm, bpad, xpad, wpad.transpose(1, 2))
        fixed = functools.partial(plan, x, w, b)
        a = native()
        z = fixed()
        torch.cuda.synchronize()
        errors = []
        for j, i in enumerate(active):
            size = widths[i]
            ref = a[j, :, :size].float()
            actual = z[:, offsets[i] : offsets[i] + size].float()
            errors.append(((actual - ref).norm() / ref.norm().clamp_min(1e-9)).item())
        assert max(errors) < 0.01, max(errors)
        order = ['native', 'fixed']
        random.Random(17 + layer).shuffle(order)
        funcs = {'native': native, 'fixed': fixed}
        times = {v: measure(funcs[v]) for v in order}
        entry = {
            'phase': 'hidden_gemm',
            'layer': layer,
            'active': len(active),
            'relative_l2_max': max(errors),
            'times': times,
        }
        result['cases'].append(entry)
        print(json.dumps(entry), flush=True)
        del plan, xpad, wpad, bpad, a, z
    xf = torch.randn(batch, stride, device=device, dtype=torch.float32)
    wf = torch.randn(w.numel(), device=device, dtype=torch.float32)
    for name, tensor in [('compact_activation', xf), ('compact_weight', wf)]:
        times = measure(functools.partial(tensor.to, torch.bfloat16))
        e = {'phase': 'cast', 'name': name, 'shape': list(tensor.shape), 'times': times}
        result['cases'].append(e)
        print(json.dumps(e), flush=True)
    del xf, wf, x, w, b
    xf = torch.randn((n, batch, 1000), device=device, dtype=torch.float32)
    wf = torch.randn((n, maxw, 1000), device=device, dtype=torch.float32)
    bf = torch.randn((n, 1, maxw), device=device, dtype=torch.float32)
    xb, wb, bb = xf.to(torch.bfloat16), wf.to(torch.bfloat16), bf.to(torch.bfloat16)
    funcs = {
        'input_cast': lambda: xf.to(torch.bfloat16),
        'weight_cast': lambda: wf.to(torch.bfloat16),
        'matmul': lambda: torch.baddbmm(bb, xb, wb.transpose(1, 2)),
        'casts_and_matmul': lambda: torch.baddbmm(
            bf.to(torch.bfloat16),
            xf.to(torch.bfloat16),
            wf.to(torch.bfloat16).transpose(1, 2),
        ),
    }
    for name, fn in funcs.items():
        e = {'phase': 'shared_first_projection', 'name': name, 'times': measure(fn)}
        result['cases'].append(e)
        print(json.dumps(e), flush=True)
    result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    (args.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
