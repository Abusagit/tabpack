"""Compare Maps-routing fixed-slot forward tile sizes with padded baddbmm."""

import argparse
import json
import random
import statistics
from pathlib import Path

import torch
from dev.pepilica.scripts.prototype_fixed_slots import FixedSlotPlan, fixed_slot_mm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_per_process_memory_fraction(0.3)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.manual_seed(42)
    config = json.loads(args.report.read_text())
    widths = config['widths']
    depths = config['depths']
    batch = 1024
    n = len(widths)
    stride = sum(widths)
    mx = max(widths)
    offsets = [sum(widths[:i]) for i in range(n)]
    woffset = [sum(v * v for v in widths[:i]) for i in range(n)]
    x = torch.randn(batch, stride, device='cuda', dtype=torch.bfloat16) * 0.1
    w = (
        torch.randn(sum(v * v for v in widths), device='cuda', dtype=torch.bfloat16)
        * 0.1
    )
    b = torch.randn(stride, device='cuda', dtype=torch.bfloat16) * 0.1
    result = {
        'scope': 'BF16 synthetic inputs with actual Maps widths and depths, prebuilt native inputs, fixed output allocated and zeroed; forward only',
        'gpu': torch.cuda.get_device_name(),
        'uuid': str(torch.cuda.get_device_properties(0).uuid),
        'cases': [],
    }
    for layer in range(1, max(depths)):
        active = tuple(i for i in range(n) if depths[i] > layer)
        plan = FixedSlotPlan(widths, batch, active, device='cuda')
        tasks = {32: plan.forward_tasks}
        tasklist = [
            (i, m, j)
            for i in active
            for m in range((batch + 63) // 64)
            for j in range((widths[i] + 63) // 64)
        ]
        tasklist.sort(key=lambda item: -widths[item[0]])
        tasks[64] = torch.tensor(tasklist, device='cuda', dtype=torch.int32)
        xp = torch.zeros(len(active), batch, mx, device='cuda', dtype=x.dtype)
        wp = torch.zeros(len(active), mx, mx, device='cuda', dtype=x.dtype)
        bp = torch.zeros(len(active), 1, mx, device='cuda', dtype=x.dtype)
        for j, i in enumerate(active):
            z = widths[i]
            xp[j, :, :z] = x[:, offsets[i] : offsets[i] + z]
            wp[j, :z, :z] = w[woffset[i] : woffset[i] + z * z].view(z, z)
            bp[j, 0, :z] = b[offsets[i] : offsets[i] + z]

        def fixed(tile):
            out = torch.zeros_like(x)
            fixed_slot_mm[(len(tasks[tile]),)](
                x,
                w,
                b,
                out,
                plan.descriptors,
                tasks[tile],
                stride,
                batch,
                tile,
                tile,
                32,
                num_warps=4,
            )
            return out

        def native():
            return torch.baddbmm(bp, xp, wp.transpose(1, 2))

        ref = native()
        outputs = {k: fixed(k) for k in (32, 64)}
        torch.cuda.synchronize()
        errors = {}
        for tile, out in outputs.items():
            e = []
            for j, i in enumerate(active):
                z = widths[i]
                expect = ref[j, :, :z].float()
                got = out[:, offsets[i] : offsets[i] + z].float()
                e.append(((got - expect).norm() / expect.norm().clamp_min(1e-9)).item())
            errors[tile] = max(e)
            assert max(e) < 0.01, max(e)
        funcs = {
            'native': native,
            'tile32': lambda: fixed(32),
            'tile64': lambda: fixed(64),
        }
        for fn in funcs.values():
            for _ in range(4):
                fn()
        torch.cuda.synchronize()
        samples = {name: [] for name in funcs}
        rng = random.Random(71 + layer)
        for _ in range(9):
            order = list(funcs)
            rng.shuffle(order)
            for name in order:
                torch.cuda.synchronize()
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                for _ in range(20):
                    funcs[name]()
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end) / 20)
        case = {
            'layer': layer,
            'active': len(active),
            'task_counts': {str(k): len(v) for k, v in tasks.items()},
            'relative_l2_max': errors,
            'times': {
                k: {'median_ms': statistics.median(v), 'samples_ms': v}
                for k, v in samples.items()
            },
        }
        result['cases'].append(case)
        print(json.dumps(case), flush=True)
    result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    (args.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
