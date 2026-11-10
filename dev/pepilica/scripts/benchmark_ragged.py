import argparse
import dataclasses
import datetime
import hashlib
import json
import platform
import random
import shutil
import statistics
import subprocess
import time
from pathlib import Path

import torch
import triton

from tabpack_experimental.benchmark import Config, make_runner
from tabpack_experimental.ragged import LinearPlan, Tile
from tabpackage.nn import LinearPack


def capture(config, checkpoint):
    runner = make_runner(config)
    if checkpoint:
        saved = torch.load(checkpoint, map_location=runner.device, weights_only=True)
        runner.model.load_state_dict(saved['model'])
    records = []
    handles = []

    def hook(name):
        def record(module, args, output):
            idx = args[1] if len(args) > 1 else None
            weight = module.weight if idx is None else module.weight[idx]
            bias = module.bias if idx is None else module.bias[idx]
            inputs = module.in_features
            outputs = module.out_features
            if idx is not None:
                inputs = inputs[idx] if inputs is not None else None
                outputs = outputs[idx] if outputs is not None else None
            sizes_in = (
                [weight.shape[-1]] * len(weight) if inputs is None else inputs.tolist()
            )
            sizes_out = (
                [weight.shape[-2]] * len(weight)
                if outputs is None
                else outputs.tolist()
            )
            item = {
                'name': name,
                'x': args[0].detach().clone().to(torch.bfloat16),
                'w': weight.detach().clone().to(torch.bfloat16),
                'bias': bias.detach().clone().to(torch.bfloat16),
                'shapes': [
                    (args[0].shape[1], int(k), int(n))
                    for k, n in zip(sizes_in, sizes_out)
                ],
                'input_requires_grad': args[0].requires_grad,
            }
            records.append(item)
            output.register_hook(
                lambda grad: item.update(dy=grad.detach().clone().to(torch.bfloat16))
            )

        return record

    for name, module in runner.model.named_modules():
        if isinstance(module, LinearPack):
            handles.append(module.register_forward_hook(hook(name)))
    predictions = runner.forward()
    runner.loss_fn(predictions, runner.target).sum().backward()
    for handle in handles:
        handle.remove()
    for record in records:
        for i, (_, k, n) in enumerate(record['shapes']):
            record['x'][i, :, k:] = 0
            record['dy'][i, :, n:] = 0
        record['compact_x'] = torch.cat(
            [record['x'][i, :, :k] for i, (_, k, _) in enumerate(record['shapes'])],
            dim=1,
        )
        record['compact_w'] = torch.cat(
            [
                record['w'][i, :n, :k].flatten()
                for i, (_, k, n) in enumerate(record['shapes'])
            ]
        )
        record['compact_bias'] = torch.cat(
            [record['bias'][i, :n] for i, (_, _, n) in enumerate(record['shapes'])]
        )
        record['compact_dy'] = torch.cat(
            [record['dy'][i, :, :n] for i, (_, _, n) in enumerate(record['shapes'])],
            dim=1,
        )
    torch.cuda.synchronize()
    return records, runner.dataset_metadata if config.data_config else None


def graph_time(fn, *, inner, replays, repeats, warm_replays=0):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(inner):
            fn()
    for _ in range(warm_replays):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / (inner * replays))
    return {'median_us': statistics.median(samples), 'samples_us': samples}


def eager_time(fn, *, steps=50, repeats=5):
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(steps):
            fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e6 / steps)
    return {'median_us': statistics.median(samples), 'samples_us': samples}


def operation(record, mode, tile):
    kernel_mode = 'forward' if mode == 'forward_mm' else mode
    plan = LinearPlan(
        record['shapes'], kernel_mode, device=record['x'].device, tile=tile
    )
    a, b = {
        'forward': (record['compact_x'], record['compact_w']),
        'input_grad': (record['compact_dy'], record['compact_w']),
        'weight_grad': (record['compact_dy'], record['compact_x']),
    }[kernel_mode]
    bias = record['compact_bias'] if mode == 'forward' else None
    out = plan.empty_output(a.dtype)
    return plan, lambda: plan.run(a, b, bias, out=out), out


def baseline(record, mode):
    x, w, dy = record['x'], record['w'], record['dy']
    if mode == 'forward':
        out = torch.empty(dy.shape, device=dy.device, dtype=dy.dtype)
        return lambda: torch.baddbmm(record['bias'][:, None], x, w.mT, out=out), out
    if mode == 'forward_mm':
        out = torch.empty(dy.shape, device=dy.device, dtype=dy.dtype)
        return lambda: torch.bmm(x, w.mT, out=out), out
    if mode == 'input_grad':
        out = torch.empty_like(x)
        return lambda: torch.bmm(dy, w, out=out), out
    out = torch.empty_like(w)
    return lambda: torch.bmm(dy.mT, x, out=out), out


def compact_reference(record, mode, padded):
    if mode == 'weight_grad':
        return torch.cat(
            [
                padded[i, :n, :k].flatten()
                for i, (_, k, n) in enumerate(record['shapes'])
            ]
        )
    return torch.cat(
        [
            padded[i, :, : n if mode in ('forward', 'forward_mm') else k]
            for i, (_, k, n) in enumerate(record['shapes'])
        ],
        dim=1,
    )


def check(record, mode, actual, expected):
    diff = (actual.float() - expected.float()).abs()
    atol = max(1e-7, expected.float().abs().max().item() * 0.001)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0.03)
    relative_l2 = (diff.norm() / expected.float().norm().clamp_min(1e-30)).item()
    if relative_l2 > 0.015:
        raise AssertionError(f'{record["name"]}/{mode}: relative L2 {relative_l2}')
    if not torch.isfinite(actual).all():
        raise RuntimeError('Nonfinite kernel output')
    return {
        'max_abs_error': diff.max().item(),
        'relative_l2_error': relative_l2,
        'atol': atol,
        'rtol': 0.03,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='dev/pepilica/configs/churn.json',
    )
    parser.add_argument(
        '--checkpoint',
        default=None,
    )
    parser.add_argument('--output', required=True)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config_dict = json.loads(Path(args.config).read_text())
    config_dict.update(implementation='padded', device='cuda:0', dtype='bfloat16')
    config = Config(**config_dict)
    torch.cuda.set_device(config.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    capture_start = time.perf_counter()
    records, dataset = capture(config, args.checkpoint)
    print(
        f'Captured {len(records)} real linear operations '
        f'in {time.perf_counter() - capture_start:.2f}s',
        flush=True,
    )
    report = {
        'created_utc': datetime.datetime.now(datetime.UTC).isoformat(),
        'config': dataclasses.asdict(config),
        'dataset': dataset,
        'checkpoint': str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
        'checkpoint_sha256': hashlib.sha256(
            Path(args.checkpoint).read_bytes()
        ).hexdigest()
        if args.checkpoint
        else None,
        'environment': {
            'torch': torch.__version__,
            'triton': triton.__version__,
            'python': platform.python_version(),
            'gpu': torch.cuda.get_device_name(),
            'compute_capability': torch.cuda.get_device_capability(),
            'cuda_visible_devices': __import__('os').environ.get(
                'CUDA_VISIBLE_DEVICES'
            ),
            'nvidia_smi': subprocess.check_output(
                [
                    'nvidia-smi',
                    '--query-gpu=index,uuid,name,utilization.gpu,memory.used,clocks.sm',
                    '--format=csv,noheader',
                ]
            ).decode(),
        },
        'scope': [
            'BF16 operands captured from one actual train minibatch and a saved model.',
            'Forward includes bias; forward_mm measures plain GEMM without bias.',
            'No activation/dropout or output-width mask is included in either case.',
            'dX/dW are isolated GEMMs with captured gradients; no full autograd run.',
            'Compact and padded buffers represent identical useful values.',
            'Casting, packing, allocation, metadata, tuning are outside timings.',
            'CUDA Graph replay measures device execution; eager wall time is separate.',
            'Preallocated outputs; operator speedups do not imply training speedups.',
            'No optimizer changes or first-layer dX; raw inputs need no gradients.',
        ],
        'results': [],
    }
    source_dir = output / 'sources'
    source_dir.mkdir()
    for name in (
        'dev/pepilica/src/tabpack_experimental/ragged.py',
        'dev/pepilica/scripts/benchmark_ragged.py',
        'src/tabpackage/nn.py',
    ):
        shutil.copy2(name, source_dir / Path(name).name)
    executable_pairs = []
    for record in records:
        modes = ['forward', 'forward_mm', 'weight_grad']
        if record['input_requires_grad']:
            modes.append('input_grad')
        for mode in modes:
            ref_fn, ref_out = baseline(record, mode)
            ref_fn()
            expected = compact_reference(record, mode, ref_out)
            candidates = []
            for m, n, k, warps, stages in (
                (32, 64, 32, 4, 3),
                (32, 64, 32, 4, 2),
                (32, 32, 32, 4, 3),
                (64, 64, 32, 4, 3),
                (32, 64, 64, 4, 3),
                (16, 64, 32, 4, 3),
            ):
                if record['shapes'][0][1] == 16 and mode in ('forward', 'forward_mm'):
                    k = 16
                if record['shapes'][0][1] == 16 and mode == 'weight_grad':
                    n = 16
                tile = Tile(m, n, k, warps, stages)
                plan, fn, actual = operation(record, mode, tile)
                fn()
                check(record, mode, actual, expected)
                timing = graph_time(fn, inner=4, replays=3, repeats=3)
                candidates.append((timing['median_us'], tile, plan.scheduled_flops))
                if plan.vector or args.quick:
                    break
            _, tile, _ = min(candidates, key=lambda x: x[0])
            plan, fn, actual = operation(record, mode, tile)
            fn()
            correctness = check(record, mode, actual, expected)
            timings = {'padded': [], 'ragged': []}
            funcs = {'padded': ref_fn, 'ragged': fn}
            rng = random.Random(config.seed)
            for _ in range(3 if args.quick else 7):
                order = list(funcs)
                rng.shuffle(order)
                for key in order:
                    timing = graph_time(funcs[key], inner=16, replays=5, repeats=1)
                    timings[key].append(timing['median_us'])
            p_time = statistics.median(timings['padded'])
            r_time = statistics.median(timings['ragged'])
            correctness = check(record, mode, actual, expected)
            batch, max_k = record['x'].shape[1:]
            max_n = record['w'].shape[1]
            padded_flops = 2 * len(record['shapes']) * batch * max_k * max_n
            row = {
                'layer': record['name'],
                'mode': mode,
                'shapes_batch_input_output': record['shapes'],
                'padded_shape': list(record['w'].shape),
                'padded_gemm_flops': padded_flops,
                'useful_gemm_flops': plan.useful_flops,
                'ragged_scheduled_flops': plan.scheduled_flops,
                'tile_waste_fraction': 1 - plan.useful_flops / plan.scheduled_flops,
                'padded_waste_fraction': 1 - plan.useful_flops / padded_flops,
                'tile': dataclasses.asdict(tile),
                'vector_kernel': plan.vector,
                'shape_specialized': plan.specialized,
                'ragged_tasks': plan.n_tasks,
                'padded_operand_output_bytes': 2
                * (record['x'].numel() + record['w'].numel() + record['dy'].numel()),
                'compact_operand_output_bytes': 2
                * (plan.a_size + plan.b_size + plan.c_size),
                'metadata_bytes': (
                    plan.descriptors.numel() * plan.descriptors.element_size()
                    + plan.tasks.numel() * plan.tasks.element_size()
                ),
                'tuning': [
                    {
                        'median_us': t,
                        'tile': dataclasses.asdict(c),
                        'scheduled_flops': f,
                    }
                    for t, c, f in candidates
                ],
                'padded_graph_us': p_time,
                'ragged_graph_us': r_time,
                'speedup': p_time / r_time,
                'samples_us': timings,
                'correctness': correctness,
                'padded_eager': eager_time(ref_fn),
                'ragged_eager': eager_time(fn),
            }
            report['results'].append(row)
            executable_pairs.append((record['name'], mode, ref_fn, fn))
            (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
            print(
                f'{row["layer"]} {mode}: {p_time:.2f} -> {r_time:.2f} us '
                f'({p_time / r_time:.2f}x)',
                flush=True,
            )
    aggregate = {}
    for scope in (
        'forward',
        'forward_mm',
        'backward',
        'forward_backward',
        'matmuls_only',
    ):
        accepted = {
            'forward': {'forward'},
            'forward_mm': {'forward_mm'},
            'backward': {'weight_grad', 'input_grad'},
            'forward_backward': {'forward', 'weight_grad', 'input_grad'},
            'matmuls_only': {'forward_mm', 'weight_grad', 'input_grad'},
        }[scope]
        selected = [p for p in executable_pairs if p[1] in accepted]

        def padded_sequence(selected=selected):
            for _, _, fn, _ in selected:
                fn()

        def ragged_sequence(selected=selected):
            for _, _, _, fn in selected:
                fn()

        samples = {'padded': [], 'ragged': []}
        funcs = {'padded': padded_sequence, 'ragged': ragged_sequence}
        rng = random.Random(config.seed + 9)
        for _ in range(7):
            order = list(funcs)
            rng.shuffle(order)
            for key in order:
                samples[key].append(
                    graph_time(funcs[key], inner=8, replays=5, repeats=1)['median_us']
                )
        p, r = (
            {'median_us': statistics.median(samples[key]), 'samples_us': samples[key]}
            for key in ('padded', 'ragged')
        )
        p_time = statistics.median(samples['padded'])
        r_time = statistics.median(samples['ragged'])
        aggregate[scope] = {
            'padded': p,
            'ragged': r,
            'speedup': p_time / r_time,
            'padded_eager': eager_time(padded_sequence),
            'ragged_eager': eager_time(ragged_sequence),
        }
    report['aggregate'] = aggregate
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        for _ in range(3):
            for name, mode, p, r in executable_pairs:
                with torch.profiler.record_function(f'padded/{name}/{mode}'):
                    p()
                with torch.profiler.record_function(f'ragged/{name}/{mode}'):
                    r()
    prof.export_chrome_trace(str(output / 'trace.json'))
    (output / 'operators.txt').write_text(
        prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=50)
    )
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == '__main__':
    main()
