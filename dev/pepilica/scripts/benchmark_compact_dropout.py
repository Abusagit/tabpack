"""Packed dropout: connected forward/backward timings and actual CUDA traces."""

import argparse
import datetime
import gc
import json
import os
import platform
import random
import shutil
import statistics
import subprocess
from pathlib import Path

import torch
import triton
from dev.pepilica.scripts.benchmark_ragged import eager_time
from torch import nn

from tabpack_experimental.benchmark import summarize_trace
from tabpack_experimental.compact_mlp import CompactMLP
from tabpack_experimental.ragged import HIDDEN_SIZES


class SeparateDropoutMLP(nn.Module):
    """Same compact linear/ReLU stack, with the proposed mask in eager PyTorch."""

    def __init__(self, source):
        super().__init__()
        self.stack = CompactMLP(
            source.widths, source.batch, source.depths, tiled=True, schedule='k_desc'
        )
        self.stack.load_state_dict(source.state_dict())
        for layer, block in enumerate(source.blocks):
            q = block.keep_probabilities.clone()
            if block.has_inactive:
                q = torch.where(block.active_columns.bool(), q, 1)
            self.register_buffer(f'q_{layer}', q)
            self.register_buffer(f'scale_{layer}', torch.where(q > 0, 1 / q, 0))

    def forward(self, x):  # ty: ignore[invalid-mutable-override]
        for layer, block in enumerate(self.stack.blocks):
            x = block(x)
            if self.training:
                q = getattr(self, f'q_{layer}')
                scale = getattr(self, f'scale_{layer}')
                keep = torch.rand(x.shape, device=x.device, dtype=torch.float32) < q
                x = (x.float() * keep * scale).to(x.dtype)
        return x


def graph_time(function):
    for _ in range(3):
        function()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=torch.cuda.current_stream()):
        for _ in range(3):
            function()
    for _ in range(100):
        graph.replay()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(10):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / 30


def measure(functions, repeats, *, graph):
    samples = {name: [] for name in functions}
    rng = random.Random(94)
    for _ in range(repeats):
        names = list(functions)
        rng.shuffle(names)
        for name in names:
            result = (
                graph_time(functions[name])
                if graph
                else eager_time(functions[name], steps=10, repeats=1)['median_us']
            )
            samples[name].append(result)
    return {
        name: {'median_us': statistics.median(values), 'samples_us': values}
        for name, values in samples.items()
    }


def run_case(
    widths,
    depths,
    batch,
    output,
    repeats,
    profile,
    epilogue=False,
    profile_only=False,
    scalar_probability=False,
):
    torch.manual_seed(581)
    probabilities = tuple((0.0, 0.1, 0.2, 0.3, 0.5)[i % 5] for i in range(len(widths)))
    settings = {'tiled': True, 'schedule': 'k_desc'}
    fused = CompactMLP(widths, batch, depths, dropout=probabilities, **settings)
    zero = CompactMLP(widths, batch, depths, **settings)
    zero.load_state_dict(fused.state_dict())
    models = {
        'no_dropout': zero,
        'fused_dropout': fused,
        'separate_dropout': SeparateDropoutMLP(fused),
    }
    if epilogue:
        fused_gemm = CompactMLP(
            widths, batch, depths, dropout=probabilities, fused=True, **settings
        )
        fused_zero = CompactMLP(widths, batch, depths, fused=True, **settings)
        fused_gemm.load_state_dict(fused.state_dict())
        fused_zero.load_state_dict(zero.state_dict())
        models = {
            'current_dropout': fused,
            'epilogue_dropout': fused_gemm,
            'current_no_dropout': zero,
            'epilogue_no_dropout': fused_zero,
        }
        if scalar_probability:
            scalar = CompactMLP(
                widths,
                batch,
                depths,
                dropout=probabilities,
                fused=True,
                scalar_dropout=True,
                **settings,
            )
            scalar.load_state_dict(fused.state_dict())
            models['epilogue_scalar_dropout'] = scalar
    x0 = torch.randn(batch, sum(widths), device='cuda', dtype=torch.bfloat16)
    dy = torch.randn_like(x0)
    inputs, phases, checks = (
        {},
        {phase: {} for phase in ('forward', 'backward', 'forward_backward')},
        {},
    )
    with torch.no_grad():
        reference = zero(x0)
        for name, model in models.items():
            model.eval()
            torch.testing.assert_close(model(x0), reference, atol=0, rtol=0)
            model.train()
    del reference
    paired = {}
    paired_state = torch.cuda.get_rng_state()
    for name, model in models.items():
        x = x0.clone().requires_grad_()
        inputs[name] = x

        def clear(model=model, x=x):
            model.zero_grad(set_to_none=False)
            if x.grad is not None:
                x.grad.zero_()

        def forward(model=model, x=x):
            with torch.no_grad():
                return model(x)

        def step(model=model, x=x, clear=clear):
            clear()
            y = model(x)
            y.backward(dy)
            return y

        stochastic = isinstance(model, SeparateDropoutMLP) or any(
            block.stochastic for block in model.blocks
        )
        if epilogue:
            torch.cuda.set_rng_state(paired_state)
        state = torch.cuda.get_rng_state()
        first = step().detach().clone()
        grads = [p.grad.clone() for p in (x, *model.parameters()) if p.grad is not None]
        if epilogue:
            if stochastic in paired:
                ref, ref_grads = paired[stochastic]
                torch.testing.assert_close(first, ref, atol=0, rtol=0)
                for a, b in zip(grads, ref_grads):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
                del ref, ref_grads
            else:
                paired[stochastic] = (first.clone(), [g.clone() for g in grads])
        torch.cuda.set_rng_state(state)
        second = step()
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        for p, saved in zip((x, *model.parameters()), grads):
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()
            torch.testing.assert_close(p.grad, saved, atol=0, rtol=0)
        if stochastic:
            assert not torch.equal(first, step())
        assert torch.isfinite(first).all()
        checks[name] = {
            'eval_matches_no_dropout': True,
            'reproducible_output_and_gradients': True,
            'finite': True,
            'paired_epilogue_outputs_and_gradients_equal': epilogue,
        }
        del first, second, grads
        saved_output = model(x)

        def backward(saved_output=saved_output, clear=clear):
            clear()
            saved_output.backward(dy, retain_graph=True)

        phases['forward'][name] = forward
        phases['backward'][name] = backward
        phases['forward_backward'][name] = step
    paired.clear()
    print('Evaluation, RNG reproducibility and finite gradients validated', flush=True)
    timings, eager = {}, {}
    for phase, functions in () if profile_only else phases.items():
        timings[phase] = measure(functions, repeats, graph=True)
        eager[phase] = measure(functions, repeats, graph=False)
        print(f'Measured {phase}', flush=True)
    memory = {}
    for name, step in phases['forward_backward'].items():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        y = step()
        torch.cuda.synchronize()
        memory[name] = {
            'incremental_peak_bytes': torch.cuda.max_memory_allocated() - before,
            'parameter_bytes': sum(
                p.numel() * p.element_size() for p in models[name].parameters()
            ),
        }
        del y
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        y = phases['forward'][name]()
        torch.cuda.synchronize()
        memory[name]['forward_incremental_peak_bytes'] = (
            torch.cuda.max_memory_allocated() - before
        )
        del y
    if profile:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            with torch.profiler.record_function('profiler/warmup'):
                for step in phases['forward_backward'].values():
                    step()
            torch.cuda.synchronize()
            for _ in range(3):
                for name, step in phases['forward_backward'].items():
                    with torch.profiler.record_function(f'tabpack/{name}'):
                        step()
            torch.cuda.synchronize()
        trace = output / 'trace.json'
        prof.export_chrome_trace(str(trace))
        (output / 'operators.txt').write_text(
            prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=60)
        )
        summary = summarize_trace(json.loads(trace.read_text()), 3, True, tuple(models))
        summary['notes'].append(
            'Profiler warm-up is outside the labelled steps. Its events appear '
            'as unassigned and are excluded from per-variant phase counts.'
        )
        for name, model in models.items():
            stack = model.stack if isinstance(model, SeparateDropoutMLP) else model
            expected = sum(1 if block.fused else 2 for block in stack.blocks)
            observed = sum(
                op['kernel_launches_per_step']
                for op in summary['operators']
                if op['phase'] == name and op['name'] == '_CompactLayer'
            )
            if observed != expected:
                raise RuntimeError(
                    f'Incomplete forward trace for {name}: {observed} != {expected}'
                )
        (output / 'trace-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return {
        'batch': batch,
        'widths': widths,
        'depths': fused.depths,
        'dropout': probabilities,
        'correctness': checks,
        'cuda_graph': timings,
        'eager': eager,
        'memory': memory,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument(
        '--case',
        choices=('all', 'block', 'uniform', 'mixed_depth', 'vocabulary'),
        default='all',
    )
    parser.add_argument('--no-profile', action='store_true')
    parser.add_argument('--epilogue', action='store_true')
    parser.add_argument('--scalar-probability', action='store_true')
    parser.add_argument('--profile-only', action='store_true')
    args = parser.parse_args()
    if args.profile_only and args.no_profile:
        parser.error('--profile-only cannot be combined with --no-profile')
    if args.scalar_probability and not args.epilogue:
        parser.error('--scalar-probability requires --epilogue')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    report = {
        'created_utc': datetime.datetime.now(datetime.UTC).isoformat(),
        'epilogue_comparison': args.epilogue,
        'scalar_probability_comparison': args.scalar_probability,
        'profile_only': args.profile_only,
        'scope': (
            'Connected synthetic compact BF16 hidden stacks with input/weight/bias '
            'gradients. Same weights and GEMM tile policy in all variants; '
            'paired masks for epilogue comparison. Per-member probabilities broadcast '
            'across '
            'batch rows; finished members bypass dropout. Forward is training-mode '
            'no_grad. Backward-only reuses one forward graph/mask. Forward+backward '
            'generates fresh seeds/masks each step and includes gradient '
            'clearing/accumulation. Fused seed generation is included, one PyTorch '
            'RNG launch per forward; backward regenerates masks. No optimizer, '
            'loss, head, input projection, compilation or boundary conversions. '
            'CUDA Graph device timings and separate synchronized eager wall '
            'timings; randomized variant order.'
        ),
        'environment': {
            'python': platform.python_version(),
            'torch': torch.__version__,
            'triton': triton.__version__,
            'gpu': torch.cuda.get_device_name(),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'nvidia_smi': subprocess.check_output(
                [
                    'nvidia-smi',
                    '--query-gpu=index,uuid,name,utilization.gpu,memory.used,clocks.sm',
                    '--format=csv,noheader',
                ]
            ).decode(),
        },
        'cases': {},
    }
    sources = args.output / 'sources'
    sources.mkdir()
    for path in (
        Path(__file__),
        Path('dev/pepilica/src/tabpack_experimental/compact_mlp.py'),
        Path('dev/pepilica/src/tabpack_experimental/ragged.py'),
        Path('dev/pepilica/tests/test_compact_dropout.py'),
        Path('dev/pepilica/tests/test_fused_compact.py'),
        Path('dev/pepilica/tests/test_scalar_dropout.py'),
        Path('dev/pepilica/scripts/probe_compact_resources.py'),
    ):
        if 'tests' not in path.parts or path.is_file():
            shutil.copy2(path, sources / path.name)
    for name, widths, depths in (
        ('block', tuple((128, 256, 384)[i % 3] for i in range(64)), 1),
        ('uniform', (256,) * 16, 3),
        (
            'mixed_depth',
            tuple((128, 256, 384)[i % 3] for i in range(64)),
            tuple(i % 4 + 1 for i in range(64)),
        ),
        ('vocabulary', HIDDEN_SIZES, 2),
    ):
        if name == 'block' and args.case == 'all' and not args.epilogue:
            continue
        if args.case not in ('all', name):
            continue
        output = args.output / name
        output.mkdir()
        print(f'Starting {name}', flush=True)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            case = run_case(
                widths,
                depths,
                args.batch,
                output,
                args.repeats,
                not args.no_profile,
                args.epilogue,
                args.profile_only,
                args.scalar_probability,
            )
        torch.cuda.current_stream().wait_stream(stream)
        report['cases'][name] = case
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'case': name, 'cuda_graph': case['cuda_graph']}), flush=True)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
