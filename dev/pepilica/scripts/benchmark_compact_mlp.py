"""Connected hidden-stack storage and scheduling experiment."""

import argparse
import dataclasses
import datetime
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import triton
from dev.pepilica.scripts.benchmark_ragged import eager_time, graph_time
from torch import nn

from tabpack_experimental.benchmark import Config, dimensions, summarize_trace
from tabpack_experimental.compact_mlp import CompactMLP
from tabpack_experimental.ragged import HIDDEN_SIZES
from tabpackage.nn import MLPBackbonePack


class PaddedReference(nn.Module):
    def __init__(self, source, *, forward_impl):
        super().__init__()
        self.backbone: Any = MLPBackbonePack(
            d_in=max(source.widths),
            n_blocks=source.depths[0]
            if len(set(source.depths)) == 1
            else list(source.depths),
            max_n_blocks=max(source.depths),
            d_block=source.widths[0]
            if len(set(source.widths)) == 1
            else list(source.widths),
            max_d_block=max(source.widths),
            dropout=0.0,
            pack_size=len(source.widths),
            forward_impl=forward_impl,
        ).to(device='cuda', dtype=torch.bfloat16)
        for layer, block in enumerate(source.blocks):
            linear = self.backbone.blocks[layer].linear
            with torch.no_grad():
                linear.weight.zero_()
                linear.bias.zero_()
                for position, member in enumerate(block.forward_plan.member_ids):
                    width = source.widths[member]
                    wo = block.forward_plan.w_offsets[position]
                    bo = block.forward_plan.bias_offsets[position]
                    linear.weight[member, :width, :width].copy_(
                        block.weight[wo : wo + width * width].view(width, width)
                    )
                    linear.bias[member, :width].copy_(block.bias[bo : bo + width])

    def forward(self, x):  # ty: ignore[invalid-mutable-override]
        return self.backbone(x)


def compare(actual, expected):
    actual, expected = actual.float(), expected.float()
    difference = actual - expected
    relative = (difference.norm() / expected.norm().clamp_min(1e-20)).item()
    if relative > 0.02 or not torch.isfinite(actual).all():
        raise AssertionError(f'Invalid output/gradient, relative L2 {relative}')
    return {'relative_l2': relative, 'max_abs': difference.abs().max().item()}


def compact_gradients(source, model, x):
    if isinstance(model, CompactMLP):
        return [x.grad, *(p.grad for p in model.parameters())]
    gradients = [source.pack(x.grad)]
    for layer, block in enumerate(source.blocks):
        linear = model.backbone.blocks[layer].linear
        gradients.extend(
            (
                torch.cat(
                    [
                        linear.weight.grad[
                            i, : source.widths[i], : source.widths[i]
                        ].flatten()
                        for i in block.forward_plan.member_ids
                    ]
                ),
                torch.cat(
                    [
                        linear.bias.grad[i, : source.widths[i]]
                        for i in block.forward_plan.member_ids
                    ]
                ),
            )
        )
    return gradients


def plans_report(model):
    if not isinstance(model, CompactMLP):
        return None
    return [
        {
            'members': block.forward_plan.member_ids,
            'modes': {
                name: {
                    'schedule': plan.schedule,
                    'tasks': plan.n_tasks,
                    'useful_flops': plan.useful_flops,
                    'scheduled_flops': plan.scheduled_flops,
                    'tiles': [dataclasses.asdict(tile) for tile in plan.problem_tiles],
                }
                for name, plan in (
                    ('forward', block.forward_plan),
                    ('input_grad', block.input_plan),
                    ('weight_grad', block.weight_plan),
                )
            },
        }
        for block in model.blocks
    ]


def run_case(
    widths,
    depths,
    batch,
    output,
    repeats,
    profile,
    replays,
    *,
    eager_steps=25,
    eager_only=False,
    seed=111,
):
    torch.manual_seed(seed)
    source = CompactMLP(widths, batch, depths)
    models = {
        'padded_dense': PaddedReference(source, forward_impl='dense'),
        'padded_indexed': PaddedReference(source, forward_impl='indexed'),
        'compact_default': source,
    }
    for name, settings in (
        ('compact_k_desc', {'schedule': 'k_desc'}),
        ('compact_fused', {'tiled': True, 'schedule': 'k_desc', 'fused': True}),
        ('compact_tiles_member', {'tiled': True}),
        ('compact_tiles_k_desc', {'tiled': True, 'schedule': 'k_desc'}),
        ('compact_tiles_round_robin', {'tiled': True, 'schedule': 'round_robin'}),
        (
            'compact_wide_k_desc',
            {'tiled': True, 'schedule': 'k_desc', 'large_tile': (64, 128, 32)},
        ),
    ):
        model = CompactMLP(widths, batch, depths, **settings)
        model.load_state_dict(source.state_dict())
        models[name] = model
    x0 = torch.randn(batch, sum(widths), device='cuda', dtype=torch.bfloat16)
    dy0 = torch.randn_like(x0)
    inputs, upstream, forward, training = {}, {}, {}, {}
    for name, model in models.items():
        compact = isinstance(model, CompactMLP)
        x = (x0.clone() if compact else source.unpack(x0)).requires_grad_()
        dy = dy0 if compact else source.unpack(dy0)
        inputs[name], upstream[name] = x, dy

        def inference(model=model, x=x):
            with torch.no_grad():
                return model(x)

        def train(model=model, x=x, dy=dy):
            model.zero_grad(set_to_none=False)
            if x.grad is not None:
                x.grad.zero_()
            y = model(x)
            y.backward(dy)
            return y

        forward[name], training[name] = inference, train
    correctness = {}
    gradient_layouts = {}
    reference = training['compact_default']().detach().clone()
    reference_grads = [
        g.clone() for g in compact_gradients(source, source, inputs['compact_default'])
    ]
    for name, model in models.items():
        layouts = {}
        handles = []
        for parameter_name, parameter in model.named_parameters():

            def record_layout(gradient, key=parameter_name, layouts=layouts):
                layouts[key] = {
                    'shape': list(gradient.shape),
                    'stride': list(gradient.stride()),
                    'contiguous': gradient.is_contiguous(),
                }

            handles.append(parameter.register_hook(record_layout))
        y = training[name]()
        for handle in handles:
            handle.remove()
        gradient_layouts[name] = layouts
        compact_y = y if isinstance(model, CompactMLP) else source.pack(y)
        correctness[name] = {
            'output': compare(compact_y, reference),
            'gradients': [
                compare(a, b)
                for a, b in zip(
                    compact_gradients(source, model, inputs[name]),
                    reference_grads,
                )
            ],
        }
    del y, compact_y, reference, reference_grads
    print('Connected outputs and gradients validated', flush=True)
    timings = {'forward': {}, 'forward_backward': {}}
    orders = {}
    device_phases = (
        () if eager_only else (('forward', forward), ('forward_backward', training))
    )
    for phase, functions in device_phases:
        samples = {name: [] for name in models}
        rng = random.Random(51)
        orders[phase] = []
        for _ in range(repeats):
            order = list(models)
            rng.shuffle(order)
            orders[phase].append(order.copy())
            for name in order:
                timing = graph_time(
                    functions[name],
                    inner=1,
                    replays=replays,
                    repeats=1,
                    warm_replays=10,
                )
                samples[name].append(timing['median_us'])
        timings[phase] = {
            name: {'median_us': statistics.median(values), 'samples_us': values}
            for name, values in samples.items()
        }
        print(f'Measured {phase}', flush=True)
    memory = {}
    for name, function in training.items():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        y = function()
        torch.cuda.synchronize()
        memory[name] = {
            'incremental_peak_bytes': torch.cuda.max_memory_allocated() - before,
            'parameter_bytes': sum(
                p.numel() * p.element_size() for p in models[name].parameters()
            ),
            'gradient_bytes': sum(
                p.grad.numel() * p.grad.element_size()
                for p in models[name].parameters()
                if p.grad is not None
            ),
            'input_bytes': inputs[name].numel() * inputs[name].element_size(),
            'output_bytes': y.numel() * y.element_size(),
        }
        del y
    for function in training.values():
        for _ in range(3):
            function()
    eager_samples = {name: [] for name in training}
    rng = random.Random(152)
    for _ in range(repeats):
        order = list(training)
        rng.shuffle(order)
        for name in order:
            eager_samples[name].append(
                eager_time(training[name], steps=eager_steps, repeats=1)['median_us']
            )
    eager = {
        name: {'median_us': statistics.median(values), 'samples_us': values}
        for name, values in eager_samples.items()
    }
    if profile:
        phase_names = tuple(models)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            for _ in range(3):
                for name, function in training.items():
                    with torch.profiler.record_function(f'tabpack/{name}'):
                        function()
        trace = output / 'trace.json'
        prof.export_chrome_trace(str(trace))
        (output / 'operators.txt').write_text(
            prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=60)
        )
        summary = summarize_trace(json.loads(trace.read_text()), 3, True, phase_names)
        (output / 'trace-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return {
        'widths': widths,
        'depths': source.depths,
        'batch': batch,
        'correctness': correctness,
        'incoming_gradient_layouts': gradient_layouts,
        'timings': timings,
        'randomized_device_orders': orders,
        'eager_forward_backward': eager,
        'memory': memory,
        'plans': {name: plans_report(model) for name, model in models.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--replays', type=int, default=20)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--eager-steps', type=int, default=25)
    parser.add_argument('--eager-only', action='store_true')
    parser.add_argument('--memory-fraction', type=float, default=0.45)
    configuration = parser.add_mutually_exclusive_group()
    configuration.add_argument(
        '--case',
        choices=('all', 'mixed_depth', 'vocabulary', 'uniform', 'skewed'),
        default='all',
    )
    configuration.add_argument(
        '--config',
        type=Path,
        help=(
            'Read geometry and seed from a benchmark or sampled-ensemble config; '
            'data stays synthetic'
        ),
    )
    parser.add_argument('--no-profile', action='store_true')
    args = parser.parse_args()
    if min(args.repeats, args.replays, args.eager_steps) < 1 or (
        args.batch is not None and args.batch < 1
    ):
        parser.error('repeats, replays, batch and eager-steps must be positive')
    if not 0 < args.memory_fraction <= 1:
        parser.error('memory-fraction must be in (0, 1]')
    config = None
    sampled_config = None
    hyperparameters = None
    if args.config is not None:
        raw_config = json.loads(args.config.read_text())
        if 'n_models' in raw_config:
            from dev.pepilica.scripts.run_diverse_california import sample_hyperparameters

            sampled_config = raw_config
            hyperparameters = sample_hyperparameters(sampled_config)
        else:
            config = Config(**raw_config)
            config.validate()
            if config.dtype != 'bfloat16':
                parser.error('The compact hidden-stack benchmark uses BF16')
    seed = sampled_config['seed'] if sampled_config else config.seed if config else 111
    args.output.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    report: dict[str, Any] = {
        'created_utc': datetime.datetime.now(datetime.UTC).isoformat(),
        'scope': (
            'Connected synthetic hidden stacks with ReLU, first-order autograd, '
            'bias gradients and input gradients. No dropout, loss, prefix, head '
            'or optimizer. Boundary packing, compilation and metadata excluded. '
            'Compact activations are [batch, sum(widths)]; all buffers BF16. '
            'Both padded controls use the current native BF16 mask semantics; '
            'dense evaluates all member slots, indexed skips depth-inactive members. '
            'Parameters and gradients are BF16 in every variant. CUDA Graph device '
            'timing and synchronized eager wall timing are reported separately.'
        ),
        'protocol': {
            'config_source': str(args.config.resolve()) if args.config else None,
            'config_scope': 'Geometry and initialization seed; no dataset or optimizer',
            'initialization_seed': seed,
            'repeats': args.repeats,
            'graph_replays': args.replays,
            'eager_steps_per_sample': args.eager_steps,
            'eager_only': args.eager_only,
            'warm_graph_replays': 10,
            'allocator_memory_fraction': args.memory_fraction,
            'reference': 'padded_indexed is the native depth-aware training baseline',
            'shared_native_hyperparameters': (
                'Identical widths/depths use scalar native settings; heterogeneous '
                'packs use lists. This avoids redundant masks for uniform packs.'
            ),
            'memory': (
                'All variants coexist; incremental peaks are transient allocations, '
                'not independent model peaks.'
            ),
        },
        'environment': {
            'torch': torch.__version__,
            'triton': triton.__version__,
            'python': platform.python_version(),
            'python_executable': sys.executable,
            'cuda_runtime': torch.version.cuda,
            'gpu_uuid': str(torch.cuda.get_device_properties(0).uuid),
            'cpu_affinity': sorted(os.sched_getaffinity(0)),
            'torch_threads': torch.get_num_threads(),
            'fp_fusion_environment': os.environ.get('TRITON_DEFAULT_FP_FUSION'),
            'git_commit': subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], text=True
            ).strip(),
            'packages': sorted(
                f'{d.metadata["Name"]}=={d.version}'
                for d in importlib.metadata.distributions()
            ),
            'gpu': torch.cuda.get_device_name(),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'bf16_reduced_precision_reduction': False,
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
    source_paths = [
        Path(__file__),
        Path('dev/pepilica/src/tabpack_experimental/compact_mlp.py'),
        Path('dev/pepilica/src/tabpack_experimental/ragged.py'),
        Path('src/tabpackage/nn.py'),
        Path('dev/pepilica/scripts/benchmark_ragged.py'),
        Path('dev/pepilica/src/tabpack_experimental/benchmark.py'),
        Path('pyproject.toml'),
        Path('uv.lock'),
        Path('.python-version'),
    ]
    if args.config is not None:
        source_paths.append(args.config)
    if hyperparameters is not None:
        source_paths.append(Path('dev/pepilica/scripts/run_diverse_california.py'))
        (args.output / 'hyperparameters.json').write_text(
            json.dumps(hyperparameters, indent=2) + '\n'
        )
    for path in source_paths:
        shutil.copy2(path, sources / path.name)
        report.setdefault('source_sha256', {})[path.name] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    cases = (
        ('uniform', (256,) * 16, 3),
        (
            'mixed_depth',
            tuple((128, 256, 384)[i % 3] for i in range(64)),
            tuple(i % 4 + 1 for i in range(64)),
        ),
        ('vocabulary', HIDDEN_SIZES, 2),
        ('skewed', (64,) * 99 + (1024,), tuple(i % 4 + 1 for i in range(100))),
    )
    if config is not None:
        depths, widths = dimensions(config)
        cases = (('config', tuple(widths), tuple(depths)),)
    elif hyperparameters is not None:
        widths = tuple(h['model']['d_block'] for h in hyperparameters)
        depths = tuple(h['model']['n_blocks'] for h in hyperparameters)
        cases = (('config', widths, depths),)
    batch = (
        args.batch
        if args.batch is not None
        else sampled_config['batch_size']
        if sampled_config
        else config.batch_size
        if config
        else 256
    )
    for name, widths, depths in cases:
        if args.config is None and args.case not in ('all', name):
            continue
        output = args.output / name
        output.mkdir()
        print(f'Starting {name}', flush=True)
        case = run_case(
            widths,
            depths,
            batch,
            output,
            args.repeats,
            not args.no_profile,
            args.replays,
            eager_steps=args.eager_steps,
            eager_only=args.eager_only,
            seed=seed,
        )
        report['cases'][name] = case
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(
            json.dumps(
                {
                    'case': name,
                    'forward_backward_us': {
                        key: value['median_us']
                        for key, value in case['timings']['forward_backward'].items()
                    },
                }
            ),
            flush=True,
        )
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
