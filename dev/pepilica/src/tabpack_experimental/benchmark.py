import contextlib
import copy
import dataclasses
import datetime
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, TypedDict

import torch
import torch.version
from torch import Tensor

from tabpackage import nn as pack_nn
from tabpackage import optim as pack_optim
from tabpackage._training import make_optimizer_pack

PHASES = ('forward', 'loss', 'zero_grad', 'backward', 'optimizer')
WORKLOADS = ('uniform', 'depth', 'width')


class LinearCost(TypedDict):
    layer: str
    active_members: int
    executed_flops: int
    useful_flops: int


@dataclasses.dataclass(frozen=True)
class Config:
    workload: str = 'depth'
    implementation: str = 'padded'
    data_config: str | None = None
    pack_size: int = 64
    batch_size: int = 256
    width: int = 384
    widths: tuple[int, ...] | None = None
    depth: int = 4
    n_num_features: int = 32
    cat_cardinalities: tuple[int, ...] = (8, 16, 4)
    task: str = 'binclass'
    n_classes: int = 10
    dtype: str = 'bfloat16'
    device: str = 'cuda:0'
    optimizer: str = 'MuonAdamWPack'
    shared_hparams: bool = False
    dropout: float = 0.1
    lr: float = 0.001
    weight_decay: float = 0.01
    muon_lr: float = 0.01
    seed: int = 0
    warmup: int = 10
    steps: int = 30
    repeats: int = 3
    profile_steps: int = 3

    def validate(self):
        for key in (
            'pack_size',
            'batch_size',
            'width',
            'depth',
            'steps',
            'repeats',
            'warmup',
            'profile_steps',
        ):
            if getattr(self, key) < 1:
                raise ValueError(f'{key} must be positive')
        if self.workload not in WORKLOADS:
            raise ValueError(f'Unknown workload: {self.workload}')
        if self.implementation not in ('padded',):
            raise ValueError(f'Unknown implementation: {self.implementation}')
        if self.widths is not None and (
            not self.widths or any(w < 1 or w > self.width for w in self.widths)
        ):
            raise ValueError('widths must be nonempty and within [1, width]')
        if self.n_num_features < 0 or any(c < 1 for c in self.cat_cardinalities):
            raise ValueError('Invalid feature dimensions')
        if not (self.n_num_features or self.cat_cardinalities):
            raise ValueError('At least one feature is required')
        if self.task not in ('regression', 'binclass', 'multiclass'):
            raise ValueError(f'Unknown task: {self.task}')
        if self.task == 'multiclass' and self.n_classes < 3:
            raise ValueError('Use binclass for two classes')
        if self.dtype not in ('float32', 'bfloat16'):
            raise ValueError('Only FP32 and BF16 autocast are supported')
        if self.optimizer not in ('MuonAdamWPack', 'AdamWPack'):
            raise ValueError(f'Unknown optimizer: {self.optimizer}')
        if not 0 <= self.dropout < 1:
            raise ValueError('dropout must be in [0, 1)')
        if min(self.lr, self.weight_decay, self.muon_lr) < 0:
            raise ValueError('Optimizer rates must be nonnegative')
        if torch.device(self.device).type not in ('cpu', 'cuda'):
            raise ValueError('Only CPU and CUDA are supported')


def dimensions(config: Config):
    depths = [
        1 + i % config.depth if config.workload == 'depth' else config.depth
        for i in range(config.pack_size)
    ]
    widths = [
        config.widths[i % len(config.widths)]
        if config.widths is not None
        else max(1, config.width * (1 + i % 4) // 4)
        if config.workload == 'width'
        else config.width
        for i in range(config.pack_size)
    ]
    return depths, widths


def resolved_configs(config: Config):
    depths, widths = dimensions(config)
    model = {
        'n_numerical_features': config.n_num_features,
        'categorical_cardinalities': list(config.cat_cardinalities),
        'd_out': config.n_classes if config.task == 'multiclass' else 1,
        'forward_impl': 'indexed',
        'pack_size': config.pack_size,
        'n_blocks': depths if config.workload == 'depth' else config.depth,
        'd_block': widths
        if config.workload == 'width' or config.widths is not None
        else config.width,
        'dropout': float(config.dropout),
    }
    if config.workload == 'depth':
        model['max_n_blocks'] = config.depth
    if config.workload == 'width' or config.widths is not None:
        model['max_d_block'] = config.width
    optimizer = {
        'type': config.optimizer,
        'lr': config.lr,
        'weight_decay': config.weight_decay,
    }
    if config.optimizer == 'MuonAdamWPack':
        optimizer['muon_lr'] = config.muon_lr
        optimizer['muon_newton_schulz_steps'] = 5
    if not config.shared_hparams:
        rng = random.Random(config.seed)
        model['dropout'] = [
            0.0 if rng.random() < 0.5 else rng.uniform(0, 0.5)
            for _ in range(config.pack_size)
        ]
        for key, low, high in (
            ('lr', 0.0001, 0.005),
            ('weight_decay', 0.001, 1.0),
            ('muon_lr', 0.001, 0.1),
        ):
            if key in optimizer:
                optimizer[key] = [
                    10 ** rng.uniform(math.log10(low), math.log10(high))
                    for _ in range(config.pack_size)
                ]
    return model, optimizer


def make_optimizer(model, optimizer_config):
    options = dict(optimizer_config)
    kind = options.pop('type')
    cls = getattr(pack_optim, kind)
    if kind == 'MuonAdamWPack':
        options['adamw_lr'] = options.pop('lr')
    return make_optimizer_pack(model, type=cls, support_replacement=False, **options)


def padding_report(config: Config):
    depths, widths = dimensions(config)
    d_in = config.n_num_features + sum(config.cat_cardinalities)
    d_out = config.n_classes if config.task == 'multiclass' else 1
    rows: list[LinearCost] = []
    muon_rows: list[LinearCost] = []
    for layer in range(config.depth):
        active = [i for i, d in enumerate(depths) if d > layer]
        executed = (
            2
            * config.batch_size
            * len(active)
            * config.width
            * (d_in if layer == 0 else config.width)
        )
        useful = (
            2
            * config.batch_size
            * sum(widths[i] * (d_in if layer == 0 else widths[i]) for i in active)
        )
        if config.implementation == 'width-bucketed':
            executed = useful
        rows.append(
            {
                'layer': f'backbone.blocks.{layer}.linear',
                'active_members': len(active),
                'executed_flops': executed,
                'useful_flops': useful,
            }
        )
        if active and config.optimizer == 'MuonAdamWPack':

            def ns_flops(m, n):
                r, c = min(m, n), max(m, n)
                return 5 * (4 * r * r * c + 2 * r**3)

            optimizer_executed = config.pack_size * ns_flops(
                config.width,
                d_in if layer == 0 else config.width,
            )
            if config.implementation == 'width-bucketed':
                optimizer_executed = sum(
                    widths.count(width)
                    * ns_flops(
                        width,
                        d_in if layer == 0 else width,
                    )
                    for width in set(widths)
                    if any(
                        depths[i] > layer for i, w in enumerate(widths) if w == width
                    )
                )
            muon_rows.append(
                {
                    'layer': f'backbone.blocks.{layer}.linear.weight',
                    'active_members': len(active),
                    'executed_flops': optimizer_executed,
                    'useful_flops': sum(
                        ns_flops(
                            widths[i],
                            d_in if layer == 0 else widths[i],
                        )
                        for i in active
                    ),
                }
            )
    rows.append(
        {
            'layer': 'output',
            'active_members': config.pack_size,
            'executed_flops': 2
            * config.batch_size
            * d_out
            * (
                sum(widths)
                if config.implementation == 'width-bucketed'
                else config.pack_size * config.width
            ),
            'useful_flops': 2 * config.batch_size * sum(widths) * d_out,
        }
    )
    executed = sum(r['executed_flops'] for r in rows)
    useful = sum(r['useful_flops'] for r in rows)

    def cost(executed, useful):
        return {
            'executed_flops': executed,
            'useful_flops': useful,
            'padding_fraction': 1 - useful / executed if executed else None,
        }

    backward_executed = sum(
        r['executed_flops'] * (1 if i == 0 else 2) for i, r in enumerate(rows)
    )
    backward_useful = sum(
        r['useful_flops'] * (1 if i == 0 else 2) for i, r in enumerate(rows)
    )
    muon_executed = sum(r['executed_flops'] for r in muon_rows)
    muon_useful = sum(r['useful_flops'] for r in muon_rows)
    return {
        'scope': 'Analytical GEMM arithmetic; multiply-add counts as two FLOPs',
        'executed_flops': executed,
        'useful_flops': useful,
        'padding_fraction': 1 - useful / executed,
        'layers': rows,
        'backward_gemms': cost(backward_executed, backward_useful),
        'optimizer_gemms': {**cost(muon_executed, muon_useful), 'layers': muon_rows},
        'all_training_gemms': cost(
            executed + backward_executed + muon_executed,
            useful + backward_useful + muon_useful,
        ),
        'notes': [
            'Top-level totals describe forward linear GEMMs.',
            (
                'Inactive-depth members are excluded from model GEMMs, but Muon '
                'still processes the full parameter pack of each executed block.'
            ),
            (
                'Backward counts weight gradients and input gradients except for '
                'the first layer (raw inputs do not require gradients).'
            ),
            (
                'Optimizer GEMMs count the default five Newton-Schulz iterations; '
                'AdamW has no GEMMs. Excludes all non-GEMM operations and hardware '
                'tile padding.'
            ),
            'Extra FLOPs do not imply an equal recoverable fraction of runtime.',
        ],
    }


def _tree_cpu(value):
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _tree_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_tree_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_tree_cpu(v) for v in value)
    return copy.deepcopy(value)


class Runner:
    def __init__(self, config: Config):
        config.validate()
        if config.data_config is not None:
            raise ValueError('Use make_runner for a real dataset')
        self.config = config
        self.device = torch.device(config.device)
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
            if config.dtype == 'bfloat16' and not torch.cuda.is_bf16_supported():
                raise RuntimeError('This GPU does not support BF16')
        torch.manual_seed(config.seed)
        self.model_config, self.optimizer_config = resolved_configs(config)
        self.model: Any = pack_nn.TabPack(**self.model_config).to(self.device).train()
        self.optimizer = make_optimizer(self.model, self.optimizer_config)
        input_generator = torch.Generator(device=self.device).manual_seed(
            config.seed + 1
        )
        shape = (config.pack_size, config.batch_size)
        self.x_num = (
            torch.randn(
                *shape,
                config.n_num_features,
                device=self.device,
                generator=input_generator,
            )
            if config.n_num_features
            else None
        )
        self.x_cat = (
            torch.stack(
                [
                    torch.randint(
                        c, shape, device=self.device, generator=input_generator
                    )
                    for c in config.cat_cardinalities
                ],
                dim=-1,
            )
            if config.cat_cardinalities
            else None
        )
        if config.task == 'regression':
            self.target = torch.randn(
                shape, device=self.device, generator=input_generator
            )
        else:
            self.target = torch.randint(
                2 if config.task == 'binclass' else config.n_classes,
                shape,
                device=self.device,
                generator=input_generator,
            )
            if config.task == 'binclass':
                self.target = self.target.float()
        self.loss_fn = {
            'binclass': lambda p, y: (
                torch.nn.functional.binary_cross_entropy_with_logits(
                    p, y, reduction='none'
                ).mean(1)
            ),
            'regression': lambda p, y: (p - y).square().mean(1),
            'multiclass': lambda p, y: (
                torch.nn.functional.cross_entropy(
                    p.flatten(0, 1), y.flatten(), reduction='none'
                )
                .view_as(y)
                .mean(1)
            ),
        }[config.task]
        self.last_loss = None
        torch.manual_seed(config.seed + 2)

    @property
    def phases(self):
        return PHASES

    @property
    def scope(self):
        return (
            'Synthetic resident per-member batches; training compute only. '
            'No data sampling, preprocessing, evaluation, or ensemble selection.'
        )

    def sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def forward(self):
        with torch.autocast(
            self.device.type,
            dtype=torch.bfloat16,
            enabled=self.config.dtype == 'bfloat16',
        ):
            return self.model(self.x_num, self.x_cat).squeeze(-1).float()

    def step(self, scope=None):
        scope = scope or (lambda _: contextlib.nullcontext())
        with scope('forward'):
            predictions = self.forward()
        with scope('loss'):
            loss = self.loss_fn(predictions, self.target).sum()
        with scope('zero_grad'):
            self.optimizer.zero_grad()
        with scope('backward'):
            loss.backward()
        with scope('optimizer'):
            self.optimizer.step()
        self.last_loss = loss.detach()

    def snapshot(self):
        self.optimizer.zero_grad()
        self.sync()
        return {
            'model': _tree_cpu(self.model.state_dict()),
            'optimizer': _tree_cpu(self.optimizer.state_dict()),
            'rng_cpu': torch.get_rng_state().clone(),
            'rng_cuda': torch.cuda.get_rng_state(self.device).clone()
            if self.device.type == 'cuda'
            else None,
        }

    def restore(self, snapshot):
        self.optimizer.zero_grad()
        self.model.load_state_dict(snapshot['model'])
        self.optimizer.load_state_dict(copy.deepcopy(snapshot['optimizer']))
        for group in self.optimizer.param_groups:
            for key, value in group.items():
                if isinstance(value, Tensor):
                    group[key] = value.to(self.device)
        torch.set_rng_state(snapshot['rng_cpu'])
        if snapshot['rng_cuda'] is not None:
            torch.cuda.set_rng_state(snapshot['rng_cuda'], self.device)
        self.sync()

    def check_finite(self):
        tensors = [self.last_loss, *self.model.parameters()]
        tensors.extend(p.grad for p in self.model.parameters() if p.grad is not None)
        if not all(t is not None and torch.isfinite(t).all().item() for t in tensors):
            raise RuntimeError('Nonfinite loss, parameters, or gradients')


def describe(values):
    values = sorted(values)

    def percentile(q):
        i = (len(values) - 1) * q
        lo = int(i)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (i - lo)

    return {
        'median': statistics.median(values),
        'p10': percentile(0.1),
        'p90': percentile(0.9),
        'samples': values,
    }


def make_runner(config: Config):
    if config.data_config is not None:
        from tabpack_experimental.real_data import RealDataRunner

        return RealDataRunner(config)
    return Runner(config)


def measure(runner: Runner, snapshot):
    config = runner.config
    phase_samples = {p: [] for p in runner.phases}
    wall_ms, memory = [], []
    for _ in range(config.repeats):
        runner.restore(snapshot)
        if runner.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(runner.device)
            baseline = {
                'baseline_allocated_bytes': torch.cuda.memory_allocated(runner.device),
                'baseline_reserved_bytes': torch.cuda.memory_reserved(runner.device),
            }
        start = time.perf_counter()
        for _ in range(config.steps):
            runner.step()
        runner.sync()
        wall_ms.append((time.perf_counter() - start) * 1000 / config.steps)
        if runner.device.type == 'cuda':
            memory.append(
                {
                    **baseline,
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated(
                        runner.device
                    ),
                    'peak_reserved_bytes': torch.cuda.max_memory_reserved(
                        runner.device
                    ),
                }
            )
        runner.check_finite()
        runner.restore(snapshot)
        events = None
        if runner.device.type == 'cuda':
            events = [
                {
                    p: (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    for p in runner.phases
                }
                for _ in range(config.steps)
            ]
            for step_events in events:
                for begin, end in step_events.values():
                    begin.record()
                    end.record()
            runner.sync()
        for step in range(config.steps):

            @contextlib.contextmanager
            def scope(phase, events=events, step=step):
                if events is None:
                    begin = time.perf_counter()
                    yield
                    phase_samples[phase].append((time.perf_counter() - begin) * 1000)
                else:
                    begin, end = events[step][phase]
                    begin.record()
                    yield
                    end.record()

            runner.step(scope)
        runner.sync()
        if events is not None:
            for step_events in events:
                for phase, (begin, end) in step_events.items():
                    phase_samples[phase].append(begin.elapsed_time(end))
        runner.check_finite()
    total = describe(wall_ms)
    return {
        'wall_ms_per_step': total,
        'model_examples_per_second': config.pack_size
        * config.batch_size
        * 1000
        / total['median'],
        'phases_ms': {p: describe(v) for p, v in phase_samples.items()},
        'phase_clock': 'CUDA events'
        if runner.device.type == 'cuda'
        else 'host perf_counter',
        'memory': memory or None,
        'notes': [
            'Throughput uses a separate uninstrumented, synchronized block.',
            'Phase events include stream idle gaps, not just kernel execution.',
            (
                'Each repetition and measurement mode restores the same post-warmup '
                'weights, optimizer state, and RNG; restoration is outside timing.'
            ),
        ],
    }


def _activity_union_us(events):
    intervals = sorted((e['ts'], e['ts'] + e['dur']) for e in events)
    end, total = float('-inf'), 0.0
    for left, right in intervals:
        total += max(0, right - max(left, end))
        end = max(end, right)
    return total


def summarize_trace(trace, steps, cuda, phase_names=PHASES):
    events = [e for e in trace['traceEvents'] if e.get('ph') == 'X']
    ranges = sorted(
        (e['ts'], e['ts'] + e['dur'], e['name'].split('/')[-1])
        for e in events
        if e.get('cat') == 'user_annotation'
        and e.get('name') in {f'tabpack/{p}' for p in phase_names}
    )

    def phase_at(event):
        ts = event['ts']
        for left, right, phase in ranges:
            if left <= ts < right:
                return phase
        return 'unassigned'

    cpu_ops = [e for e in events if e.get('cat') == 'cpu_op']
    owners = {
        e['args']['External id']: e
        for e in cpu_ops
        if 'External id' in e.get('args', {})
    }
    runtime = {
        e['args']['correlation']: e
        for e in events
        if e.get('cat') in ('cuda_runtime', 'cuda_driver')
        and 'correlation' in e.get('args', {})
    }
    operators = defaultdict(lambda: {'calls': 0, 'kernels': 0, 'kernel_us': 0.0})
    phases = {
        p: {'aten_calls': 0, 'kernels': 0, 'kernel_us': 0.0}
        for p in (*phase_names, 'unassigned')
    }
    for e in cpu_ops:
        if e['name'].startswith('aten::'):
            phase = phase_at(e)
            operators[(phase, e['name'])]['calls'] += 1
            phases[phase]['aten_calls'] += 1
    kernels = [e for e in events if e.get('cat') == 'kernel']
    kernel_names = defaultdict(lambda: {'calls': 0, 'duration_us': 0.0})
    for e in kernels:
        args = e.get('args', {})
        owner = owners.get(args.get('External id'))
        launch = runtime.get(args.get('correlation'))
        phase = (
            phase_at(owner if owner is not None else launch)
            if (owner is not None or launch is not None)
            else 'unassigned'
        )
        phases[phase]['kernels'] += 1
        phases[phase]['kernel_us'] += e['dur']
        name = owner['name'] if owner is not None else '<unknown>'
        operators[(phase, name)]['kernels'] += 1
        operators[(phase, name)]['kernel_us'] += e['dur']
        kernel_names[e['name']]['calls'] += 1
        kernel_names[e['name']]['duration_us'] += e['dur']
    activities = [
        e for e in events if e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')
    ]
    busy = _activity_union_us(activities)
    span = (
        (max(e['ts'] + e['dur'] for e in activities) - min(e['ts'] for e in activities))
        if activities
        else 0
    )
    allocations = [
        e
        for e in trace['traceEvents']
        if e.get('name') == '[memory]' and e.get('args', {}).get('Bytes', 0) > 0
    ]
    return {
        'steps': steps,
        'cuda_kernels_available': bool(kernels) if cuda else None,
        'phases': {
            p: {
                'aten_calls_per_step': v['aten_calls'] / steps,
                'kernel_launches_per_step': v['kernels'] / steps if cuda else None,
                'kernel_time_ms_per_step': v['kernel_us'] / 1000 / steps
                if cuda
                else None,
            }
            for p, v in phases.items()
        },
        'operators': [
            {
                'phase': p,
                'name': name,
                'calls_per_step': v['calls'] / steps,
                'kernel_launches_per_step': v['kernels'] / steps if cuda else None,
                'kernel_time_ms_per_step': v['kernel_us'] / 1000 / steps
                if cuda
                else None,
            }
            for (p, name), v in sorted(
                operators.items(), key=lambda item: -item[1]['kernel_us']
            )
        ],
        'kernels': [
            {
                'name': name,
                'launches_per_step': v['calls'] / steps,
                'time_ms_per_step': v['duration_us'] / 1000 / steps,
            }
            for name, v in sorted(
                kernel_names.items(), key=lambda item: -item[1]['duration_us']
            )
        ],
        'gpu_activity': {
            'union_ms': busy / 1000,
            'envelope_ms': span / 1000,
            'gap_fraction_within_envelope': 1 - busy / span if span else None,
            'memcpy_events': Counter(e['cat'] for e in activities)['gpu_memcpy'],
            'memset_events': Counter(e['cat'] for e in activities)['gpu_memset'],
        }
        if cuda
        else None,
        'allocation_requests': {
            'count': len(allocations),
            'bytes': sum(e['args']['Bytes'] for e in allocations),
        },
        'notes': [
            (
                'Kernel phases use CPU External id or runtime/driver launch '
                'correlation; '
                'asynchronous GPU timestamps are never matched to CPU phase ranges.'
            ),
            (
                'ATen calls include nested operators; kernel attribution counts each '
                'launch once, including launches on autograd worker threads.'
            ),
            'GPU gaps are from an instrumented trace, not hardware utilization.',
            'Allocation requests are tensor allocations, not cudaMalloc calls.',
        ],
    }


def profile(runner, snapshot, output: Path):
    runner.restore(snapshot)
    activities = [torch.profiler.ProfilerActivity.CPU]
    cuda = runner.device.type == 'cuda'
    if cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
    ) as profiler:
        for _ in range(runner.config.profile_steps):
            runner.step(lambda p: torch.profiler.record_function(f'tabpack/{p}'))
            profiler.step()
        runner.sync()
    runner.check_finite()
    trace_path = output / 'trace.json'
    profiler.export_chrome_trace(str(trace_path))
    (output / 'operators.txt').write_text(
        profiler.key_averages(
            group_by_input_shape=True,
        ).table(
            sort_by='self_cuda_time_total' if cuda else 'self_cpu_time_total',
            row_limit=80,
        )
    )
    summary = summarize_trace(
        json.loads(trace_path.read_text()),
        runner.config.profile_steps,
        cuda,
        runner.phases,
    )
    if cuda and not summary['cuda_kernels_available']:
        raise RuntimeError('CUDA trace has no kernels; check CUPTI profiler support')
    return summary


def _command(args):
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=15, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def environment(output: Path):
    source_paths = ['pyproject.toml', 'uv.lock', 'dev/pepilica/scripts/benchmark_current.py']
    source_paths += [str(p) for p in sorted(Path('src/tabpackage').rglob('*.py'))]
    source_paths += [str(p) for p in sorted(Path('dev/pepilica/src/tabpack_experimental').rglob('*.py'))]
    source_paths += ['dev/pepilica/pyproject.toml']
    hashes = {}
    for name in source_paths:
        path = Path(name)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        target = output / 'sources' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    patch = _command(['git', 'diff', '--binary', 'HEAD'])
    if patch:
        (output / 'source.patch').write_text(patch + '\n')
    return {
        'started_at_utc': datetime.datetime.now(datetime.UTC).isoformat(),
        'argv': sys.argv,
        'hostname': platform.node(),
        'platform': platform.platform(),
        'python': platform.python_version(),
        'torch': torch.__version__,
        'cuda_runtime': torch.version.cuda,
        'git_commit': _command(['git', 'rev-parse', 'HEAD']),
        'git_status': _command(['git', 'status', '--short']),
        'source_sha256': hashes,
        'packages': sorted(
            f'{d.metadata["Name"]}=={d.version}'
            for d in importlib.metadata.distributions()
        ),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'cpu_threads': torch.get_num_threads(),
        'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
        'tf32_cudnn': torch.backends.cudnn.allow_tf32,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'nvidia_smi_before': gpu_status(),
    }


def gpu_status():
    return _command(
        [
            'nvidia-smi',
            (
                '--query-gpu=index,uuid,name,driver_version,memory.used,'
                'utilization.gpu,clocks.sm,clocks.mem,power.draw'
            ),
            '--format=csv',
        ]
    )


def run(config: Config, output: Path, with_profile: bool):
    config.validate()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_float32_matmul_precision('highest')
    metadata = environment(output)
    runner = make_runner(config)
    config = runner.config
    if runner.device.type == 'cuda':
        props = torch.cuda.get_device_properties(runner.device)
        metadata['gpu'] = {
            'name': props.name,
            'total_memory': props.total_memory,
            'compute_capability': [props.major, props.minor],
            'uuid': str(getattr(props, 'uuid', 'unavailable')),
        }
    report = {
        'schema_version': 1,
        'config': dataclasses.asdict(config),
        'resolved_model': runner.model_config,
        'resolved_optimizer': runner.optimizer_config,
        'input_seed': config.seed + 1 if config.data_config is None else None,
        'batch_shuffle_seed': config.seed + 3
        if config.data_config is not None
        else None,
        'training_rng_seed': config.seed + 2,
        'input_sha256': {
            name: hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
            if value is not None
            else None
            for name, value in (
                ('x_num', runner.x_num),
                ('x_cat', runner.x_cat),
                ('target', runner.target),
            )
        },
        'environment': metadata,
        'padding': padding_report(config),
        'scope': runner.scope,
        'dataset': getattr(runner, 'dataset_metadata', None),
    }
    (output / 'config.json').write_text(json.dumps(report['config'], indent=2) + '\n')
    for _ in range(config.warmup):
        runner.step()
    runner.sync()
    runner.check_finite()
    snapshot = runner.snapshot()
    report['timing'] = measure(runner, snapshot)
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    if with_profile:
        report['profile'] = profile(runner, snapshot, output)
    metadata['nvidia_smi_after'] = gpu_status()
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
