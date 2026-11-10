"""Benchmark Maps Routing on the original TabReD default split."""

import argparse
import contextlib
import functools
import gc
import hashlib
import json
import math
import platform
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
import triton

import tabpackage
import tabpackage._training as training
import tabpackage.ensemble
import tabpackage.nn
import tabpackage.optim

if __package__:
    from .fixed_slot_tabpack import FixedSlotTabPackBackend
else:
    from dev.pepilica.scripts.fixed_slot_tabpack import FixedSlotTabPackBackend
from tabpack_experimental.benchmark import _tree_cpu, summarize_trace
from tabpack_experimental.compact_backbone import CompactBackboneBackend


def dump(path, value):
    def default(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
        if isinstance(x, np.generic):
            return x.item()
        return str(x)

    Path(path).write_text(json.dumps(value, indent=2, default=default) + '\n')


def sample_hyperparameters(config):
    rng = random.Random(config['seed'])

    def loguniform(bounds):
        return math.exp(rng.uniform(*map(math.log, bounds)))

    result = []
    for _ in range(config['n_models']):
        model = {
            'd_block': rng.choice(config['widths']),
            'n_blocks': rng.randint(config['depth_min'], config['depth_max']),
            'dropout': rng.uniform(0, config['dropout_max'])
            if rng.randint(0, 1)
            else 0.0,
        }
        optimizer = {
            key: loguniform(config[key])
            for key in ('adamw_lr', 'muon_lr', 'weight_decay')
        }
        result.append({'model': model, 'optimizer': optimizer})
    return result


def prepare_data(config, archive, data_root, output):
    del config, archive, data_root
    directory = Path('.cache/maps-routing-paper')
    metadata = json.loads((directory / 'metadata.json').read_text())
    arrays = {}
    for part in ('train', 'val', 'test'):
        arrays[part] = {
            key: np.load(directory / f'{part}_{key}.npy', allow_pickle=False)
            for key in ('x_num', 'x_cat', 'y')
        }
    metadata['sizes'] = {part: len(v['y']) for part, v in arrays.items()}
    metadata['dataset'] = 'TabReD Maps Routing, original default split'
    metadata['preprocessing'] = 'Original TabPack build_dataset recipe, seed 0'
    dump(output / 'dataset.json', metadata)
    return arrays, metadata


@dataclass
class OriginalRMSE:
    scale: float

    def __call__(self, *, y_true, y_pred):
        return (y_pred - y_true).square().mean(-1).sqrt() * self.scale


class TrainingBackend:
    def __init__(self, variant, config, independent_shuffle):
        self.variant, self.config = variant, config
        self.independent_shuffle = independent_shuffle
        self.backend = None
        self.statistics = {}
        self.batch_hashes = []

    @contextlib.contextmanager
    def instrument(self):
        factory = training.make_optimizer_pack
        sampler = training.make_shuffled_batch_indices_pack
        remove = training.pack_remove
        ids = list(range(self.config['n_models']))
        epoch = 0

        def make_optimizer(model, **kwargs):
            result = factory(model, **kwargs)
            if self.variant in ('compact', 'fused'):
                self.backend = CompactBackboneBackend(
                    model.backbone, fused=self.variant == 'fused'
                )
            elif self.variant == 'fixed_grouped':
                self.backend = FixedSlotTabPackBackend(
                    model,
                    result,
                    ids,
                    sample_hyperparameters(self.config),
                )
            return result

        def make_batches(**kwargs):
            nonlocal epoch
            if not self.independent_shuffle:
                return sampler(**kwargs)
            epoch += 1
            generator = torch.Generator(device=kwargs['device']).manual_seed(
                self.config['seed'] + 1000003 * epoch
            )
            kwargs['pack_size'] = self.config['n_models']
            batches = sampler(**kwargs, generator=generator)
            selection = torch.tensor(ids, device=kwargs['device'])
            if epoch <= 3:
                self.batch_hashes.append(
                    hashlib.sha256(batches[0].cpu().numpy().tobytes()).hexdigest()
                )
            return [b.index_select(0, selection) for b in batches]

        def remove_members(*args, pack_idx, **kwargs):
            removed = set(pack_idx.tolist())
            ids[:] = [member for i, member in enumerate(ids) if i not in removed]
            return remove(*args, pack_idx=pack_idx, **kwargs)

        with (
            patch.object(training, 'make_optimizer_pack', make_optimizer),
            patch.object(training, 'make_shuffled_batch_indices_pack', make_batches),
            patch.object(training, 'pack_remove', remove_members),
        ):
            try:
                yield
            finally:
                if self.backend is not None:
                    self.statistics = {
                        'compact_calls': self.backend.calls,
                        'fallback_calls': self.backend.fallback_calls,
                        'geometry_rebuilds': self.backend.rebuilds,
                    }
                    self.backend.remove()


class EpochProfile:
    def __init__(self, output, epoch, steps, hyperparameters, enabled):
        self.output, self.epoch, self.steps = output, epoch, steps
        self.start_step = (epoch - 1) * steps
        self.completed_steps = 0
        self.hyperparameters = hyperparameters
        self.enabled, self.captured = enabled, False
        self.profiler: Any = None
        self.optimizer: Any = None

    def warm(self):
        if self.enabled:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ):
                torch.zeros(1, device='cuda').add_(1)
                torch.cuda.synchronize()

    def begin_forward(self, model, batch):
        if (
            self.enabled
            and not self.captured
            and self.profiler is None
            and model.training
            and torch.is_grad_enabled()
            and self.completed_steps == self.start_step
        ):
            torch.cuda.synchronize()
            torch.save(
                {
                    'model': _tree_cpu(model.state_dict()),
                    'optimizer': _tree_cpu(self.optimizer.state_dict()),
                    'batch': _tree_cpu(batch),
                    'rng_cpu': torch.get_rng_state(),
                    'rng_cuda': torch.cuda.get_rng_state(),
                    'hyperparameters': self.hyperparameters,
                    'completed_steps': self.completed_steps,
                },
                self.output / 'profile_start.pt',
            )
            dump(
                self.output / 'profile_geometry.json',
                {
                    'pack_size': model.pack_size,
                    'blocks': [
                        {
                            'input_widths': block.linear.in_features,
                            'output_widths': block.linear.out_features,
                            'padded_weight_shape': list(block.linear.weight.shape),
                        }
                        for block in model.backbone.blocks
                    ],
                },
            )
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
            )
            self.profiler.start()
            print(f'Profiling training epoch {self.epoch}', flush=True)

    def after_zero_grad(self):
        self.completed_steps += 1
        if (
            self.profiler is not None
            and self.completed_steps == self.start_step + self.steps
        ):
            torch.cuda.synchronize()
            self.profiler.stop()
            trace = self.output / f'epoch{self.epoch:02d}-trace.json'
            self.profiler.export_chrome_trace(str(trace))
            (self.output / 'operators.txt').write_text(
                self.profiler.key_averages().table(
                    sort_by='self_cuda_time_total',
                    row_limit=100,
                )
            )
            phases = (
                'forward',
                'loss',
                'backward',
                'optimizer',
                'zero_grad',
                'evaluation',
            )
            summary = summarize_trace(
                json.loads(trace.read_text()), self.steps, True, phases
            )
            if not summary['cuda_kernels_available']:
                raise RuntimeError('Profile contains no CUDA kernels')
            dump(self.output / 'trace-summary.json', summary)
            self.profiler = None
            self.captured = True
            print('Epoch profile and replay checkpoint saved', flush=True)

    @contextlib.contextmanager
    def instrument(self):
        if not self.enabled:
            yield
            return
        original_forward = training.forward
        original_factory = training.make_optimizer_pack
        original_backward = torch.Tensor.backward

        def forward(model, batch, **kwargs):
            self.begin_forward(model, batch)
            label = 'forward' if model.training else 'evaluation'
            with torch.profiler.record_function(f'tabpack/{label}'):
                return original_forward(model, batch, **kwargs)

        def factory(*args, **kwargs):
            optimizer: Any = original_factory(*args, **kwargs)
            self.optimizer = optimizer
            step, zero_grad = optimizer.step, optimizer.zero_grad

            def measured_step(*a, **kw):
                with torch.profiler.record_function('tabpack/optimizer'):
                    return step(*a, **kw)

            def measured_zero(*a, **kw):
                with torch.profiler.record_function('tabpack/zero_grad'):
                    result = zero_grad(*a, **kw)
                self.after_zero_grad()
                return result

            optimizer.step, optimizer.zero_grad = measured_step, measured_zero
            return optimizer

        def backward(tensor, *args, **kwargs):
            with torch.profiler.record_function('tabpack/backward'):
                return original_backward(tensor, *args, **kwargs)

        with (
            patch.object(training, 'forward', forward),
            patch.object(training, 'make_optimizer_pack', factory),
            patch.object(torch.Tensor, 'backward', backward),
        ):
            try:
                yield
            finally:
                if self.profiler is not None:
                    self.profiler.stop()
                    self.profiler = None
                self.optimizer = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument(
        '--data-archive',
        type=Path,
        default=Path(__file__).resolve().parents[3]
        / '.cache/downloads/tabpack-data.tar.gz',
    )
    parser.add_argument(
        '--data-root', type=Path, default=Path('.cache/data/california')
    )
    parser.add_argument('--max-epochs', type=int)
    parser.add_argument('--no-profile', action='store_true')
    parser.add_argument('--profile-steps', type=int, default=20)
    parser.add_argument(
        '--variant',
        choices=('native', 'compiled', 'compact', 'fused', 'fixed_grouped'),
        default='native',
    )
    parser.add_argument('--independent-shuffle', action='store_true')
    parser.add_argument('--memory-fraction', type=float, default=0.45)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.max_epochs is not None and args.max_epochs < 1:
        parser.error('--max-epochs must be positive')
    if not 0 < args.memory_fraction <= 1:
        parser.error('--memory-fraction must be in (0, 1]')
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.config, args.output / 'config.json')
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    hyperparameters = sample_hyperparameters(config)
    dump(args.output / 'hyperparameters.json', hyperparameters)
    arrays, metadata = prepare_data(
        config, args.data_archive, args.data_root, args.output
    )
    datasets = {
        name: {k: torch.as_tensor(v, device='cuda') for k, v in arrays[name].items()}
        for name in ('train', 'val')
    }
    steps = math.ceil(len(arrays['train']['y']) / config['batch_size'])
    profile = EpochProfile(
        args.output,
        config['profile_epoch'],
        steps,
        hyperparameters,
        not args.no_profile,
    )
    if not 1 <= args.profile_steps <= steps:
        raise ValueError('profile-steps must be within the epoch')
    profile.start_step = (config['profile_epoch'] - 1) * steps
    profile.steps = args.profile_steps
    profile.warm()
    metric = OriginalRMSE(metadata['target_std'])

    def loss_fn(input, target):
        with torch.profiler.record_function('tabpack/loss'):
            return (input.squeeze(-1) - target).square().mean(1)

    criterion = tabpackage.EvalCriterionConfig(
        dataset='val',
        metric='rmse',
        metric_direction=tabpackage.MetricDirection.MINIMIZE,
    )
    environment = {
        'python': platform.python_version(),
        'torch': torch.__version__,
        'triton': triton.__version__,
        'cuda_runtime': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(),
        'gpu_uuid': str(torch.cuda.get_device_properties(0).uuid),
        'backend': args.variant,
        'precision': 'BF16 autocast, FP32 native master parameters and optimizer state',
        'compact_scope': 'Connected blocks after the native first block; native head',
        'independent_shuffle': args.independent_shuffle,
        'compile_model': args.variant == 'compiled',
        'compile_optimizer_muon_newton_schulz': args.variant == 'compiled',
        'memory_fraction': args.memory_fraction,
        'sampling': 'Independent uniform categorical choices; log-uniform ranges; '
        'half zero dropout, otherwise Uniform(0, dropout_max), matching the example',
        'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
        'epochs_override': args.max_epochs,
        'n_steps_per_epoch': steps,
    }
    dump(args.output / 'environment.json', environment)
    sources = args.output / 'sources'
    sources.mkdir()
    paths = [
        Path('dev/pepilica/scripts/run_diverse_california.py'),
        Path('pyproject.toml'),
        Path('uv.lock'),
        *Path('src/tabpackage').rglob('*.py'),
        *Path('dev/pepilica/src/tabpack_experimental').rglob('*.py'),
        Path('dev/pepilica/pyproject.toml'),
    ]
    if args.variant == 'fixed_grouped':
        paths.extend(
            Path(name)
            for name in (
                'dev/pepilica/scripts/fixed_slot_tabpack.py',
                'dev/pepilica/scripts/fixed_slot_mlp.py',
                'dev/pepilica/scripts/prototype_fixed_slots.py',
            )
        )
    hashes = {}
    for path in paths:
        destination = sources / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    dump(args.output / 'source_sha256.json', hashes)
    print(
        json.dumps(
            {
                'sizes': metadata['sizes'],
                'steps_per_epoch': steps,
                'models': len(hyperparameters),
            }
        ),
        flush=True,
    )
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    backend = TrainingBackend(args.variant, config, args.independent_shuffle)
    with backend.instrument(), profile.instrument():
        trained: Any = tabpackage.train(
            datasets=datasets,
            n_models=config['n_models'],
            model={
                'categorical_cardinalities': metadata['cat_cardinalities'],
                'd_out': 1,
                'numerical_embeddings': None,
                'forward_impl': 'indexed',
            },
            prediction_transform=functools.partial(torch.squeeze, dim=-1),
            loss_fn=loss_fn,
            optimizer={
                'type': tabpackage.optim.MuonAdamWPack,
                'muon_skip_zero_updates': False,
            },
            batch_size=config['batch_size'],
            max_epochs=args.max_epochs or config['max_epochs'],
            early_stopping_patience=config['model_patience'],
            per_model_hyperparameters=hyperparameters,
            eval_metrics={'val': {'rmse': metric}},
            eval_criterion=criterion,
            eval_batch_size=config['inference_batch_size'],
            ensembles={
                'greedy': tabpackage.EnsembleConfig(
                    algorithm={
                        'type': tabpackage.ensemble.greedy_ensemble,
                        'max_ensemble_size': config['max_ensemble_size'],
                    },
                    online=True,
                    prediction_transform=None,
                    early_stopping_patience=config['ensemble_patience'],
                    algorithm_metric=metric,
                    algorithm_metric_direction=tabpackage.MetricDirection.MINIMIZE,
                    algorithm_dataset='val',
                    algorithm_device=torch.device('cuda'),
                )
            },
            ensemble_eval_metrics={'val': {'rmse': metric}},
            ensemble_eval_criterion=criterion,
            stop_when_all_ensembles_are_finalized=True,
            seed=config['seed'],
            device='cuda',
            aux_device='cpu',
            autocast_dtype=torch.bfloat16,
            compile_model=args.variant == 'compiled',
            compile_optimizer_muon_newton_schulz=args.variant == 'compiled',
            print_progress=True,
            show_progress_bar=False,
        )
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - start
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    history = trained.load_model_history()
    dump(args.output / 'history.json', history)
    dump(args.output / 'ensemble_history.json', trained.load_ensemble_history('greedy'))
    predictor = trained.make_predictor(
        ensemble_names=['greedy'], autocast_dtype=torch.bfloat16
    )
    predictor.cuda()
    predictions = {}
    scores = {}
    with torch.inference_mode():
        for name in ('val', 'test'):
            parts = []
            for begin in range(
                0, len(arrays[name]['y']), config['inference_batch_size']
            ):
                x = torch.as_tensor(
                    arrays[name]['x_num'][
                        begin : begin + config['inference_batch_size']
                    ],
                    device='cuda',
                )
                cat = torch.as_tensor(
                    arrays[name]['x_cat'][
                        begin : begin + config['inference_batch_size']
                    ],
                    device='cuda',
                )
                _, ensemble_predictions = predictor(x, cat)
                parts.append(ensemble_predictions['greedy'].float().cpu())
            normalized = torch.cat(parts)
            if not torch.isfinite(normalized).all():
                raise RuntimeError('Nonfinite selected-ensemble predictions')
            predictions[name] = (
                normalized.numpy() * metadata['target_std'] + metadata['target_mean']
            )
            true = arrays[name]['y'] * metadata['target_std'] + metadata['target_mean']
            scores[name] = float(np.sqrt(np.mean((predictions[name] - true) ** 2)))
    torch.save(predictor.cpu(), args.output / 'predictor.pt')
    np.savez_compressed(args.output / 'predictions.npz', **predictions)
    report = {
        'config': config,
        'environment': environment,
        'dataset': metadata,
        'wall_seconds': wall_seconds,
        'training_loop_seconds': trained.get_training_loop_time(),
        'training_steps': trained.get_training_loop_step(),
        'epochs': trained.get_training_loop_step() / steps,
        'ensemble_member_ids': trained.load_ensemble_member_ids('greedy'),
        'ensemble_member_steps': trained.load_ensemble_member_steps('greedy'),
        'rmse': scores,
        'native_validation_rmse': trained.load_ensemble_metrics('greedy'),
        'profile_epoch': config['profile_epoch'] if profile.captured else None,
        'variant': args.variant,
        'peak_allocated_bytes': peak_allocated,
        'peak_reserved_bytes': peak_reserved,
        'backend_statistics': backend.statistics,
        'first_three_epoch_batch_hashes': backend.batch_hashes,
        'compile_counters': {
            str(name): dict(counts)
            for name, counts in torch._dynamo.utils.counters.items()
        },
    }
    dump(args.output / 'report.json', report)
    print(
        json.dumps(
            {
                'rmse': scores,
                'steps': report['training_steps'],
                'wall_seconds': wall_seconds,
                'profiled': profile.captured,
            }
        ),
        flush=True,
    )
    gc.collect()


if __name__ == '__main__':
    main()
