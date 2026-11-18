"""Fixed-budget TabPack control measurements on the paper's TabReD splits.

Run through scripts/gpu_run.py. Data must be downloaded with the official
TabReD CLI first; this script never authenticates or downloads implicitly.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import random
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer

import tabpackage
import tabpackage.ensemble
import tabpackage.metrics
import tabpackage.nn
import tabpackage.optim


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(path, name, seed):
    splits = {
        key: np.load(path / 'splits' / 'default' / f'{key}.npy')
        for key in ('train', 'val', 'test')
    }
    # Detect changed artifacts rather than silently switching benchmark sizes.
    expected = {
        'maps-routing': [160019, 59975, 59951],
        'homecredit-default': [267645, 58018, 56001],
    }
    if [len(x) for x in splits.values()] != expected[name]:
        raise ValueError(f'Split sizes differ from TabPack Table 5: {splits}')
    all_idx = np.concatenate(list(splits.values()))
    if len(np.unique(all_idx)) != len(all_idx):
        raise ValueError('Dataset splits overlap or contain duplicate indices')
    files = [path / 'info.json', path / 'y.npy', path / 'x_num.npy']
    files += list((path / 'splits' / 'default').glob('*.npy'))
    files += [p for p in (path / 'x_cat.npy', path / 'x_bin.npy') if p.exists()]
    hashes = {str(p.relative_to(path)): sha256(p) for p in sorted(files)}
    raw = np.load(path / 'x_num.npy', mmap_mode='r')
    num = {k: np.array(raw[idx], dtype=np.float32) for k, idx in splits.items()}
    # Match the paper loader: separate binary numeric columns before normalization.
    merged = np.concatenate(list(num.values()))
    unique = [np.unique(column) for column in merged.T]
    binary = np.array([len(x) == 2 and np.isfinite(x).all() for x in unique])
    cats = {k: [] for k in splits}
    for k in splits:
        if binary.any():
            cats[k].append(num[k][:, binary])
        num[k] = num[k][:, ~binary]
    del merged, raw, unique
    if name == 'homecredit-default':
        normalizer = QuantileTransformer(
            n_quantiles=max(min(len(num['train']) // 30, 1000), 10),
            output_distribution='normal',
            subsample=10**9,
            random_state=seed,
        )
        noise = np.random.RandomState(seed).normal(0, 1e-5, num['train'].shape)
        normalizer.fit(num['train'] + noise.astype(np.float32))
        num = {k: normalizer.transform(x) for k, x in num.items()}
    num = {k: np.nan_to_num(x).astype(np.float32) for k, x in num.items()}
    keep = np.array([len(np.unique(x)) > 1 for x in num['train'].T])
    num = {k: x[:, keep] for k, x in num.items()}
    for kind in ('x_cat', 'x_bin'):
        p = path / f'{kind}.npy'
        if p.exists():
            raw = np.load(p, mmap_mode='r')
            for k, idx in splits.items():
                cats[k].append(np.array(raw[idx]))
    cardinalities = []
    if cats['train']:
        cats = {k: np.column_stack(x).astype(str) for k, x in cats.items()}
        encoder = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1,
            dtype=np.int64,
        ).fit(cats['train'])
        cats = {k: encoder.transform(x) for k, x in cats.items()}
        cardinalities = [len(x) for x in encoder.categories_]
        for k in cats:
            for column, cardinality in enumerate(cardinalities):
                cats[k][cats[k][:, column] == -1, column] = cardinality
    else:
        cats = {}
    raw_y = np.load(path / 'y.npy')
    y = {k: np.array(raw_y[idx]) for k, idx in splits.items()}
    mean, std = 0.0, 1.0
    if name == 'maps-routing':
        mean, std = float(y['train'].mean()), float(y['train'].std())
        y = {k: ((x - mean) / std).astype(np.float32) for k, x in y.items()}
    else:
        y = {k: x.astype(np.int64) for k, x in y.items()}
        if not np.array_equal(np.unique(y['train']), [0, 1]):
            raise ValueError('Expected binary labels 0 and 1')
    datasets = {
        k: {
            'x_num': torch.from_numpy(num[k]),
            'y': torch.from_numpy(y[k]),
            **({'x_cat': torch.from_numpy(cats[k])} if cats else {}),
        }
        for k in splits
    }
    metadata = {
        'source': 'https://github.com/yandex-research/tabred',
        'split': 'default',
        'sha256': hashes,
        'split_sizes': {k: len(v) for k, v in splits.items()},
        'n_num': int(keep.sum()),
        'categorical_cardinalities': cardinalities,
        'target_mean': mean,
        'target_std': std,
    }
    return datasets, metadata


def regression_metric(*, y_true, y_pred):
    return (y_pred - y_true).square().mean(-1).sqrt()


def classification_metric(*, y_true, y_pred):
    return tabpackage.metrics.binary_roc_auc_pack(y_true=y_true, y_score=y_pred)


def regression_loss(pred, target):
    return (pred.squeeze(-1) - target).square().mean(1)


def classification_loss(pred, target):
    return F.binary_cross_entropy_with_logits(
        pred.squeeze(-1), target.float(), reduction='none'
    ).mean(1)


def probability(pred):
    return pred.squeeze(-1).float().sigmoid()


def measure(config, datasets, metadata, seed, output):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)

    def loguniform(bounds):
        return math.exp(rng.uniform(*map(math.log, bounds)))

    params = [
        {
            'model': {
                'd_block': config['width'],
                'n_blocks': rng.randint(1, 3),
                'dropout': rng.uniform(0, 0.5) if rng.randrange(2) else 0.0,
                'numerical_embeddings': {
                    'd_embedding': rng.choice(config['embedding_dims']),
                    'init_scale': loguniform([0.01, 10.0]),
                },
            },
            'optimizer': {
                'adamw_lr': loguniform([0.0001, 0.005]),
                'muon_lr': loguniform([0.001, 0.1]),
                'weight_decay': loguniform([0.001, 1.0]),
            },
        }
        for _ in range(config['n_models'])
    ]
    regression = config['dataset'] == 'maps-routing'
    metric = regression_metric if regression else classification_metric

    def scalar_metric(*, y_true, y_pred):
        return metric(y_true=y_true, y_pred=y_pred[None])[0]

    direction = (
        tabpackage.MetricDirection.MINIMIZE
        if regression
        else tabpackage.MetricDirection.MAXIMIZE
    )
    metric_name = 'rmse' if regression else 'roc_auc'
    criterion = tabpackage.EvalCriterionConfig(
        dataset='val',
        metric=metric_name,
        metric_direction=direction,
    )
    device = torch.device('cuda')
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    trained = tabpackage.train(
        datasets={
            k: {key: x.to(device) for key, x in v.items()}
            for k, v in datasets.items()
            if k != 'test'
        },
        model={
            'categorical_cardinalities': metadata['categorical_cardinalities'],
            'd_out': 1,
            'numerical_embeddings': {'type': tabpackage.nn.CosineEmbeddingsPack},
        },
        prediction_transform=partial(torch.squeeze, dim=-1)
        if regression
        else probability,
        loss_fn=regression_loss if regression else classification_loss,
        optimizer={'type': tabpackage.optim.MuonAdamWPack},
        batch_size=config['batch_size'],
        max_epochs=config['max_epochs'],
        early_stopping_patience=16,
        per_model_hyperparameters=params,
        eval_metrics={'val': {metric_name: metric}},
        eval_criterion=criterion,
        eval_batch_size=config['eval_batch_size'],
        ensembles={
            'greedy': tabpackage.EnsembleConfig(
                algorithm={
                    'type': tabpackage.ensemble.greedy_ensemble,
                    'max_ensemble_size': 32,
                },
                online=True,
                prediction_transform=None,
                early_stopping_patience=32,
                algorithm_metric=metric,
                algorithm_metric_direction=direction,
                algorithm_dataset='val',
                algorithm_device=device,
            )
        },
        ensemble_eval_metrics={'val': {metric_name: scalar_metric}},
        ensemble_eval_criterion=criterion,
        seed=seed,
        device=device,
        aux_device='cpu',
        stop_when_all_ensembles_are_finalized=True,
        autocast_dtype=torch.bfloat16,
        compile_model=config['compile_model'],
        compile_optimizer_muon_newton_schulz=config.get(
            'compile_optimizer_muon_newton_schulz', False
        ),
        show_progress_bar=False,
    )
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - start
    train_peak = torch.cuda.max_memory_allocated()
    history = trained.load_model_history()
    training_result = {
        'config': config,
        'seed': seed,
        'dataset': metadata,
        'sampled_models': params,
        'train_wall_seconds': train_seconds,
        'train_peak_allocated_bytes': train_peak,
        'epochs_completed': len(history),
        'epoch_eval_elapsed_seconds': [x['time'] for x in history],
    }
    output.with_name(output.stem + '-training.json').write_text(
        json.dumps(training_result, indent=2) + '\n'
    )
    predictor = trained.make_predictor(ensemble_names=['greedy']).to(device).eval()
    test = datasets['test']
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        predictions = torch.cat(
            [
                predictor(
                    test['x_num'][i : i + 256].to(device),
                    test['x_cat'][i : i + 256].to(device) if 'x_cat' in test else None,
                )[1]['greedy']
                .float()
                .cpu()
                for i in range(0, len(test['y']), 256)
            ]
        )
        test_score = float(scalar_metric(y_true=test['y'], y_pred=predictions))
        if regression:
            test_score *= metadata['target_std']
        x = test['x_num'][: config['inference_batch_size']].to(device)
        cat = test['x_cat'][: len(x)].to(device) if 'x_cat' in test else None
        reference = predictor(x, cat)
        latencies = {}
        compile_seconds = None
        for mode in ('eager', 'compiled'):
            forward = predictor
            if mode == 'compiled':
                start = time.perf_counter()
                forward = torch.compile(predictor, fullgraph=True)
                actual = forward(x, cat)
                torch.cuda.synchronize()
                compile_seconds = time.perf_counter() - start
                for left, right in (
                    (actual[0], reference[0]),
                    (actual[1]['greedy'], reference[1]['greedy']),
                ):
                    torch.testing.assert_close(left, right, rtol=0.02, atol=0.02)
            for _ in range(10):
                forward(x, cat)
            torch.cuda.synchronize()
            samples = []
            for _ in range(50):
                start = time.perf_counter()
                forward(x, cat)
                torch.cuda.synchronize()
                samples.append(time.perf_counter() - start)
            latencies[mode] = samples
    result = {
        **training_result,
        'test_metric': metric_name,
        'test_score': test_score,
        'inference_compile_seconds': compile_seconds,
        'inference_seconds': latencies,
        'selected_member_ids': trained.load_ensemble_member_ids('greedy').tolist(),
        'selected_member_steps': trained.load_ensemble_member_steps('greedy').tolist(),
    }
    output.write_text(json.dumps(result, indent=2) + '\n')
    print('RESULT', output, test_score, train_seconds, flush=True)
    del trained, predictor
    gc.collect()
    torch.cuda.empty_cache()
    torch._dynamo.reset()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--data-root', default='~/tabpack-data/tabred-v0.1.2')
    parser.add_argument('--output', default='../output')
    parser.add_argument('--environment-only', action='store_true')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if torch.__version__.split('+')[0] != '2.14.0':
        raise RuntimeError(f'Expected PyTorch 2.14.0, got {torch.__version__}')
    environment = {
        'python': sys.version,
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(),
        'cuda_visible_devices': os.getenv('CUDA_VISIBLE_DEVICES'),
        'nvidia_smi': subprocess.check_output(['nvidia-smi'], text=True),
        'packages': sorted(
            f'{p.metadata["Name"]}=={p.version}'
            for p in importlib.metadata.distributions()
        ),
    }
    # Real CUDA execution verifies the wheel/driver combination.
    x = torch.randn(256, 256, device='cuda')
    assert torch.isfinite(x @ x).all()
    torch.cuda.synchronize()
    (output / 'environment.json').write_text(json.dumps(environment, indent=2) + '\n')
    if args.environment_only:
        print('CUDA environment verified:', environment['torch'], environment['gpu'])
        return
    config = json.loads(Path(args.config).read_text())
    torch.set_num_threads(4)
    datasets, metadata = prepare(
        Path(args.data_root).expanduser() / config['dataset'],
        config['dataset'],
        config['preprocessing_seed'],
    )
    (output / 'dataset.json').write_text(json.dumps(metadata, indent=2) + '\n')
    # Record competing GPU activity throughout the measured workload.
    with (output / 'gpu-telemetry.csv').open('w') as log:
        monitor = subprocess.Popen(
            [
                'nvidia-smi',
                '-i',
                os.environ['CUDA_VISIBLE_DEVICES'],
                '--query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm',
                '--format=csv',
                '--loop-ms=1000',
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            for seed in config['seeds']:
                measure(config, datasets, metadata, seed, output / f'seed-{seed}.json')
        finally:
            monitor.terminate()
            monitor.wait()


if __name__ == '__main__':
    main()
