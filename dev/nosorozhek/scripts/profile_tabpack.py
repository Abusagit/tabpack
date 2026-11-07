"""Capture a configured training epoch and full prediction path."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

import tabpackage
import tabpackage.ensemble
import tabpackage.metrics
import tabpackage.nn
import tabpackage.optim


def _loss(task_type: str, input: Tensor, target: Tensor) -> Tensor:
    if task_type == 'multiclass':
        values = F.cross_entropy(input.transpose(1, 2), target, reduction='none')
    else:
        fn = (
            F.mse_loss
            if task_type == 'regression'
            else F.binary_cross_entropy_with_logits
        )
        values = fn(input.squeeze(-1), target.float(), reduction='none')
    return values.mean(1)


def _metric(task_type: str, *, y_true: Tensor, y_pred: Tensor) -> Tensor:
    if task_type == 'regression':
        return tabpackage.metrics.mse_pack(
            y_true=y_true, y_pred=y_pred.squeeze(-1)
        ).sqrt()
    labels = (
        y_pred.argmax(-1)
        if task_type == 'multiclass'
        else (y_pred.squeeze(-1) >= 0.5).long()
    )
    return tabpackage.metrics.accuracy_pack(y_true=y_true, y_pred=labels)


def _identity(x: Tensor) -> Tensor:
    return x


def _ensemble_metric(
    metric_fn: tabpackage.MetricFn,
    *,
    y_true: Tensor,
    y_pred: Tensor,
) -> Tensor:
    return metric_fn(y_true=y_true, y_pred=y_pred[None])[0]


def _prepare_output(output: Path) -> None:
    if output.exists():
        if Path.cwd().resolve().is_relative_to(output.resolve()):
            raise SystemExit(
                'Cannot replace the working directory or one of its parents'
            )
        try:
            answer = input(
                f'{output.resolve()} already exists. '
                'Replace this directory and all its contents? [y/N] '
            )
        except EOFError:
            answer = ''
        if answer.strip().lower() not in ('y', 'yes'):
            raise SystemExit('Cancelled.')
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=False)


def _sample_hyperparameters(config: dict[str, Any]) -> list[dict[str, Any]]:
    rng = np.random.default_rng(config['seed'])

    def log_uniform(bounds: list[float]) -> float:
        return float(np.exp(rng.uniform(*np.log(bounds))))

    members = []
    for _ in range(config['n_models']):
        model = {
            'd_block': int(rng.choice(config['widths'])),
            'n_blocks': int(rng.integers(config['depth_min'], config['depth_max'] + 1)),
            'dropout': float(rng.uniform(0, config['dropout_max'])),
            'numerical_embeddings': {
                'd_embedding': int(rng.choice(config['embedding_dims'])),
                'init_scale': log_uniform(config['init_scale']),
            },
        }
        members.append(
            {
                'model': model,
                'optimizer': {
                    key: log_uniform(config[key])
                    for key in ('adamw_lr', 'muon_lr', 'weight_decay')
                },
            }
        )
    return members


def _load_experiment(
    config: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, dict[str, Tensor]], dict[str, Any]]:
    with np.load(config['data'], allow_pickle=False) as arrays:
        datasets = {
            part: {
                key: torch.from_numpy(arrays[f'{part}.{key}'])
                for key in ('x_num', 'x_cat', 'y')
                if f'{part}.{key}' in arrays
            }
            for part in ('train', 'val', 'test')
        }
    task_type = config.get('task_type', 'binclass')
    transform = {
        'binclass': torch.sigmoid,
        'multiclass': partial(torch.softmax, dim=-1),
        'regression': _identity,
    }[task_type]
    metric_fn = partial(_metric, task_type)
    metric_name = 'rmse' if task_type == 'regression' else 'accuracy'
    direction = (
        tabpackage.MetricDirection.MINIMIZE
        if task_type == 'regression'
        else tabpackage.MetricDirection.MAXIMIZE
    )
    for dataset in datasets.values():
        dataset['y'] = (
            dataset['y'].float() if task_type == 'regression' else dataset['y'].long()
        )
    train = datasets['train']
    train_kwargs = {
        'seed': config['seed'],
        'n_models': config['n_models'],
        'batch_size': config['batch_size'],
        'eval_batch_size': config['inference_batch_size'],
        'max_epochs': config['max_epochs'],
        'early_stopping_patience': config['model_patience'],
        'compile_model': config.get('compile_model', False),
        'model': {
            'd_out': int(train['y'].max()) + 1 if task_type == 'multiclass' else 1,
            'categorical_cardinalities': [
                max(int(part['x_cat'][:, i].max()) for part in datasets.values()) + 1
                for i in range(train['x_cat'].shape[1])
            ]
            if 'x_cat' in train
            else [],
            'numerical_embeddings': {'type': tabpackage.nn.CosineEmbeddingsPack},
        },
        'per_model_hyperparameters': _sample_hyperparameters(config),
    }
    criterion = tabpackage.EvalCriterionConfig(
        dataset='val',
        metric=metric_name,
        metric_direction=direction,
    )
    train_kwargs.update(
        datasets={
            part: {k: v.to(device) for k, v in dataset.items()}
            for part, dataset in datasets.items()
        },
        prediction_transform=transform,
        loss_fn=partial(_loss, task_type),
        optimizer={'type': tabpackage.optim.MuonAdamWPack},
        eval_metrics={part: {metric_name: metric_fn} for part in ('val', 'test')},
        eval_criterion=criterion,
        ensemble_eval_criterion=criterion,
        ensembles={
            'greedy': tabpackage.EnsembleConfig(
                algorithm={
                    'type': tabpackage.ensemble.greedy_ensemble,
                    'max_ensemble_size': config['max_ensemble_size'],
                },
                prediction_transform=None,
                online=True,
                early_stopping_patience=config['ensemble_patience'],
                algorithm_metric=metric_fn,
                algorithm_metric_direction=direction,
                algorithm_dataset='val',
                algorithm_device=device,
            )
        },
        ensemble_eval_metrics={
            part: {metric_name: partial(_ensemble_metric, metric_fn)}
            for part in ('val', 'test')
        },
        stop_when_all_ensembles_are_finalized=False,
        device=device,
        aux_device='cpu',
        autocast_dtype=(
            getattr(torch, config['autocast_dtype'])
            if config.get('autocast_dtype')
            else None
        ),
        show_progress_bar=False,
    )
    return datasets, train_kwargs


def _save_trace(profiler: torch.profiler.profile, path: Path) -> None:
    stats = profiler.key_averages()
    if not any(event.self_device_time_total for event in stats):
        raise RuntimeError(f'No CUDA activity captured for {path}')
    profiler.export_chrome_trace(str(path))
    print(f'Trace: {path.resolve()}')


def _make_predict_fn(
    trained_pack: tabpackage.TrainedTabPack,
    test_data: dict[str, Tensor],
    train_kwargs: dict[str, Any],
) -> Callable[[int], Tensor]:
    """Build the annotated full prediction path."""
    predictor = trained_pack.make_predictor(
        ensemble_names=['greedy'], autocast_dtype=train_kwargs['autocast_dtype']
    ).to(train_kwargs['device'])
    if train_kwargs.get('compile_model', False):
        predictor.model_pack.forward = torch.compile(
            predictor.model_pack.forward, options={'emulate_precision_casts': True}
        )

    predictor.model_pack.forward = torch.profiler.record_function('inference.forward')(
        predictor.model_pack.forward
    )
    predictor.prediction_transform = torch.profiler.record_function(
        'inference.postprocess'
    )(predictor.prediction_transform)
    for head in predictor.ensembles.values():
        head.forward = torch.profiler.record_function('inference.ensemble_average')(
            head.forward
        )

    def predict(batch_size: int) -> Tensor:
        with torch.profiler.record_function('inference.host_to_device'):
            features = {
                k: v.to(train_kwargs['device'])
                for k, v in test_data.items()
                if k != 'y'
            }
        predictions = []
        for start in range(0, len(test_data['y']), batch_size):
            batch = {k: v[start : start + batch_size] for k, v in features.items()}
            _, ensemble_predictions = predictor(batch.get('x_num'), batch.get('x_cat'))
            predictions.append(ensemble_predictions['greedy'])
        prediction = torch.cat(predictions)
        with torch.profiler.record_function('inference.device_to_host'):
            return prediction.cpu()

    return predict


def _capture_inference(
    trained_pack: tabpackage.TrainedTabPack,
    test_data: dict[str, Tensor],
    train_kwargs: dict[str, Any],
    output: Path,
    dataset_name: str,
) -> dict[str, Any]:
    predict = _make_predict_fn(
        trained_pack,
        test_data,
        train_kwargs,
    )
    batch_size = train_kwargs['eval_batch_size']

    with torch.inference_mode():
        reference = predict(batch_size)
        torch.cuda.synchronize()
        with (
            torch.profiler.profile() as profiler,
            torch.profiler.record_function('inference.predict'),
        ):
            prediction = predict(batch_size)
    _save_trace(profiler, output / f'{dataset_name}-inference-trace.json')
    torch.testing.assert_close(prediction, reference, rtol=0, atol=0)
    metric_name, metric_fn = next(
        iter(train_kwargs['ensemble_eval_metrics']['test'].items())
    )
    labels = test_data['y']
    expected_score = trained_pack.load_ensemble_metrics('greedy')['test'][metric_name]
    actual_score = metric_fn(y_true=labels, y_pred=prediction)
    # Pack shapes can change bfloat16 rounding between training and prediction.
    rtol = 1e-3 if train_kwargs['autocast_dtype'] == torch.bfloat16 else 1e-5
    torch.testing.assert_close(actual_score, expected_score, rtol=rtol, atol=1e-6)
    return {
        'ensemble': 'greedy',
        'member_ids': trained_pack.load_ensemble_member_ids('greedy').tolist(),
        'member_steps': trained_pack.load_ensemble_member_steps('greedy').tolist(),
        'training_step': trained_pack.get_training_loop_step(),
        'test_rows': len(test_data['y']),
        'batch_size': train_kwargs['eval_batch_size'],
        'test_metric': metric_name,
        'test_score': actual_score.item(),
        'max_prediction_error': (prediction - reference).abs().max().item(),
        'autocast_dtype': str(train_kwargs['autocast_dtype']),
        'ensemble_averaging_device': str(train_kwargs['device']),
        'compile_model': train_kwargs.get('compile_model', False),
    }


def _capture_train(
    train_kwargs: dict[str, Any],
    output: Path,
    dataset_name: str,
) -> tabpackage.TrainedTabPack:
    train_kwargs = {**train_kwargs, 'max_epochs': 1}
    with (
        torch.profiler.profile() as profiler,
        torch.profiler.record_function('training'),
    ):
        trained_pack = tabpackage.train(**train_kwargs)
    _save_trace(profiler, output / f'{dataset_name}-training-trace.json')
    return trained_pack


def capture_traces(
    config: dict[str, Any],
    output: Path,
    *,
    environment: dict[str, Any] | None = None,
) -> tuple[tabpackage.TrainedTabPack, dict[str, Tensor], dict[str, Any]]:
    """Train a pack and save its training/inference trace pair and metrics."""
    dataset_name = Path(config['data']).stem
    datasets, train_kwargs = _load_experiment(config, torch.device('cuda'))
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    start = time.perf_counter()
    trained_pack = _capture_train(train_kwargs, output, dataset_name)
    results = _capture_inference(
        trained_pack,
        datasets['test'],
        train_kwargs,
        output,
        dataset_name,
    )
    results['capture_wall_seconds'] = time.perf_counter() - start
    if environment is not None:
        results['environment'] = environment
    (output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    return trained_pack, datasets['test'], train_kwargs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('output', type=Path, help='Asks before overwriting')
    parser.add_argument(
        '--compile-model',
        action='store_true',
        help='Compile model forward passes for training and inference',
    )
    args = parser.parse_args()
    config: dict[str, Any] = json.loads(args.config.read_text())
    config['compile_model'] = args.compile_model or config.get('compile_model', False)
    _prepare_output(args.output)
    capture_traces(config, args.output)


if __name__ == '__main__':
    main()
