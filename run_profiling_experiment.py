"""Profile actual tabpackage.train and its selected compiled predictor on California Housing."""
import argparse
import json
import math
import hashlib
import os
import random
import time
from functools import partial
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
import torch
import torch.nn.functional as F
import tabpackage
import tabpackage.nn
import tabpackage.optim
import tabpackage.ensemble
import tabpackage.metrics


def metric(*, y_true, y_pred):
    return ((y_pred - y_true) ** 2).mean(dim=1).sqrt()


def scalar_metric(*, y_true, y_pred):
    return ((y_pred - y_true) ** 2).mean().sqrt()


def loss(pred, target):
    return ((pred.squeeze(-1) - target) ** 2).mean(dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--output', default='.')
    args = parser.parse_args()
    c = json.loads(Path(args.config).read_text())
    root = Path(args.output)
    (root / 'profiles').mkdir(parents=True, exist_ok=True)
    (root / 'results').mkdir(parents=True, exist_ok=True)
    rng = random.Random(c['seed'])
    torch.manual_seed(c['seed'])
    def loguniform(bounds):
        return math.exp(rng.uniform(*map(math.log, bounds)))
    params = []
    for _ in range(c['n_models']):
        params.append({'model': {'d_block': rng.choice(c['widths']),
            'n_blocks': rng.randint(c['depth_min'], c['depth_max']),
            'dropout': rng.uniform(0, c['dropout_max']) if rng.randrange(2) else 0.,
            'numerical_embeddings': {'d_embedding': rng.choice(c['embedding_dims']),
                                     'init_scale': loguniform(c['init_scale'])}},
            'optimizer': {'adamw_lr': loguniform(c['adamw_lr']),
                          'muon_lr': loguniform(c['muon_lr']),
                          'weight_decay': loguniform(c['weight_decay'])}})
    # California Housing, a dataset in the paper. Random split with the same
    # proportions as the paper; it is not claimed to be the paper's exact split.
    device = torch.device('cuda')
    raw = fetch_california_housing()
    raw_hash = hashlib.sha256(raw.data.tobytes() + raw.target.tobytes()).hexdigest()
    trainval, test_idx = train_test_split(np.arange(len(raw.target)), test_size=0.2, random_state=c['seed'])
    train_idx, val_idx = train_test_split(trainval, test_size=0.2, random_state=c['seed'])
    x_raw = raw.data.astype(np.float32)
    y_raw = raw.target.astype(np.float32)
    x_train = x_raw[train_idx]
    transformer = QuantileTransformer(n_quantiles=max(min(len(train_idx)//30, 1000), 10),
        output_distribution='normal', subsample=10**9, random_state=c['seed'])
    noise = np.random.default_rng(0).normal(0., 1e-5, x_train.shape).astype(np.float32)
    transformer.fit(x_train + noise)
    x = transformer.transform(x_raw).astype(np.float32)
    y_mean, y_std = float(y_raw[train_idx].mean()), float(y_raw[train_idx].std())
    y = (y_raw - y_mean) / y_std
    def dataset(idx):
        return {'x_num': torch.as_tensor(x[idx], device=device), 'y': torch.as_tensor(y[idx], device=device)}
    datasets = {'train': dataset(train_idx), 'val': dataset(val_idx)}
    criterion = tabpackage.EvalCriterionConfig(dataset='val', metric='rmse', metric_direction=tabpackage.MetricDirection.MINIMIZE)
    ensemble = tabpackage.EnsembleConfig(algorithm={'type': tabpackage.ensemble.greedy_ensemble, 'max_ensemble_size': c['max_ensemble_size']}, online=True,
        prediction_transform=None, early_stopping_patience=c['ensemble_patience'], algorithm_metric=metric,
        algorithm_metric_direction=tabpackage.MetricDirection.MINIMIZE, algorithm_dataset='val', algorithm_device=device)

    # Instrument an epoch boundary without changing TabPack. Capture one complete
    # steady-state epoch, including validation and the online ensemble update.
    import tabpackage._training as training
    original_batches = training.make_shuffled_batch_indices_pack
    state = {'epoch': 0, 'profiler': None}
    def profiled_batches(*args, **kwargs):
        state['epoch'] += 1
        if state['epoch'] == c['profile_epoch']:
            state['profiler'] = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True)
            state['profiler'].__enter__()
        elif state['epoch'] == c['profile_epoch'] + 1:
            torch.cuda.synchronize()
            state['profiler'].__exit__(None, None, None)
            state['profiler'].export_chrome_trace(str(root / 'profiles' / (c['name'] + '_train.json')))
            state['profiler'] = None
        return original_batches(*args, **kwargs)
    training.make_shuffled_batch_indices_pack = profiled_batches
    torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        trained = tabpackage.train(datasets=datasets, model={'categorical_cardinalities': [], 'd_out': 1,
            'numerical_embeddings': {'type': tabpackage.nn.CosineEmbeddingsPack}}, prediction_transform=partial(torch.squeeze, dim=-1),
            loss_fn=loss, optimizer={'type': tabpackage.optim.MuonAdamWPack}, batch_size=c['batch_size'], max_epochs=c['max_epochs'],
            early_stopping_patience=c['model_patience'], per_model_hyperparameters=params,
            eval_metrics={'val': {'rmse': metric}}, eval_criterion=criterion, eval_batch_size=512,
            ensembles={'greedy': ensemble}, ensemble_eval_metrics={'val': {'rmse': scalar_metric}}, ensemble_eval_criterion=criterion,
            stop_when_all_ensembles_are_finalized=True, seed=c['seed'], device=device, aux_device='cpu', autocast_dtype=None,
            compile_model=c.get('compile_model', False), show_progress_bar=False)
    finally:
        training.make_shuffled_batch_indices_pack = original_batches
        if state['profiler'] is not None:
            state['profiler'].__exit__(None, None, None)
            state['profiler'].export_chrome_trace(str(root / 'profiles' / (c['name'] + '_train.json')))
    torch.cuda.synchronize()
    train_s = time.perf_counter() - start
    # Snapshot Dynamo counters before compiling the inference predictor.
    dynamo_counters = {k: dict(v) for k, v in torch._dynamo.utils.counters.items()} if c.get('compile_model', False) else {}
    dynamo_compile_times = torch._dynamo.utils.compile_times() if c.get('compile_model', False) else ''
    predictor = trained.make_predictor(ensemble_names=['greedy']).to(device)
    x = datasets['val']['x_num'][:c['inference_batch_size']]
    x_cat = None
    with torch.inference_mode(): original = predictor(x, x_cat)
    # PyTorch 2.7 Dynamo cannot trace ParameterPack subclasses. The predictor is
    # inference-only: replace them with ordinary Parameters without changing values.
    for module in predictor.modules():
        for name, param in list(module._parameters.items()):
            if param is not None and type(param) is not torch.nn.Parameter:
                module._parameters[name] = torch.nn.Parameter(param.detach(), requires_grad=False)
    # Verify full graph capture and numerical equivalence before profiling; compilation excluded.
    compiled = torch.compile(predictor, fullgraph=True)
    with torch.inference_mode():
        eager = predictor(x, x_cat)
        torch.testing.assert_close(original[0], eager[0], rtol=0, atol=0)
        torch.testing.assert_close(original[1]['greedy'], eager[1]['greedy'], rtol=0, atol=0)
        out = compiled(x, x_cat)
        torch.testing.assert_close(out[0], eager[0], rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(out[1]['greedy'], eager[1]['greedy'], rtol=1e-4, atol=1e-5)
        for _ in range(5): compiled(x, x_cat)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as iprof:
            for _ in range(10):
                with torch.profiler.record_function('compiled_predictor_forward'):
                    compiled(x, x_cat)
            torch.cuda.synchronize()
        infer_s = time.perf_counter() - start
    iprof.export_chrome_trace(str(root / 'profiles' / (c['name'] + '_inference.json')))
    # Describe the actual selected pack, including its padded allocation and depth masks.
    pack = predictor.model_pack
    def vals(buf, n, default):
        return [default] * n if buf is None else buf.detach().cpu().tolist()
    n = pack.pack_size
    blocks = pack.backbone.blocks
    depths = vals(pack.backbone.n_blocks, n, len(blocks))
    layers = []
    for idx, block in enumerate(blocks):
        linear = next(m for m in block.modules() if isinstance(m, tabpackage.nn.LinearPack))
        ins = vals(linear.in_features, n, linear.max_in_features)
        outs = vals(linear.out_features, n, linear.max_out_features)
        layers.append({'allocated': n * linear.max_in_features * linear.max_out_features,
                       'useful': sum(a*b for a,b,d in zip(ins,outs,depths) if idx < d),
                       'max_in': linear.max_in_features, 'max_out': linear.max_out_features})
    linear = pack.output
    ins = vals(linear.in_features, n, linear.max_in_features)
    outs = vals(linear.out_features, n, linear.max_out_features)
    layers.append({'allocated': n*linear.max_in_features*linear.max_out_features,
                   'useful': sum(a*b for a,b in zip(ins,outs)), 'max_in': linear.max_in_features, 'max_out': linear.max_out_features})
    with torch.inference_mode():
        test_pred = predictor(torch.as_tensor(transformer.transform(x_raw[test_idx]).astype(np.float32), device=device), None)[1]['greedy']
        test_rmse = float((((test_pred * y_std + y_mean) - torch.as_tensor(y_raw[test_idx], device=device)) ** 2).mean().sqrt())
    member_ids = trained.load_ensemble_member_ids('greedy').tolist()
    member_steps = trained.load_ensemble_member_steps('greedy').tolist()
    history = trained.load_model_history()
    result = {'dataset': 'California Housing sklearn', 'dataset_sha256': raw_hash, 'split_sizes': [len(train_idx), len(val_idx), len(test_idx)], 'y_mean': y_mean, 'y_std': y_std,
              'epochs_completed': len(history), 'profiled_epoch': c['profile_epoch'], 'members_ids': member_ids, 'member_steps': member_steps, 'validation_rmse_original_units': float(trained.load_ensemble_metrics('greedy')['val']['rmse']) * y_std, 'test_rmse_original_units': test_rmse,
              'dynamo_training_counters': dynamo_counters, 'dynamo_training_compile_times': dynamo_compile_times, 'config': c, 'sampled_models': params, 'selected_members': n, 'depths': depths, 'linear_layers': layers,
              'train_profile_wall_seconds': train_s, 'inference_profile_10_forwards_seconds': infer_s,
              'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
              'peak_memory_bytes': torch.cuda.max_memory_allocated(), 'pid': os.getpid()}
    (root / 'results' / (c['name'] + '.json')).write_text(json.dumps(result, indent=2))
    print('RESULT', json.dumps({k:v for k,v in result.items() if k not in ('sampled_models','config')}), flush=True)

if __name__ == '__main__': main()
