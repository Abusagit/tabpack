"""Paired correctness and event timings for experimental grouped Muon, no profiler.

Run through gpu_run.py. Reports raw samples, device UUID, environment, sampled
architectures, preparation costs and errors. No tolerance is fitted to these data.
"""

# The closures below are consumed synchronously within their loop iteration.
# ruff: noqa: B023, E501

import argparse
import copy
import hashlib
import json
import os
import random
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

from tabpackage import nn, optim
from tabpackage._muon_grouped import GroupedNewtonSchulz, extension
from tabpackage._training import make_optimizer_pack


def differences(actual, reference):
    actual, reference = actual.float(), reference.float()
    delta = actual - reference
    return {
        'max_abs': delta.abs().max().item(),
        'rms': delta.square().mean().sqrt().item(),
        'relative_l2': (delta.norm() / reference.norm().clamp_min(1e-30)).item(),
        'finite': bool(actual.isfinite().all()),
        'exact': bool(torch.equal(actual, reference)),
    }


def summary(samples):
    return {
        'median': statistics.median(samples),
        'p10': float(np.percentile(samples, 10)),
        'p90': float(np.percentile(samples, 90)),
        'minimum': min(samples),
        'maximum': max(samples),
        'samples': samples,
    }


def timed(fn, count=1):
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    torch.cuda.synchronize()
    wall = time.perf_counter()
    start.record()
    for _ in range(count):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / count, (time.perf_counter() - wall) * 1000 / count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='dev/veleboks/grouped_gemm/experiment.json')
    parser.add_argument('--device-index', type=int, default=0)
    args = parser.parse_args()
    os.environ['PATH'] = (
        str(Path.home() / 'tabpack/.venv/bin') + ':' + os.environ['PATH']
    )
    torch.cuda.set_device(args.device_index)
    torch.set_num_threads(4)
    config = json.loads(Path(args.config).read_text())
    out = Path('../output')
    out.mkdir(exist_ok=True)
    device = torch.device('cuda', args.device_index)
    prop = torch.cuda.get_device_properties(device)
    report = {
        'config': config,
        'device': str(device),
        'gpu': prop.name,
        'uuid': str(prop.uuid),
        'capability': torch.cuda.get_device_capability(device),
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'driver': subprocess.check_output(['nvidia-smi'], text=True),
        'precision_flags': {
            'allow_bf16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            'allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        },
        'limitations': [
            'Shared GPU; other jobs may affect timings.',
            'Train step is forward + loss + backward + optimizer on preloaded batches; excludes validation, sampling, transfers.',
            'Short custom training loop uses native TabPack and optimizer, not online ensemble selection.',
        ],
    }

    def save():
        (out / 'experiment.json').write_text(json.dumps(report, indent=2) + '\n')

    def checkpoint(label):
        save()
        print(label, flush=True)

    try:
        t = time.perf_counter()
        extension()
        report['extension_load_build_seconds'] = time.perf_counter() - t
        # Mixture includes zero matrices at every width, plus fully nonzero updates.
        report['newton_schulz_checks'] = []
        for seed in config['ns_check_seeds']:
            torch.manual_seed(seed)
            widths = torch.tensor(config['widths'] * 2, device=device)
            g = torch.zeros(len(widths), 512, 512, device=device)
            for i, w in enumerate(widths.tolist()):
                if i < len(config['widths']):
                    g[i, :w, :w] = torch.randn(w, w, device=device)
            runtime = GroupedNewtonSchulz()
            start = time.perf_counter()
            got = runtime(g, widths).clone()
            torch.cuda.synchronize()
            setup = time.perf_counter() - start
            expected = optim._zeropower_via_newtonschulz5(g, 5)
            record = {
                'seed': seed,
                'cublas_version': runtime.plan.version,
                'setup_first_call_seconds': setup,
                'all': differences(got, expected),
                'by_width': [],
            }
            assert got.dtype == expected.dtype == torch.bfloat16
            for i, w in enumerate(config['widths']):
                record['by_width'].append(
                    dict(width=w, **differences(got[i, :w, :w], expected[i, :w, :w]))
                )
                assert torch.count_nonzero(got[i, w:, :]) == 0
                assert torch.count_nonzero(got[i, :, w:]) == 0
            assert torch.count_nonzero(got[len(config['widths']) :]) == 0
            assert got.isfinite().all()
            bad = g.clone()
            bad[0, 40, 40] = 1
            try:
                runtime(bad, widths)
            except ValueError:
                record['nonzero_padding_rejected'] = True
            else:
                raise AssertionError('nonzero padding silently discarded')
            # Address changes and stream changes must not reuse stale pointers.
            assert torch.equal(runtime(g.clone(), widths), got)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                other = runtime(g.clone(), widths).clone()
            torch.cuda.current_stream().wait_stream(stream)
            assert torch.equal(other, got)
            record['changed_addresses_and_stream_passed'] = True
            report['newton_schulz_checks'].append(record)
        checkpoint('Newton-Schulz checks completed')

        # Optimizer correctness with identical gradients on each step, including
        # nonzero padded parameter values. Momentum must remain exactly identical.
        report['optimizer_checks'] = []
        for skip in (False, True):
            torch.manual_seed(config['seed'])
            widths_list = config['widths'] * 2
            layer = nn.LinearPack(
                widths_list,
                widths_list,
                bias=False,
                pack_size=len(widths_list),
                device=device,
            )
            twin = copy.deepcopy(layer)
            layers = {'padded': layer, 'grouped': twin}
            opts = {
                name: optim.MuonAdamWPack(
                    [
                        optim.MuonAdamWPack.make_muon_parameter_group(
                            mod, grouped_square=True
                        )
                    ],
                    adamw_lr=config['adamw_lr'],
                    muon_lr=config['muon_lr'],
                    weight_decay=config['weight_decay'],
                    muon_skip_zero_updates=skip,
                    pack_size=len(widths_list),
                    allow_inplace_grad_mutation=False,
                    support_replacement=True,
                    muon_backend=name,
                )
                for name, mod in layers.items()
            }
            initial = layer.weight.detach().clone()
            rows = []
            for step in range(config['optimizer_comparison_steps']):
                grad = torch.zeros_like(layer.weight)
                for i, w in enumerate(widths_list):
                    if i % 3 != 0:
                        grad[i, :w, :w] = torch.randn(w, w, device=device)
                for name, mod in layers.items():
                    mod.weight.grad = grad.clone()
                    opts[name].step()
                momentum = [
                    opts[name].state[mod.weight]['muon_momentum_buffer']
                    for name, mod in layers.items()
                ]
                assert torch.equal(*momentum)
                rows.append(
                    {
                        'step': step + 1,
                        'parameters': differences(twin.weight, layer.weight),
                        'momentum': differences(*momentum),
                    }
                )
                for i, w in enumerate(widths_list):
                    for values in (layer.weight, twin.weight):
                        # Native decay is applied to padding too; its nonzero
                        # values must be preserved, with exactly the same rounding.
                        assert torch.equal(values[i, w:, :], layer.weight[i, w:, :])
                        assert torch.equal(values[i, :, w:], layer.weight[i, :, w:])
                        if w < 512:
                            assert values[i, w:, :].count_nonzero() > 0
            # Replacement resets the selected state and refreshes widths.
            idx = torch.tensor([1], device=device)
            for name, mod in layers.items():
                mod.replace(pack_idx=idx, in_features=[32], out_features=[32])
                # Replacement initial values are paired explicitly below.
                opts[name].replace(pack_idx=idx, module=mod)
            twin.weight.data.copy_(layer.weight)
            assert opts['grouped'].param_groups[0]['muon_grouped_widths'][1] == 32
            # Removal remaps parameters and group metadata; rebuild on next call.
            for name, mod in layers.items():
                mapping = mod.remove(pack_idx=idx)
                optim.optimizer_pack_remove(
                    opts[name], pack_idx=idx, old_to_new_parameters=mapping
                )
            grad = torch.zeros_like(layer.weight)
            kept_widths = [w for i, w in enumerate(widths_list) if i != 1]
            for i, w in enumerate(kept_widths):
                grad[i, :w, :w] = torch.randn(w, w, device=device)
            for name, mod in layers.items():
                mod.weight.grad = grad.clone()
                opts[name].step()
            assert torch.equal(
                opts['padded'].state[layer.weight]['muon_momentum_buffer'],
                opts['grouped'].state[twin.weight]['muon_momentum_buffer'],
            )
            report['optimizer_checks'].append(
                {
                    'skip_zero_updates': skip,
                    'steps': rows,
                    'replacement_and_removal': True,
                    'after_removal': differences(twin.weight, layer.weight),
                    'initial_parameter_norm': initial.norm().item(),
                }
            )
        checkpoint('Optimizer correctness checks completed')

        # Historical width/depth/embedding/dropout sampling in the same RNG order.
        rng = random.Random(config['seed'])
        sampled = []

        def loguniform(low, high):
            return float(np.exp(rng.uniform(np.log(low), np.log(high))))

        for _ in range(config['n_models']):
            sampled.append(
                {
                    'width': rng.choice(config['widths']),
                    'depth': rng.randint(config['depth_min'], config['depth_max']),
                    'dropout': rng.uniform(0, 0.7) if rng.randrange(2) else 0.0,
                    'embedding': rng.choice([4, 8, 16, 32, 48, 64]),
                    'init_scale': loguniform(0.001, 20),
                    'adamw_lr': loguniform(3e-5, 0.01),
                    'muon_lr': loguniform(0.0003, 0.15),
                    'weight_decay': loguniform(0.0001, 2.0),
                }
            )
        report['sampled_models'] = sampled
        torch.manual_seed(config['seed'])
        model = nn.TabPack(
            n_numerical_features=8,
            categorical_cardinalities=[],
            d_out=1,
            n_blocks=[s['depth'] for s in sampled],
            d_block=[s['width'] for s in sampled],
            dropout=[s['dropout'] for s in sampled],
            pack_size=len(sampled),
            numerical_embeddings={
                'type': nn.CosineEmbeddingsPack,
                'd_embedding': [s['embedding'] for s in sampled],
                'init_scale': [s['init_scale'] for s in sampled],
            },
        ).to(device)
        initial_model = copy.deepcopy(model.state_dict())
        raw = fetch_california_housing()
        trainval, test_idx = train_test_split(
            np.arange(len(raw.target)), test_size=0.2, random_state=config['seed']
        )
        train_idx, val_idx = train_test_split(
            trainval, test_size=0.2, random_state=config['seed']
        )
        x_raw = raw.data.astype(np.float32)
        transform = QuantileTransformer(
            n_quantiles=max(min(len(train_idx) // 30, 1000), 10),
            output_distribution='normal',
            subsample=10**9,
            random_state=config['seed'],
        )
        transform.fit(
            x_raw[train_idx]
            + np.random.default_rng(0)
            .normal(0.0, 1e-5, x_raw[train_idx].shape)
            .astype(np.float32)
        )
        x = torch.tensor(transform.transform(x_raw).astype(np.float32), device=device)
        y_mean, y_std = raw.target[train_idx].mean(), raw.target[train_idx].std()
        y = torch.tensor(
            ((raw.target - y_mean) / y_std).astype(np.float32), device=device
        )
        report['dataset'] = {
            'name': 'California Housing sklearn',
            'sha256': hashlib.sha256(
                raw.data.tobytes() + raw.target.tobytes()
            ).hexdigest(),
            'train_indices_sha256': hashlib.sha256(train_idx.tobytes()).hexdigest(),
            'split_sizes': [len(train_idx), len(val_idx), len(test_idx)],
        }
        batch_idx = torch.tensor(train_idx[: config['batch_size']], device=device)
        xb, yb = x[batch_idx], y[batch_idx]
        widths = torch.tensor([s['width'] for s in sampled], device=device)
        g = torch.zeros(len(sampled), 512, 512, device=device)
        for i, s in enumerate(sampled):
            if s['depth'] > 1:
                w = s['width']
                g[i, :w, :w] = torch.randn(w, w, device=device)
        report['covered_layers'] = [
            {
                'block': i,
                'shape': list(b.linear.weight.shape),
                'grouped': i > 0,
                'active_members': sum(s['depth'] > i for s in sampled),
            }
            for i, b in enumerate(model.backbone.blocks)
        ]
        report['timings'] = []
        for skip in (False, True):
            # Both variants restore identical states and gradients before each trial.
            runtime = GroupedNewtonSchulz()

            def base_ns():
                if skip:
                    active = torch.nonzero(g.flatten(1).any(1))[:, 0]
                    z = optim._zeropower_via_newtonschulz5(g.index_select(0, active), 5)
                    torch.zeros_like(g, dtype=z.dtype).index_copy_(0, active, z)
                else:
                    optim._zeropower_via_newtonschulz5(g, 5)

            def grouped_ns():
                active = torch.nonzero(g.flatten(1).any(1))[:, 0] if skip else None
                runtime(g, widths, active_pack_idx=active)

            ns = {'padded': base_ns, 'grouped': grouped_ns}

            def make_opt(backend):
                return make_optimizer_pack(
                    model,
                    type=optim.MuonAdamWPack,
                    adamw_lr=[s['adamw_lr'] for s in sampled],
                    muon_lr=[s['muon_lr'] for s in sampled],
                    weight_decay=[s['weight_decay'] for s in sampled],
                    muon_skip_zero_updates=skip,
                    muon_backend=backend,
                )

            opts = {name: make_opt(name) for name in ns}
            model.load_state_dict(initial_model)
            model.train()
            torch.manual_seed(config['seed'])
            ((model(xb, None).squeeze(-1) - yb) ** 2).mean(1).sum().backward()
            grads = [
                None if p.grad is None else p.grad.detach().clone()
                for p in model.parameters()
            ]
            # Initialize both optimizer states with the same first gradient.
            for name in ns:
                model.load_state_dict(initial_model)
                for p, grad in zip(model.parameters(), grads):
                    p.grad = None if grad is None else grad.clone()
                opts[name].step()
            state = copy.deepcopy(opts['padded'].state_dict())
            model.load_state_dict(initial_model)
            state_model = copy.deepcopy(model.state_dict())
            prep = {}
            for name in ns:
                opts[name].load_state_dict(copy.deepcopy(state))
                for p, grad in zip(model.parameters(), grads):
                    p.grad = None if grad is None else grad.clone()
                torch.cuda.synchronize()
                t = time.perf_counter()
                opts[name].step()
                torch.cuda.synchronize()
                prep[name] = time.perf_counter() - t

            # Restore only mutable optimizer values in place. Loading metadata
            # during a timed trial would invalidate grouped plans and measure setup.
            def restore(name):
                model.load_state_dict(state_model)
                with torch.no_grad():
                    for saved_group, group in zip(
                        state['param_groups'], opts[name].param_groups
                    ):
                        for saved_id, parameter in zip(
                            saved_group['params'], group['params']
                        ):
                            for key, value in state['state'].get(saved_id, {}).items():
                                if isinstance(value, torch.Tensor):
                                    opts[name].state[parameter][key].copy_(value)
                                else:
                                    opts[name].state[parameter][key] = copy.deepcopy(
                                        value
                                    )

            for scope in ('newton_schulz', 'optimizer', 'train_step'):
                samples = {name: {'cuda_ms': [], 'wall_ms': []} for name in ns}

                def action(name):
                    if scope == 'newton_schulz':
                        return ns[name]()
                    if scope == 'train_step':
                        opts[name].zero_grad(set_to_none=True)
                        ((model(xb, None).squeeze(-1) - yb) ** 2).mean(
                            1
                        ).sum().backward()
                    opts[name].step()

                for name in ns:
                    for _ in range(config['warmup']):
                        if scope == 'optimizer':
                            for p, grad in zip(model.parameters(), grads):
                                p.grad = None if grad is None else grad.clone()
                        action(name)
                for repeat in range(config['repeats']):
                    order = (
                        ('padded', 'grouped')
                        if repeat % 2 == 0
                        else ('grouped', 'padded')
                    )
                    for name in order:
                        restore(name)
                        torch.manual_seed(config['seed'] + repeat)
                        if scope == 'optimizer':
                            for p, grad in zip(model.parameters(), grads):
                                p.grad = None if grad is None else grad.clone()
                        # Each optimizer/train sample starts from exactly paired
                        # weights, momentum and RNG; gradient restoration is untimed.
                        count = (
                            config['steps_per_repeat']
                            if scope == 'newton_schulz'
                            else 1
                        )
                        event, wall = timed(lambda: action(name), count)
                        samples[name]['cuda_ms'].append(event)
                        samples[name]['wall_ms'].append(wall)
                entry = {
                    'skip_zero_updates': skip,
                    'scope': scope,
                    'first_step_including_plan_setup_seconds': prep,
                    'timed_steps_per_sample': count,
                    'gradient_restoration_timed': False,
                    'results': {
                        name: {clock: summary(v) for clock, v in clocks.items()}
                        for name, clocks in samples.items()
                    },
                }
                entry['speedup_cuda_median'] = (
                    entry['results']['padded']['cuda_ms']['median']
                    / entry['results']['grouped']['cuda_ms']['median']
                )
                report['timings'].append(entry)
                checkpoint(
                    f'Timing {scope} skip={skip}: speedup {entry["speedup_cuda_median"]:.3f}'
                )
        # Short paired training, fixed batch order and dropout RNG, fresh optimizer.
        report['training'] = {}
        rng_np = np.random.default_rng(config['seed'])
        batch_sequence = [
            torch.tensor(
                rng_np.choice(train_idx, config['batch_size'], replace=False),
                device=device,
            )
            for _ in range(config['training_steps'])
        ]
        for name in ('padded', 'grouped'):
            model.load_state_dict(initial_model)
            opt = make_optimizer_pack(
                model,
                type=optim.MuonAdamWPack,
                adamw_lr=[s['adamw_lr'] for s in sampled],
                muon_lr=[s['muon_lr'] for s in sampled],
                weight_decay=[s['weight_decay'] for s in sampled],
                muon_backend=name,
                muon_skip_zero_updates=False,
            )
            model.train()
            torch.manual_seed(config['seed'])
            losses = []
            t = time.perf_counter()
            for idx in batch_sequence:
                opt.zero_grad(set_to_none=True)
                values = ((model(x[idx], None).squeeze(-1) - y[idx]) ** 2).mean(1)
                assert values.isfinite().all()
                values.sum().backward()
                opt.step()
                losses.append(values.detach().cpu().tolist())
            torch.cuda.synchronize()
            wall = time.perf_counter() - t
            model.eval()
            sse = torch.zeros(len(sampled), device=device)
            with torch.no_grad():
                for chunk in torch.tensor(val_idx, device=device).split(256):
                    pred = model(x[chunk], None).squeeze(-1)
                    assert pred.isfinite().all()
                    sse += ((pred - y[chunk]) ** 2).sum(1)
            rmse = (sse / len(val_idx)).sqrt() * float(y_std)
            assert all(p.isfinite().all() for p in model.parameters())
            assert all(
                v.isfinite().all()
                for s in opt.state.values()
                for v in s.values()
                if isinstance(v, torch.Tensor)
            )
            report['training'][name] = {
                'loss_by_step_and_member': losses,
                'validation_rmse': rmse.tolist(),
                'validation_rmse_median': rmse.median().item(),
                'wall_seconds': wall,
                'parameters_and_optimizer_finite': True,
            }
            checkpoint(f'Training {name} completed')
        report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        report['complete'] = True
    except Exception as exc:
        import traceback

        report['error'] = repr(exc)
        report['traceback'] = traceback.format_exc()
        save()
        raise
    save()


if __name__ == '__main__':
    main()
