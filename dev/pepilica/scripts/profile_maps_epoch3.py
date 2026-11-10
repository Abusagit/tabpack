"""Training-step benchmark on preprocessed Maps Routing without evaluation."""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from dev.pepilica.scripts.fixed_slot_tabpack import FixedSlotTabPackBackend
from dev.pepilica.scripts.run_maps_routing import sample_hyperparameters
from tabpackage._training import make_optimizer_pack, make_shuffled_batch_indices_pack
from tabpack_experimental.benchmark import summarize_trace
from tabpackage.nn import TabPack
from tabpackage.optim import MuonAdamWPack


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variant', choices=('native', 'fixed_grouped'), required=True)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--steps', type=int, default=12)
    parser.add_argument('--memory-fraction', type=float, default=0.75)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    cfg = json.loads(args.config.read_text())
    hp = sample_hyperparameters(cfg)
    meta = json.loads(Path('.cache/maps-routing-paper/metadata.json').read_text())
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    model = (
        TabPack(
            n_numerical_features=meta['n_num'],
            categorical_cardinalities=meta['cat_cardinalities'],
            d_out=1,
            d_block=[x['model']['d_block'] for x in hp],
            n_blocks=[x['model']['n_blocks'] for x in hp],
            dropout=[x['model']['dropout'] for x in hp],
            numerical_embeddings=None,
            forward_impl='indexed',
            pack_size=cfg['n_models'],
        )
        .cuda()
        .train()
    )
    optimizer = make_optimizer_pack(
        model,
        type=MuonAdamWPack,
        muon_lr=[x['optimizer']['muon_lr'] for x in hp],
        adamw_lr=[x['optimizer']['adamw_lr'] for x in hp],
        weight_decay=[x['optimizer']['weight_decay'] for x in hp],
        muon_skip_zero_updates=False,
    )
    bridge = (
        FixedSlotTabPackBackend(model, optimizer, list(range(cfg['n_models'])), hp)
        if args.variant == 'fixed_grouped'
        else None
    )
    p = Path('.cache/maps-routing-paper')
    train = {
        k: torch.as_tensor(np.load(p / f'train_{k}.npy'), device='cuda')
        for k in ('x_num', 'x_cat', 'y')
    }

    def batches(epoch):
        generator = torch.Generator(device='cuda').manual_seed(
            cfg['seed'] + 1000003 * epoch
        )
        return make_shuffled_batch_indices_pack(
            total_size=len(train['y']),
            batch_size=cfg['batch_size'],
            pack_size=cfg['n_models'],
            generator=generator,
            device='cuda',
        )

    def gather(indices):
        return tuple(train[k][indices] for k in ('x_num', 'x_cat', 'y'))

    x, cat, target = gather(batches(1)[0])

    def step(timed):
        with torch.profiler.record_function('tabpack/zero_grad'):
            optimizer.zero_grad(set_to_none=True)
        marks = (
            [torch.cuda.Event(enable_timing=True) for _ in range(4)] if timed else None
        )
        if timed:
            marks[0].record()
        with (
            torch.profiler.record_function('tabpack/forward'),
            torch.autocast('cuda', dtype=torch.bfloat16),
        ):
            pred = model(x, cat)
        with torch.profiler.record_function('tabpack/loss'):
            loss = (pred.squeeze(-1) - target).square().mean()
        if timed:
            marks[1].record()
        with torch.profiler.record_function('tabpack/backward'):
            loss.backward()
        if timed:
            marks[2].record()
        with torch.profiler.record_function('tabpack/optimizer'):
            optimizer.step()
        if timed:
            marks[3].record()
        return marks

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    first = None
    for epoch in (1, 2):
        for indices in batches(epoch):
            x, cat, target = gather(indices)
            t = time.perf_counter()
            step(False)
            if first is None:
                torch.cuda.synchronize()
                first = time.perf_counter() - t
        torch.cuda.synchronize()
        print(
            f'Finished training epoch {epoch}: {time.perf_counter() - started:.1f}s',
            flush=True,
        )
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as profiler:
        samples = []
        for indices in batches(3)[: args.steps]:
            with torch.profiler.record_function('tabpack/gather'):
                x, cat, target = gather(indices)
            torch.cuda.synchronize()
            t = time.perf_counter()
            marks = step(True)
            torch.cuda.synchronize()
            wall = (time.perf_counter() - t) * 1000
            samples.append(
                {
                    'forward_loss_ms': marks[0].elapsed_time(marks[1]),
                    'backward_ms': marks[1].elapsed_time(marks[2]),
                    'optimizer_ms': marks[2].elapsed_time(marks[3]),
                    'wall_ms': wall,
                }
            )
    trace = args.output / 'epoch03-trace.json'
    profiler.export_chrome_trace(str(trace))
    (args.output / 'operators.txt').write_text(
        profiler.key_averages().table(sort_by='self_cuda_time_total', row_limit=100)
    )
    summary = summarize_trace(
        json.loads(trace.read_text()),
        len(samples),
        True,
        ('gather', 'forward', 'loss', 'backward', 'optimizer', 'zero_grad'),
    )
    (args.output / 'trace-summary.json').write_text(
        json.dumps(summary, indent=2) + '\n'
    )
    keys = samples[0].keys()
    result = {
        'variant': args.variant,
        'config': cfg,
        'scope': (
            'Epoch 3 after two training epochs, no evaluation or model removal; '
            'forward includes loss; CUDA events and synchronized wall time'
        ),
        'gpu': torch.cuda.get_device_name(),
        'gpu_uuid': str(torch.cuda.get_device_properties(0).uuid),
        'first_step_seconds': first,
        'warmup_steps': 314,
        'measured_steps': args.steps,
        'median_ms': {k: statistics.median(s[k] for s in samples) for k in keys},
        'samples': samples,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        'widths': [v['model']['d_block'] for v in hp],
        'depths': [v['model']['n_blocks'] for v in hp],
        'backend_calls': None if bridge is None else bridge.calls,
    }
    (args.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['median_ms']), flush=True)
    print('peak_gib', result['peak_allocated_bytes'] / 2**30, flush=True)


if __name__ == '__main__':
    main()
