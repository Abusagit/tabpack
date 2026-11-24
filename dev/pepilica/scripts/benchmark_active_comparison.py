"""Launch native/active-only comparisons through common.run, on one physical GPU.

No training loop or profiler implementation lives here. Each measurement is a
fresh common.run process using the registration-only active_variants wrapper.
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import shlex
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from dev.pepilica.scripts import DEFAULT_VARIANT

ROOT = Path(__file__).resolve().parents[3]
VARIANTS = ('native', 'active_square_matrices_only', 'active_grouped_first')
AVAILABLE_VARIANTS = VARIANTS
DEFAULT_VARIANTS = ('native', DEFAULT_VARIANT)
MODULE = 'dev.pepilica.scripts.active_variants'


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    return sorted(
        {
            p
            for folder in (
                'common',
                'dev/pepilica/src',
                'dev/pepilica/scripts',
                'src/tabpackage',
            )
            for p in (ROOT / folder).rglob('*.py')
            if '__pycache__' not in p.parts
        }
    )


def jobs(args):
    result = [('unprofiled', ['--profile', 'none', '--steps', '0'])]
    result += [
        (f'timing-{i + 1}', ['--profile', 'timing', '--steps', '0'])
        for i in range(args.timing_repeats)
    ]
    result += [('memory', ['--profile', 'memory', '--steps', '0'])]
    trace = [
        '--steps',
        str(args.trace_steps),
        '--warmup',
        '1' if args.trace_steps else '0',
    ]
    if not args.no_trace:
        result += [('torch', ['--profile', 'torch', *trace, '--no-torch-with-stack'])]
    if args.nsys:
        result += [('nsys', ['--profile', 'nvtx', *trace])]
    return result


def command(args, config, variant, job, options, directory):
    cmd = [
        str(ROOT / '.venv/bin/python'),
        '-m',
        MODULE,
        str(config),
        '--variant',
        variant,
        '--epoch',
        str(args.epochs[0]),
        '--max-epochs',
        str(args.epochs[-1]),
        '--measure-epochs',
        *map(str, args.epochs),
        '--out',
        str(directory),
        *options,
    ]
    if args.cpu_affinity:
        cmd = ['taskset', '-c', args.cpu_affinity, *cmd]
    if job == 'nsys':
        cmd = [
            'nsys',
            'profile',
            '--trace=cuda,nvtx,osrt',
            '--sample=none',
            '--cpuctxsw=none',
            '--capture-range=cudaProfilerApi',
            f'--capture-range-end=repeat:{len(args.epochs)}',
            f'--output={directory / "timeline"}',
            *cmd,
        ]
    return cmd


def gpu_state(gpu):
    line = subprocess.check_output(
        [
            'nvidia-smi',
            '-i',
            str(gpu),
            '--query-gpu=uuid,utilization.gpu,memory.free',
            '--format=csv,noheader,nounits',
        ],
        text=True,
    ).strip()
    uuid, util, free = (x.strip() for x in line.split(','))
    return {'uuid': uuid, 'utilization': int(util), 'free_mib': int(free)}


def preflight(gpu, minimum):
    samples = []
    for i in range(6):
        samples.append(gpu_state(gpu))
        if i < 5:
            time.sleep(2)
    print('GPU preflight:', json.dumps(samples), flush=True)
    if any(x['utilization'] for x in samples):
        raise RuntimeError(
            f'GPU {gpu} is active; choose an idle device and a new output directory'
        )
    if min(x['free_mib'] for x in samples) < minimum:
        raise RuntimeError(f'GPU {gpu} has less than {minimum} MiB free')
    return samples


def prepare_data(config, out):
    import numpy as np

    from common import data, run

    arrays, meta = data.load(config)
    hps = run.sample_hyperparameters(config)
    categorical_width = sum(meta['cat_cardinalities'])
    first_inputs = [
        meta['n_num'] * h['model'].get('numerical_embeddings', {}).get('d_embedding', 1)
        + categorical_width
        for h in hps
    ]
    stats = {
        'dataset': config['dataset'],
        'meta': meta,
        'sizes': {s: len(a['y']) for s, a in arrays.items()},
        'one_hot_columns': categorical_width,
        'first_layer_input_widths': first_inputs,
        'train_val_array_mib': sum(
            v.nbytes
            for s, a in arrays.items()
            if s in ('train', 'val')
            for v in a.values()
        )
        / 2**20,
        'split_arrays': {
            s: {k: {'shape': v.shape, 'dtype': str(v.dtype)} for k, v in a.items()}
            for s, a in arrays.items()
        },
        'note': 'CPU preprocessing; all jobs load the same common.data result.',
    }
    if any(not np.isfinite(a['x_num']).all() for a in arrays.values() if 'x_num' in a):
        raise RuntimeError('Non-finite preprocessed numerical inputs')
    dump(out / 'dataset.json', stats)
    dump(out / 'sampled-hyperparameters.json', hps)
    return stats, hps


def epoch_record(report, epoch):
    profile = report['train_profile']
    if 'epochs' in profile:
        return profile['epochs'][str(epoch)]
    return {'completed': True, 'train_profile': profile, 'pack_start': {}}


def summarize(out):
    plan = json.loads((out / 'suite.json').read_text())
    epochs = plan.get('epochs', [plan.get('epoch', 3)])
    rows, uuids = [], set()
    all_completed = True
    for variant in plan['variants']:
        base = out / variant
        raw = json.loads((base / 'unprofiled/report.json').read_text())
        timings = [
            json.loads((base / f'timing-{i + 1}/report.json').read_text())
            for i in range(plan['timing_repeats'])
        ]
        memory = json.loads((base / 'memory/report.json').read_text())
        for report in [raw, *timings, memory]:
            uuids.add(report['environment']['gpu_uuid'])
        for epoch in epochs:
            windows = [epoch_record(r, epoch) for r in timings]
            mem = epoch_record(memory, epoch)
            completed = all(w['completed'] for w in [*windows, mem])
            all_completed &= completed
            counts = [w.get('pack_start', {}).get('count') for w in windows]
            counts = [n for n in counts if n is not None]
            row = {
                'variant': variant,
                'measurement_epoch': epoch,
                'completed': completed,
                'active_models': counts[0]
                if counts and len(set(counts)) == 1
                else None,
                'active_models_min': min(counts) if counts else None,
                'active_models_max': max(counts) if counts else None,
                'memory_active_models': mem.get('pack_start', {}).get('count'),
                'run_max_epochs': epochs[-1],
                'unprofiled_train_wall_s': raw['train_wall_seconds'],
                'unprofiled_training_loop_s': raw['training_loop_seconds'],
                'unprofiled_run_peak_allocated_mib': raw['train_peak_allocated_mib'],
                'warm_epoch_s': (
                    statistics.median(
                        w['train_profile']['window_ms'] / 1000 for w in windows
                    )
                    if completed
                    else None
                ),
            }
            for phase in (
                'gather',
                'forward',
                'loss',
                'backward',
                'optimizer',
                'zero_grad',
                'eval',
            ):
                row[f'{phase}_median_ms'] = (
                    statistics.median(
                        w['train_profile']['phase_ms'][phase]['median'] for w in windows
                    )
                    if completed
                    else None
                )
                m = (
                    mem['train_profile']['phase_memory_mib'][phase]
                    if mem['completed']
                    else {}
                )
                row[f'{phase}_peak_mib'] = m.get('max_peak')
                row[f'{phase}_peak_over_before_mib'] = m.get('max_peak_over_before')
            rows.append(row)
    if len(uuids) != 1:
        raise RuntimeError(f'Measurements came from different GPUs: {uuids}')
    dump(
        out / 'comparison.json',
        {
            'rows': rows,
            'gpu_uuid': next(iter(uuids)),
            'all_measurements_completed': all_completed,
            'timing': 'Repeated step medians; warm epoch includes eval/glue.',
            'memory': 'Separate phase-memory runs; unprofiled peak covers all epochs.',
            'scope': f'Trains through epoch {epochs[-1]}; measures {epochs}. '
            'No resets between epochs; no test evaluation.',
        },
    )
    with (out / 'comparison.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(
        '\nvariant                         epoch active epoch(s) '
        'forward(ms) backward(ms) optimizer(ms)'
    )
    for r in rows:
        if r['completed']:
            print(
                f'{r["variant"]:31} {r["measurement_epoch"]:5} '
                f'{r["active_models"]!s:>6}'
                f' {r["warm_epoch_s"]:8.3f} {r["forward_median_ms"]:11.3f}'
                f' {r["backward_median_ms"]:12.3f} {r["optimizer_median_ms"]:13.3f}'
            )
        else:
            print(
                f'{r["variant"]:31} {r["measurement_epoch"]:5} '
                'measurement unavailable (training ended early)'
            )
    return all_completed


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', type=int, help='physical GPU index, used for every process')
    p.add_argument(
        '--config', type=Path, default=ROOT / 'common/configs/maps_routing.json'
    )
    p.add_argument('--out', type=Path)
    p.add_argument(
        '--variants',
        nargs='+',
        choices=AVAILABLE_VARIANTS,
        default=list(DEFAULT_VARIANTS),
        help='setups to benchmark (default: native control and active_grouped_first)',
    )
    p.add_argument(
        '--epochs',
        type=int,
        nargs='+',
        default=[3, 21],
        help='measurement epochs within each training run (default: 3 21)',
    )
    p.add_argument(
        '--epoch', type=int, help='compatibility alias for a single measurement epoch'
    )
    p.add_argument('--timing-repeats', type=int, default=3)
    p.add_argument(
        '--trace-steps',
        type=int,
        default=20,
        help='0 captures the entire epoch (large trace)',
    )
    p.add_argument('--no-trace', action='store_true')
    p.add_argument('--nsys', action='store_true')
    p.add_argument(
        '--cpu-affinity', default='6,7', help='empty string disables CPU affinity'
    )
    p.add_argument('--min-free-mib', type=int, default=6144)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument(
        '--summarize',
        type=Path,
        help='summarize a completed suite without running training',
    )
    args = p.parse_args(argv)
    if args.summarize:
        summarize(args.summarize.resolve())
        return
    if args.epoch is not None:
        args.epochs = [args.epoch]
    if (
        not args.epochs
        or min(args.epochs) < 2
        or args.epochs != sorted(set(args.epochs))
    ):
        p.error('--epochs must be strictly increasing and >= 2')
    if len(set(args.variants)) != len(args.variants):
        p.error('--variants must be unique')
    if args.gpu is None or args.gpu < 0:
        p.error('--gpu must specify a physical GPU index')
    if args.timing_repeats < 1 or args.trace_steps < 0 or args.min_free_mib < 1:
        p.error('Need epoch >= 2, positive repeats/memory and nonnegative trace steps')
    os.chdir(ROOT)
    config = args.config.resolve()
    config_text = config.read_text()
    cfg = json.loads(config_text)
    if any(
        cfg.get(k)
        for k in (
            'compile_model',
            'compile_model_numerical_embeddings',
            'compile_model_first_block',
            'compile_optimizer_muon_newton_schulz',
        )
    ):
        p.error('Choose an eager config with all compile flags disabled')
    if cfg.get('autocast', cfg.get('autocast_dtype', 'bfloat16')) != 'bfloat16':
        p.error('The active-only backends require BF16 autocast')
    out = (
        args.out
        or Path('.cache/benchmarks')
        / (
            f'{cfg["dataset"]}-active-comparison-{datetime.datetime.now():%Y%m%d-%H%M%S}'
        )
    ).resolve()
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        OMP_NUM_THREADS='2',
        MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2',
        TRITON_DEFAULT_FP_FUSION='1',
        PYTHONUNBUFFERED='1',
        PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH=f'{ROOT}:{ROOT}/src:{ROOT}/dev/pepilica/src',
    )
    env.pop('CUDA_LAUNCH_BLOCKING', None)
    measurement_jobs = jobs(args)
    if args.dry_run:
        print(
            f'GPU {args.gpu}; trains through epoch {args.epochs[-1]} per job; '
            f'measure epochs {args.epochs}; same config/seed for all variants.'
        )
        print('CUDA_VISIBLE_DEVICES=' + str(args.gpu))
        for variant in args.variants:
            for job, options in measurement_jobs:
                print(
                    shlex.join(
                        command(
                            args, config, variant, job, options, out / variant / job
                        )
                    )
                )
        return
    if args.nsys and not shutil.which('nsys'):
        p.error('nsys is unavailable; omit --nsys to keep the torch traces')
    if out.exists():
        p.error(f'Output exists; choose a new directory: {out}')
    import fcntl

    with open(f'/tmp/tabpack-gpu-{args.gpu}.lock', 'w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            p.error('Another cooperating benchmark holds the selected GPU')
        first_samples = preflight(args.gpu, args.min_free_mib)
        out.mkdir(parents=True)
        (out / 'config.json').write_text(config_text)
        config = out / 'config.json'
        dump(
            out / 'suite.json',
            {
                **vars(args),
                'variants': args.variants,
                'runner': 'common.run through registration-only active_variants',
                'jobs': [name for name, _ in measurement_jobs],
                'unprofiled_scope': f'{args.epochs[-1]} epochs; not full convergence',
                'output_heads': {v: 'native LinearPack' for v in args.variants},
                'warmup_epochs': args.epochs[0] - 1,
                'gpu_preflight': first_samples,
            },
        )
        (out / 'commit.txt').write_text(
            subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True)
        )
        (out / 'worktree.txt').write_text(
            subprocess.check_output(['git', 'status', '--short'], text=True)
        )
        tracked_sources = sources()
        hashes = {str(f.relative_to(ROOT)): digest(f) for f in tracked_sources}
        dump(out / 'source-sha256.json', hashes)
        with tarfile.open(out / 'source-snapshot.tar.gz', 'w:gz') as tar:
            for f in tracked_sources:
                tar.add(f, arcname=str(f.relative_to(ROOT)))
        sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'dev/pepilica/src')]
        stats, expected_hps = prepare_data(cfg, out)
        steps = (stats['sizes']['train'] + cfg['batch_size'] - 1) // cfg['batch_size']
        if args.trace_steps and args.trace_steps + 1 > steps:
            p.error(f'Trace warmup+steps exceeds {steps} batches per epoch')
        dump(
            out / 'measurement-scope.json',
            {
                'steps_per_epoch': steps,
                'first_layer_input_widths': stats['first_layer_input_widths'],
                'measurement_epochs': args.epochs,
                'single_training_per_job': True,
                'phases': [
                    'gather',
                    'forward',
                    'loss',
                    'backward',
                    'optimizer',
                    'zero_grad',
                    'eval',
                ],
            },
        )
        for index, variant in enumerate(args.variants):
            if index:
                time.sleep(2)
                dump(
                    out / f'preflight-{variant}.json',
                    preflight(args.gpu, args.min_free_mib),
                )
            for job, options in measurement_jobs:
                if any(digest(ROOT / f) != sha for f, sha in hashes.items()):
                    raise RuntimeError(
                        'Source changed during the suite; results are not comparable'
                    )
                if digest(config) != digest(args.config.resolve()):
                    raise RuntimeError('The source config changed during the suite')
                directory = out / variant / job
                directory.mkdir(parents=True)
                cmd = command(args, config, variant, job, options, directory)
                (directory / 'command.txt').write_text(shlex.join(cmd) + '\n')
                job_env = dict(
                    env,
                    TRITON_CACHE_DIR=str(out / variant / 'cache/triton'),
                    TORCHINDUCTOR_CACHE_DIR=str(out / variant / 'cache/inductor'),
                )
                print(
                    f'Running {variant}/{job} on GPU {args.gpu}; '
                    f'log: {directory}/run.log',
                    flush=True,
                )
                with (directory / 'run.log').open('w') as log:
                    result = subprocess.run(
                        cmd,
                        env=job_env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                (directory / 'exit.txt').write_text(str(result.returncode) + '\n')
                if result.returncode:
                    raise RuntimeError(f'Failed: {directory}/run.log')
                hp = json.loads(
                    (directory / 'sampled-hyperparameters.json').read_text()
                )
                if hp != expected_hps:
                    raise RuntimeError(f'Candidate hyperparameters differ: {directory}')
                report = json.loads((directory / 'report.json').read_text())
                expected_uuid = first_samples[0]['uuid'].removeprefix('GPU-')
                if (
                    report['environment']['gpu_uuid'].removeprefix('GPU-')
                    != expected_uuid
                ):
                    raise RuntimeError('Runner used an unexpected GPU')
                backend = json.loads((directory / 'backend.json').read_text())
                if any(b['fallback_calls'] for b in backend['backends']):
                    raise RuntimeError(
                        'Active backend unexpectedly used its native fallback'
                    )
                if job != 'unprofiled':
                    for epoch in args.epochs:
                        record = epoch_record(report, epoch)
                        if not record['completed']:
                            print(
                                f'{variant}/{job}: epoch {epoch} unavailable; '
                                'training ended early.',
                                flush=True,
                            )
                            continue
                        if (
                            job.startswith('timing')
                            and record['train_profile']['phase_ms']['forward']['count']
                            != steps
                        ):
                            raise RuntimeError(
                                'Timing window did not cover the whole epoch'
                            )
                        if (
                            job == 'torch'
                            and not (
                                directory / f'epoch-{epoch}' / 'trace.json'
                            ).is_file()
                        ):
                            raise RuntimeError('Torch trace is missing')
                (directory / 'SUCCESS').touch()
        completed = summarize(out)
        (out / ('SUCCESS' if completed else 'PARTIAL_MEASUREMENTS')).touch()
        print(f'Completed: {out}', flush=True)


if __name__ == '__main__':
    main()
