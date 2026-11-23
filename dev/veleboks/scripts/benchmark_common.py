"""Run matched config pairs through upstream common.run; no custom training code."""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONFIGS = ROOT / 'dev/veleboks/configs'
PHASES = ('gather', 'forward', 'loss', 'backward', 'optimizer', 'zero_grad')


def wait_for_gpu(timeout, memory_limit):
    """Check before each subprocess, without creating a parent CUDA context."""
    gpu = os.environ['CUDA_VISIBLE_DEVICES']
    deadline = time.monotonic() + timeout
    stable = 0
    while True:
        status = subprocess.check_output(
            [
                'nvidia-smi',
                '-i',
                gpu,
                '--query-gpu=memory.used,utilization.gpu',
                '--format=csv,noheader,nounits',
            ],
            text=True,
        )
        memory, utilization = map(int, status.strip().split(','))
        stable = stable + 1 if memory < memory_limit and utilization < 5 else 0
        if stable == 3:
            return
        if time.monotonic() > deadline:
            raise RuntimeError('GPU remained busy: ' + status.strip())
        print('Waiting for GPU:', status.strip(), flush=True)
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=('california', 'maps-routing', 'homecredit-default'),
        default=['california', 'maps-routing', 'homecredit-default'],
    )
    parser.add_argument('--data-root', default='~/tabpack-data/tabred-v0.1.2')
    parser.add_argument('--output', type=Path, default=Path('../output'))
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--wait-seconds', type=int, default=600)
    parser.add_argument('--gpu-memory-limit-mib', type=int, default=1200)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for dataset in args.datasets:
        base = json.loads((CONFIGS / f'{dataset}-diverse-embeddings.json').read_text())
        base.pop('use_cublas_muon', None)
        base['data_root'] = os.path.expanduser(args.data_root)
        reports = []
        for index, enabled in enumerate((False, True, True, False)):
            wait_for_gpu(args.wait_seconds, args.gpu_memory_limit_mib)
            case = out / f'{dataset}-{index}-{int(enabled)}'
            config = out / f'{case.name}.json'
            config.write_text(
                json.dumps({**base, 'use_cublas_muon': enabled}, indent=2)
            )
            command = [
                sys.executable,
                '-m',
                'common.run',
                str(config),
                '--profile',
                'timing',
                '--epoch',
                '1',
                '--wait',
                '20',
                '--warmup',
                '3',
                '--steps',
                str(args.steps),
                '--stop-after-profile',
                '--out',
                str(case),
            ]
            print('Running:', command, flush=True)
            activity = out / f'{case.name}-processes.log'
            with (
                activity.open('w') as telemetry,
                (out / f'{case.name}.log').open('w') as log,
            ):
                monitor = subprocess.Popen(
                    [
                        'nvidia-smi',
                        'pmon',
                        '-i',
                        os.environ['CUDA_VISIBLE_DEVICES'],
                        '-s',
                        'u',
                        '-d',
                        '1',
                    ],
                    stdout=telemetry,
                    stderr=subprocess.STDOUT,
                )
                try:
                    completed = subprocess.run(
                        command,
                        cwd=ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                finally:
                    monitor.terminate()
                    monitor.wait()
            if completed.returncode:
                print((out / f'{case.name}.log').read_text()[-5000:], flush=True)
                raise RuntimeError(f'{case.name} exited {completed.returncode}')
            if (
                'memory allocation failed with OOM'
                in (out / f'{case.name}.log').read_text()
            ):
                raise RuntimeError(
                    f'{case.name}: allocator retried after OOM; reject timing'
                )
            report = json.loads((case / 'report.json').read_text())
            own_samples = 0
            for line in activity.read_text().splitlines():
                fields = line.split()
                if len(fields) < 5 or not fields[0].isdigit():
                    continue
                active = any(
                    value.isdigit() and int(value) > 0 for value in fields[3:5]
                )
                if active:
                    assert int(fields[1]) == report['pid'], (
                        'Foreign GPU activity: ' + line
                    )
                    own_samples += 1
            assert own_samples, 'No process activity samples; cannot validate isolation'
            if reports:
                for key in ('environment', 'sampled_hyperparameters', 'data_hashes'):
                    assert reports[0][key] == report[key], (
                        f'{dataset}: mismatched {key}'
                    )
                cfg = {
                    k: v for k, v in report['config'].items() if k != 'use_cublas_muon'
                }
                assert cfg == base, f'{dataset}: mismatched config'
            phases = report['train_profile']['phase_ms']
            assert report['stopped_after_profile']
            assert all(phases[p]['count'] == args.steps for p in PHASES)
            report['phase_sum_ms'] = sum(phases[p]['mean'] for p in PHASES)
            reports.append(report)
            print(case.name, report['phase_sum_ms'], flush=True)
        means = [
            statistics.fmean(reports[i]['phase_sum_ms'] for i in indices)
            for indices in ((0, 3), (1, 2))
        ]
        item = {
            'dataset': dataset,
            'native_ms': means[0],
            'cublas_ms': means[1],
            'speedup': means[0] / means[1],
            'time_reduction_percent': 100 * (1 - means[1] / means[0]),
            'optimizer_ms': [
                statistics.fmean(
                    reports[i]['train_profile']['phase_ms']['optimizer']['mean']
                    for i in indices
                )
                for indices in ((0, 3), (1, 2))
            ],
            'reports': [
                str(out / f'{dataset}-{i}-{int(i in (1, 2))}' / 'report.json')
                for i in range(4)
            ],
        }
        results.append(item)
        (out / 'comparison.json').write_text(
            json.dumps(
                {
                    'protocol': 'common.run; ABBA; 20 wait + 3 warmup + active steps; '
                    'sum of six nonoverlapping CUDA-event phase means, not '
                    'wall latency or training to convergence',
                    'cases': results,
                },
                indent=2,
            )
        )
        print(item, flush=True)


if __name__ == '__main__':
    main()
