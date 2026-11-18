"""Compare completed compiled-Muon runs against the original A4000 controls."""

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text())


def save_json(path, value):
    # Existing results remain immutable; repeating the same comparison is safe.
    if path.exists():
        if read_json(path) != value:
            raise FileExistsError(f'Refusing to replace different results: {path}')
    else:
        path.write_text(json.dumps(value, indent=2) + '\n')


def stats(values):
    return {'mean': statistics.mean(values), 'sample_sd': statistics.stdev(values)}


def measurements(report):
    elapsed = report['epoch_eval_elapsed_seconds']
    intervals = [elapsed[i] - elapsed[i - 1] for i in range(1, len(elapsed))]
    return {
        'test_score': report['test_score'],
        'train_wall_seconds': report['train_wall_seconds'],
        'first_validation_elapsed_seconds': elapsed[0],
        # Includes the preceding ensemble update, then training and validation.
        'later_validation_interval_mean_seconds': statistics.mean(intervals),
        'peak_allocated_gib': report['train_peak_allocated_bytes'] / 2**30,
        'inference_eager_mean_ms': statistics.mean(report['inference_seconds']['eager'])
        * 1000,
        'inference_compiled_mean_ms': statistics.mean(
            report['inference_seconds']['compiled']
        )
        * 1000,
        'selected_checkpoints': len(
            set(zip(report['selected_member_ids'], report['selected_member_steps']))
        ),
    }


def main():
    workspace = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maps-run', required=True)
    parser.add_argument('--homecredit-run', required=True)
    parser.add_argument(
        '--output',
        type=Path,
        default=workspace / 'results/tabred-torch214-a4000-compile-muon-ns',
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    baseline_dir = workspace / 'results/tabred-torch214-a4000'
    runs = workspace / 'artifacts/runs'
    result = {
        'comparison_script_sha256': hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        'changed_parameter': {'compile_optimizer_muon_newton_schulz': True},
        'comparison': 'paired by seed against previously completed controls',
        'train_wall_includes_lazy_compilation': True,
        'compiler_cache_cleared': False,
        'later_interval_definition': (
            'Mean of four intervals between validation ends for epochs 1..5; '
            'includes the preceding ensemble update, next training and validation. '
            'This excludes the initial lazy compilation, '
            'not all possible recompilation.'
        ),
        'datasets': {},
    }
    for name, run_id in (
        ('maps-routing', args.maps_run),
        ('homecredit-default', args.homecredit_run),
    ):
        run = runs / run_id
        assert (run / 'exit-code.txt').read_text().strip() == '0', run
        manifest = read_json(run / 'manifest.json')
        environment = read_json(run / 'output/environment.json')
        telemetry = list(csv.DictReader((run / 'output/gpu-telemetry.csv').open()))
        uuid = telemetry[0][' uuid'].strip()
        pairs = []
        for seed in (0, 1, 2):
            old = read_json(baseline_dir / f'{name}-seed-{seed}.json')
            new = read_json(run / 'output' / f'seed-{seed}.json')
            config = new['config'].copy()
            assert config.pop('compile_optimizer_muon_newton_schulz') is True
            assert config == old['config']
            assert new['dataset'] == old['dataset']
            assert new['sampled_models'] == old['sampled_models']
            assert new['seed'] == old['seed'] == seed
            assert new['epochs_completed'] == old['epochs_completed'] == 5
            assert environment['torch'] == old['provenance']['torch']
            assert uuid == old['provenance']['gpu_uuid']
            old_environment = read_json(
                runs / old['provenance']['run_id'] / 'output/environment.json'
            )
            assert environment['packages'] == old_environment['packages']
            assert math.isfinite(new['test_score'])
            for samples in new['inference_seconds'].values():
                assert len(samples) == 50
                assert all(math.isfinite(x) and x > 0 for x in samples)
            relevant_paths = {
                'dev/veleboks/run_tabred_baseline.py',
                f'dev/veleboks/configs/{name}-a4000-compile-muon-ns.json',
            }
            source_sha256 = {
                entry['path']: entry['sha256']
                for entry in manifest['files']
                if entry['path'] in relevant_paths
            }
            assert len(source_sha256) == 2
            new['provenance'] = {
                'run_id': run_id,
                'host': manifest['host'],
                'gpu_index': manifest['gpu'],
                'gpu_uuid': uuid,
                'gpu': environment['gpu'],
                'torch': environment['torch'],
                'cuda': environment['cuda'],
                'commit': manifest['commit'],
                'branch': manifest['branch'],
                'source_sha256': source_sha256,
                'manifest_sha256': hashlib.sha256(
                    (run / 'manifest.json').read_bytes()
                ).hexdigest(),
            }
            save_json(args.output / f'{name}-seed-{seed}.json', new)
            a, b = measurements(old), measurements(new)
            pairs.append(
                {
                    'seed': seed,
                    'baseline': a,
                    'compiled_muon_ns': b,
                    'test_score_delta': b['test_score'] - a['test_score'],
                    'ensemble_checkpoints_identical': (
                        new['selected_member_ids'] == old['selected_member_ids']
                        and new['selected_member_steps'] == old['selected_member_steps']
                    ),
                    'train_wall_speedup': a['train_wall_seconds']
                    / b['train_wall_seconds'],
                }
            )
        keys = pairs[0]['baseline'].keys()
        aggregates = {
            variant: {k: stats([p[variant][k] for p in pairs]) for k in keys}
            for variant in ('baseline', 'compiled_muon_ns')
        }
        result['datasets'][name] = {
            'run_id': run_id,
            'baseline_run_id': old['provenance']['run_id'],
            'gpu_uuid': uuid,
            'test_metric': new['test_metric'],
            'aggregates': aggregates,
            'train_wall_speedup': aggregates['baseline']['train_wall_seconds']['mean']
            / aggregates['compiled_muon_ns']['train_wall_seconds']['mean'],
            'later_interval_speedup': aggregates['baseline'][
                'later_validation_interval_mean_seconds'
            ]['mean']
            / aggregates['compiled_muon_ns']['later_validation_interval_mean_seconds'][
                'mean'
            ],
            'test_score_delta': stats([p['test_score_delta'] for p in pairs]),
            'paired_results': pairs,
        }
    save_json(args.output / 'comparison.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
