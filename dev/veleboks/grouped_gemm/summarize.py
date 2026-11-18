"""Create a compact, provenance-checked summary without changing old run bundles."""

# Report prose is kept as single JSON strings.
# ruff: noqa: E501

import argparse
import hashlib
import json
import statistics
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--a4000-run', required=True)
parser.add_argument('--turing-run', required=True)
parser.add_argument('--diagnostics-run', required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[3]
runs = root / 'dev/veleboks/artifacts/runs'
output = root / 'dev/veleboks/results/muon-grouped-gemm'
output.mkdir(parents=True, exist_ok=True)


def write_once(path, value):
    serialized = json.dumps(value, indent=2, ensure_ascii=False) + '\n'
    if path.exists() and path.read_text() != serialized:
        raise FileExistsError(f'Refusing to replace different results: {path}')
    path.write_text(serialized)


report = {
    'experiment': 'cuBLAS grouped GEMM for actual-size square hidden Muon updates',
    'backend_option': "optimizer={'type': tabpackage.optim.MuonAdamWPack, 'muon_backend': 'grouped'}",
    'default': 'padded',
    'algorithm': 'Unchanged five-iteration BF16 Newton-Schulz, coefficients 3.4445/-4.7750/2.0315; unchanged normalization, LR, momentum, Nesterov, weight decay and muon_update_scale.',
    'covered_layers': 'Square hidden layers after the first backbone block. First block, rectangular groups and AdamW parameters use the original implementation.',
    'padding': 'Updates must be zero outside each [:width, :width]. Every grouped call checks this and raises on nonzero padding. Nonzero parameter padding is retained and decayed identically to the original optimizer.',
    'memory_and_streams': 'Persistent padded BF16 workspaces and device pointer arrays, actual GEMM sizes and padded leading dimensions, no matrix gather/scatter. Plan rebuilds on width identity/version, active member set, shape, device or stream change. Runtime cache is per parameter; replacement/removal invalidate it. Output aliases a workspace and must be consumed before its next call.',
    'preparation': 'Normalization and the strict padding check still scan padded tensors; pointer metadata is reused. The synchronizing padding check, workspace copies and skip-zero detection are included in the main measurements.',
    'timing_protocol': '8 warmups, 15 interleaved paired samples, CUDA events and wall time; median/p10/p90 and raw samples saved. Each optimizer/train sample starts from identical weights, optimizer state and dropout RNG. Optimizer gradient restoration and state reset are outside timing. Backend preparation is inside optimizer.step. NS samples average 3 calls; optimizer/train samples measure one step. Compilation excluded and recorded separately. No profiler.',
    'training_protocol': 'Native TabPack, 64 sampled diverse models, California Housing random 64/16/20 split, historical sampled width/depth/embedding/dropout/LR distribution, batch 256. FP32 training/parameters, BF16 NS for both. Paired 60-step smoke training; no online ensemble or early stopping. Validation RMSE is per model in original target units, not an ensemble RMSE.',
    'limitations': [
        'Single shared GPU per hardware type, one seed for timing/training and three NS correctness seeds; no inference about converged quality or other packs.',
        'Timing covers forward/loss/backward/optimizer on a preloaded batch; excludes shuffling, CPU data preparation, transfers, validation and ensemble selection.',
        'Historical 86% sum of Muon kernel durations motivated this experiment; it is not directly comparable to these wall/event times.',
        'Experimental source-checkout extension requires compiler, CUDA headers, Ninja and an exported grouped API; unsupported environments raise rather than change precision.',
        'GPU numbers in CUDA enumeration differ from nvidia-smi; match the recorded UUID.',
    ],
    'toolchain': {
        'headers_toolkit': 'CUDA 13.3.73',
        'compiler': 'GCC 15.2.0',
        'torch': '2.7.1+cu126',
        'runtime_cuda': '12.6',
        'cublas': 120604,
        'driver': '610.57.04',
        'python': '3.11.17',
        'ninja': '1.13.2',
    },
    'hardware_results': [],
    'superseded_runs': {
        '20261110T144046Z-43b5d1d8': 'Probe stopped before compilation: Ninja missing; subsequently installed.',
        '20261110T144147Z-32e9424f': 'Initial successful A4000 six-width probe.',
        '20261110T150119Z-4ba29041': 'Correctness passed, benchmark hit OOM: runtime cache keyed by reloaded group identity retained obsolete plans. Fixed to parameter identity, with removal invalidation. Incomplete timings excluded.',
        '20261110T150402Z-93380c63': 'Complete preliminary A4000 series; optimizer timings included external gradient restoration and train/optimizer samples averaged 3 evolving steps. Superseded by strict one-step paired timing.',
        '20261110T150558Z-b39d04bc': 'Successful unchanged-BF16 grouped probe on Quadro RTX 4000.',
        '20261110T150654Z-17d7eab5': 'Initial FP64 reference and breakdown/lifecycle checks; superseded by final diagnostics including setup timing.',
    },
}
for run_id in (args.a4000_run, args.turing_run):
    run = runs / run_id
    assert (run / 'exit-code.txt').read_text().strip() == '0'
    value = json.loads((run / 'output/experiment.json').read_text())
    assert value['complete']
    manifest = json.loads((run / 'manifest.json').read_text())
    assert value['config']['muon_newton_schulz_steps'] == 5
    timings = []
    for measurement in value['timings']:
        assert measurement['timed_steps_per_sample'] == (
            3 if measurement['scope'] == 'newton_schulz' else 1
        )
        timings.append(
            {
                'scope': measurement['scope'],
                'skip_zero_updates': measurement['skip_zero_updates'],
                'cuda_ms': {
                    name: {
                        key: result['cuda_ms'][key] for key in ('median', 'p10', 'p90')
                    }
                    for name, result in measurement['results'].items()
                },
                'speedup': measurement['speedup_cuda_median'],
                'grouped_time_reduction_percent': 100
                * (1 - 1 / measurement['speedup_cuda_median']),
            }
        )
    train = value['training']
    deltas = [
        x - y
        for x, y in zip(
            train['grouped']['validation_rmse'], train['padded']['validation_rmse']
        )
    ]
    checks = value['newton_schulz_checks']
    row = {
        'run_id': run_id,
        'gpu': value['gpu'],
        'uuid': value['uuid'],
        'capability': value['capability'],
        'grouped_bf16_supported': True,
        'cublas': checks[0]['cublas_version'],
        'build_and_load_seconds': value['extension_load_build_seconds'],
        'first_ns_plan_and_call_seconds': [
            x['setup_first_call_seconds'] for x in checks
        ],
        'first_optimizer_step_and_setup_seconds': value['timings'][0][
            'first_step_including_plan_setup_seconds'
        ],
        'peak_allocated_gib': value['peak_allocated_bytes'] / 2**30,
        'covered_layers': value['covered_layers'],
        'timings': timings,
        'max_ns_relative_l2': max(x['all']['relative_l2'] for x in checks),
        'max_ns_absolute_error': max(x['all']['max_abs'] for x in checks),
        'momentum_exact_with_identical_gradients': all(
            step['momentum']['exact']
            for check in value['optimizer_checks']
            for step in check['steps']
        ),
        'five_step_parameter_relative_l2': [
            x['steps'][-1]['parameters']['relative_l2']
            for x in value['optimizer_checks']
        ],
        'five_step_parameter_max_abs': [
            x['steps'][-1]['parameters']['max_abs'] for x in value['optimizer_checks']
        ],
        'nonzero_padding_rejected': all(x['nonzero_padding_rejected'] for x in checks),
        'replacement_removal_stream_address_checks_passed': all(
            x['replacement_and_removal'] for x in value['optimizer_checks']
        )
        and all(x['changed_addresses_and_stream_passed'] for x in checks),
        'training': {
            name: {
                'validation_rmse_median': result['validation_rmse_median'],
                'initial_mean_loss': statistics.mean(
                    result['loss_by_step_and_member'][0]
                ),
                'final_mean_loss': statistics.mean(
                    result['loss_by_step_and_member'][-1]
                ),
                'wall_seconds': result['wall_seconds'],
                'parameters_and_optimizer_finite': result[
                    'parameters_and_optimizer_finite'
                ],
            }
            for name, result in train.items()
        },
        'validation_rmse_mean_delta': statistics.mean(deltas),
        'validation_rmse_max_abs_delta': max(abs(x) for x in deltas),
        'source_hashes': {
            item['path']: item['sha256']
            for item in manifest['files']
            if item['path']
            in (
                'src/tabpackage/optim.py',
                'src/tabpackage/_training.py',
                'src/tabpackage/_muon_grouped.py',
                'dev/veleboks/grouped_gemm/grouped.cpp',
                'dev/veleboks/grouped_gemm/experiment.py',
                'dev/veleboks/grouped_gemm/experiment.json',
            )
        },
    }
    report['hardware_results'].append(row)
    write_once(output / f'{run_id}.json', value)
    if len(report['hardware_results']) > 1:
        previous = json.loads(
            (runs / args.a4000_run / 'output/experiment.json').read_text()
        )
        assert value['sampled_models'] == previous['sampled_models']
        assert value['dataset'] == previous['dataset']
        assert value['config'] == previous['config']
        for path, sha in row['source_hashes'].items():
            if (
                path != 'src/tabpackage/optim.py'
            ):  # Optimizer changed only docstrings between runs.
                assert report['hardware_results'][0]['source_hashes'][path] == sha

diag_run = runs / args.diagnostics_run
assert (diag_run / 'exit-code.txt').read_text().strip() == '0'
diag = json.loads((diag_run / 'output/diagnostics.json').read_text())
write_once(output / f'{args.diagnostics_run}.json', diag)
report['diagnostics'] = {
    'run_id': args.diagnostics_run,
    'metadata_and_workspace_setup_wall_ms': diag[
        'metadata_and_workspace_setup_wall_ms'
    ],
    'breakdown_cuda_ms': {
        name: {key: result[key] for key in ('median', 'p10', 'p90')}
        for name, result in diag['breakdown'].items()
    },
    'metadata_mutation_and_repeated_state_load_passed': diag[
        'metadata_mutation_and_repeated_state_load_passed'
    ],
    'fp64_reference_relative_l2_ranges': {
        key: [
            min(x[key]['relative_l2'] for x in diag['reference_checks']),
            max(x[key]['relative_l2'] for x in diag['reference_checks']),
        ]
        for key in ('padded_vs_fp64', 'grouped_vs_fp64')
    },
    'interpretation': 'A4000 grouped output matches standalone actual-width native NS exactly for widths 32/64/128/256 in these seeds; larger shapes can differ due to GEMM kernel rounding. Both BF16 paths differ by about 1% from the FP64 iteration initialized with identical rounded normalization. Numerical differences are reported, without choosing pass tolerances from observed errors.',
}
report['interpretation'] = (
    'Actual-size grouped GEMM reduces isolated NS work, but full-step gains depend on hardware and zero skipping. '
    'On A4000, the baseline zero-skip optimization already removes much of the wasted work; grouped metadata/guard/copies and the unchanged first block/forward/backward limit the remaining gain. '
    'The breakdown measures preparation cost directly; it does not identify individual cuBLAS kernels because primary measurements intentionally omit profiling. '
    'Use the per-hardware paired one-step table for conclusions; do not mix runs or compare historical profiler times.'
)
report['summary_script_sha256'] = hashlib.sha256(
    Path(__file__).read_bytes()
).hexdigest()
write_once(output / 'report.json', report)
print(output / 'report.json')
