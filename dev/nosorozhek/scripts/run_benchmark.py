"""Benchmark custom TabPack implementations through the shared common harness."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

import common.run
from dev.nosorozhek.custom.torch_gram_muon import TorchGramMuonAdamWPack
from dev.nosorozhek.custom.triton_linear_coo import (
    TritonCOOLinearPack,
    TritonCOOLinearPackHilbert,
    TritonCOOLinearPackZ,
)
from dev.nosorozhek.custom.triton_linear_pack import TritonGroupedLinearPack
from dev.nosorozhek.custom.triton_muon import TritonMuonAdamWPack

# CONFIG_PATH = Path('dev/nosorozhek/configs/diverse_california.json')
CONFIG_PATH = Path('common/configs/maps_routing.json')
OUTPUT_DIR: Path

WARMUPS = 1

class LinearCases(tuple, Enum):
    Original = ('', None)
    COO = ('coo', {'type': TritonCOOLinearPack})
    COOZ = ('coo-z', {'type': TritonCOOLinearPackZ})
    COOHilbert = ('coo-hilbert', {'type': TritonCOOLinearPackHilbert})
    Triton = (
        'linear',
        {
            'block_m': 64,
            'block_n': 64,
            'block_k': 64,
            'num_warps': 4,
            'num_stages': 2,
        },
    )


class MuonCases(tuple, Enum):
    Original = ('', None)
    Triton = (
        'muon',
        {
            'block_m': 64,
            'block_n': 64,
            'block_k': 64,
            'num_warps': 4,
            'num_stages': 2,
        },
    )
    Gram = (
        'gram-muon',
        {
            'block_m': 64,
            'block_n': 64,
            'block_k': 64,
            'num_warps': 4,
            'num_stages': 2,
            'newton_schulz_algorithm': 'gram',
        },
    )
    TorchGram = ('torch-gram-muon', {'type': TorchGramMuonAdamWPack})


class SkipUpdates(Enum):
    Skip = True
    NoSkip = False

    def __bool__(self) -> bool:
        return self.value


Case = tuple[str, tuple[dict[str, Any] | None, dict[str, Any] | None, SkipUpdates]]
EXTRA_CASES: list[Case] = []


REPEATS = 10
COMPILE_MODEL = True
CASE_CONFIGS = [
    (LinearCases.Original, MuonCases.Original, SkipUpdates.NoSkip),
    (LinearCases.COO, MuonCases.Original, SkipUpdates.NoSkip),
    (LinearCases.COOZ, MuonCases.Original, SkipUpdates.NoSkip),
    (LinearCases.COOHilbert, MuonCases.Original, SkipUpdates.NoSkip),
    (LinearCases.Triton, MuonCases.Triton, SkipUpdates.NoSkip),
    (LinearCases.Triton, MuonCases.TorchGram, SkipUpdates.NoSkip),
    (LinearCases.Triton, MuonCases.Gram, SkipUpdates.NoSkip),
]

BENCHMARKS: list[Case] = [
    (
        '-'.join(
            filter(
                None,
                (
                    linear_name,
                    muon_name,
                    'skip' if skip_zero_updates else 'noskip',
                ),
            )
        ),
        (linear, muon, skip_zero_updates),
    )
    for (linear_name, linear), (muon_name, muon), skip_zero_updates in CASE_CONFIGS
] + EXTRA_CASES


def _install_case(
    linear: dict[str, Any] | None,
    muon: dict[str, Any] | None,
    skip_zero_updates: SkipUpdates,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def variant(kwargs: dict[str, Any]) -> dict[str, Any]:
        if linear is not None:
            linear_options = dict(linear)
            linear_type = linear_options.pop('type', TritonGroupedLinearPack)
            for key, value in linear_options.items():
                setattr(linear_type, key, value)
            common.run.tabpackage.nn.LinearPack = linear_type

        optimizer = {
            **kwargs['optimizer'],
            'muon_skip_zero_updates': bool(skip_zero_updates),
        }
        if muon is not None:
            muon_options = dict(muon)
            optimizer['type'] = muon_options.pop('type', TritonMuonAdamWPack)
            for key in ('block_m', 'block_n', 'block_k', 'num_warps', 'num_stages'):
                if key in muon_options:
                    setattr(optimizer['type'], key, muon_options.pop(key))
            optimizer.update(muon_options)
        return {**kwargs, 'optimizer': optimizer}

    return variant


def _case_variant(name: str) -> str:
    return f'case_{name.replace("-", "_")}'


def _checkpoint_path(name: str, trace: bool = False) -> Path:
    directory = OUTPUT_DIR / 'benchmark' / 'checkpoints'
    directory /= 'traces' if trace else 'full-training'
    return directory / f'{name}.csv'


def _write_checkpoint(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.csv')
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)


def _read_checkpoint(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict('records')


def _run_case(
    name: str,
    round_number: int,
    trace: bool,
    config_path: Path,
    output_root: Path,
    checkpoint: Path,
    force: bool,
) -> None:
    linear, muon, skip_zero_updates = dict(BENCHMARKS)[name]
    variant = _case_variant(name)
    common.run.VARIANTS[variant] = _install_case(linear, muon, skip_zero_updates)

    rows = [] if force else _read_checkpoint(checkpoint)
    completed = {(int(row['round']), int(row['rep'])) for row in rows}
    config = json.loads(config_path.read_text())
    config.update(
        dataset=config.get('dataset', 'california-sklearn'),
        compile_model=COMPILE_MODEL,
        max_epochs=1 if trace else config.get('max_epochs'),
        profile_epoch=1,
        name=f'{config.get("name", config_path.stem)}-{name}',
    )
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.json', prefix='tabpack-common-', delete=False
    ) as config_file:
        json.dump(config, config_file)
        worker_config = Path(config_file.name)

    try:
        repetitions = 1 if trace else (REPEATS + 2 - round_number) // 2
        for rep in range(1, repetitions + 1):
            if (round_number, rep) in completed:
                print(f'{name} round {round_number} rep {rep}: checkpointed')
                continue

            output = output_root / name / f'round-{round_number}-rep-{rep}'
            args = [
                str(worker_config),
                '--profile',
                'torch' if trace else 'none',
                '--variant',
                variant,
                '--out',
                str(output),
            ]
            if trace:
                args.extend(['--steps', str(WARMUPS), '--warmup', '1'])
            report = common.run.main(args)
            row = {
                'module': name,
                'round': round_number,
                'rep': rep,
                'seconds': report['train_wall_seconds'],
                'rmse': (
                    report.get('ensemble_metrics', {})
                    .get('test', report.get('ensemble_metrics', {}).get('val', {}))
                    .get('rmse')
                ),
                'output': str(output),
            }
            rows = [
                existing
                for existing in rows
                if (int(existing['round']), int(existing['rep'])) != (round_number, rep)
            ]
            rows.append(row)
            _write_checkpoint(checkpoint, rows)
            print(f'{name} round {round_number} rep {rep}: checkpointed', flush=True)
    finally:
        worker_config.unlink()


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    summary = (
        table.groupby('module')
        .seconds.agg(
            runs='size',
            median_s='median',
            mean_s='mean',
            sample_std_s='std',
            sem_s='sem',
        )
        .rename(columns={'sample_std_s': 'std_s'})
    )
    table = table.sort_values('seconds').copy()
    groups = table.groupby('module')
    count = groups.seconds.transform('size')
    trim = (count + 19) // 20
    position = groups.cumcount()
    table['seconds'] = table.seconds.where(
        (position >= trim) & (position < count - trim)
    )
    trimmed = (
        table.groupby('module')
        .seconds.agg(mean_s='mean', sample_std_s='std', sem_s='sem')
        .rename(
            columns={
                'mean_s': 'mean_s_90%',
                'sample_std_s': 'std_s_90%',
                'sem_s': 'sem_s_90%',
            }
        )
    )
    return summary.join(trimmed)


def _aggregate_checkpoints() -> pd.DataFrame:
    tables = [
        pd.read_csv(path)
        for path in sorted(
            (OUTPUT_DIR / 'benchmark' / 'checkpoints' / 'full-training').glob('*.csv')
        )
        if not path.name.endswith('.tmp.csv') and path.stat().st_size > 0
    ]
    table = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    if table.empty:
        return table
    table = table.sort_values(['module', 'round', 'rep'])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / 'benchmark.csv'
    summary_path = OUTPUT_DIR / 'benchmark_summary.csv'
    table.to_csv(output_path, index=False)
    summary = summarize(table.dropna(subset=['seconds']))
    summary.to_csv(summary_path)
    print(summary.to_string())
    print(f'Wrote {output_path}')
    print(f'Wrote {summary_path}')
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output-dir', type=Path, required=True, help='Benchmark output directory.'
    )
    parser.add_argument(
        '--trace', action='store_true', help='Capture common.run torch traces.'
    )
    parser.add_argument(
        '--case',
        action='append',
        help=(
            'Run only this configuration. May be repeated. '
            'Selected checkpoints are replaced.'
        ),
    )
    parser.add_argument(
        '--aggregate-only',
        action='store_true',
        help='Rebuild the aggregate CSVs from per-case checkpoints.',
    )
    args = parser.parse_args()
    global OUTPUT_DIR
    OUTPUT_DIR = args.output_dir

    names = [name for name, _ in BENCHMARKS]
    if len(set(names)) != len(names):
        parser.error('Benchmark case names must be unique')
    selected = names if args.case is None else args.case
    unknown = sorted(set(selected) - set(names))
    if unknown:
        parser.error(f'Unknown case(s): {", ".join(unknown)}')

    if args.aggregate_only:
        _aggregate_checkpoints()
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in selected:
        if args.case is not None:
            _checkpoint_path(name, args.trace).unlink(missing_ok=True)

    output_root = OUTPUT_DIR / 'benchmark' / 'common'
    output_root /= 'traces' if args.trace else 'full-training'
    rounds = (1,) if args.trace else (1, 2)
    try:
        for round_number in rounds:
            order = selected if round_number == 1 else list(reversed(selected))
            for name in order:
                worker = (
                    'from pathlib import Path\n'
                    'from dev.nosorozhek.scripts.run_benchmark import _run_case\n'
                    f'_run_case({name!r}, {round_number}, {args.trace!r}, '
                    f'Path({str(CONFIG_PATH)!r}), '
                    f'Path({str(output_root.resolve())!r}), '
                    f'Path({str(_checkpoint_path(name, args.trace).resolve())!r}), '
                    f'{args.case is not None and round_number == 1!r})'
                )
                subprocess.run([sys.executable, '-u', '-c', worker], check=True)
    except subprocess.CalledProcessError:
        _aggregate_checkpoints()
        raise

    if not args.trace:
        _aggregate_checkpoints()


if __name__ == '__main__':
    main()
