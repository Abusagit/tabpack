"""Benchmark packed TabPack modules using parameter grids."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, ExitStack, contextmanager
from functools import partial
from itertools import product
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import torch

import tabpackage
from dev.nosorozhek.custom.triton_linear_pack import TritonGroupedLinearPack
from dev.nosorozhek.custom.triton_muon import TritonMuonAdamWPack
from dev.nosorozhek.scripts import profile_tabpack
from tabpackage import nn

CONFIG_PATH = Path('dev/nosorozhek/configs/diverse_california.json')
OUTPUT_PATH = Path('dev/nosorozhek/benchmarking/benchmark.csv')
SUMMARY_PATH = Path('dev/nosorozhek/benchmarking/benchmark_summary.csv')

BATCH_SIZE = 2048
WARMUPS = 3

# Repetitions per round. The second round runs the cases in reverse order.
# Therefore, the total number of reps will be 2 * (REPEATS + WARMUPS) * len(BENCHMARKS).
# If `--trace` flag is specified this number reduces to (WARMUPS + 1) * len(BENCHMARKS).
REPEATS = 25

LINEAR_CASES: dict[str, dict[str, int] | None] = {
    '': None,
    'linear': {
        'block_m': 64,
        'block_n': 64,
        'block_k': 64,
        'num_warps': 4,
        'num_stages': 2,
    },
}
MUON_CASES: dict[str, dict[str, int] | None] = {
    '': None,
    'muon': {
        'block_m': 64,
        'block_n': 64,
        'block_k': 64,
        'num_warps': 4,
        'num_stages': 2,
    },
}
SKIP_ZERO_UPDATES = (False, True)

Setup = Callable[[], AbstractContextManager[None]]
Case = tuple[str, Setup]
EXTRA_CASES: list[Case] = []


@contextmanager
def _setup(
    linear: dict[str, int] | None,
    muon: dict[str, int] | None,
    skip_zero_updates: bool,
) -> Generator[None]:
    with ExitStack() as stack:
        if linear is not None:
            if linear:
                stack.enter_context(patch.multiple(TritonGroupedLinearPack, **linear))
            stack.enter_context(patch.object(nn, 'LinearPack', TritonGroupedLinearPack))
        if muon is not None:
            if muon:
                stack.enter_context(patch.multiple(TritonMuonAdamWPack, **muon))
            train = tabpackage.train

            def train_with_muon(**kwargs: Any) -> tabpackage.TrainedTabPack:
                kwargs['optimizer'] = {
                    **kwargs['optimizer'],
                    'type': TritonMuonAdamWPack,
                    'muon_skip_zero_updates': skip_zero_updates,
                }
                return train(**kwargs)

            stack.enter_context(patch.object(tabpackage, 'train', train_with_muon))
        yield


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
        partial(_setup, linear, muon, skip_zero_updates),
    )
    for (linear_name, linear), (muon_name, muon), skip_zero_updates in product(
        LINEAR_CASES.items(), MUON_CASES.items(), SKIP_ZERO_UPDATES
    )
] + EXTRA_CASES


def _train_once(setup: Setup, kwargs: dict[str, Any]) -> tuple[float, float]:
    with setup():
        torch.cuda.synchronize()
        start = time.perf_counter()
        trained = tabpackage.train(**kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    score = trained.load_ensemble_metrics('greedy')['test']['rmse'].item()
    return elapsed, score


def _run_case(name: str, result: Path, trace: bool) -> None:
    setup = dict(BENCHMARKS)[name]
    torch._dynamo.config.recompile_limit = 64
    config = json.loads(CONFIG_PATH.read_text())
    config.update(batch_size=BATCH_SIZE, max_epochs=1, compile_model=True)
    _, kwargs = profile_tabpack._load_experiment(config, torch.device('cuda'))
    kwargs['print_progress'] = False

    for _ in range(WARMUPS):
        with setup():
            tabpackage.train(**kwargs)
    torch.cuda.synchronize()
    if trace:
        output = OUTPUT_PATH.with_suffix('')
        output.mkdir(parents=True, exist_ok=True)
        with setup():
            profile_tabpack._capture_train(kwargs, output, f'{config["name"]}-{name}')
        return

    rows = []
    for rep in range(1, REPEATS + 1):
        seconds, score = _train_once(setup, kwargs)
        rows.append({'module': name, 'rep': rep, 'seconds': seconds, 'rmse': score})
        print(f'{name} rep {rep}: {seconds:.3f}s RMSE={score:.6f}', flush=True)
    result.write_text(json.dumps(rows))


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    summary = table.groupby('module').seconds.agg(
        runs='size', median_s='median', mean_s='mean', sample_std_s='std', sem_s='sem'
    )
    if 'original' in summary.index:
        summary['relative_to_original'] = (
            summary['median_s'] / summary.loc['original', 'median_s']
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--trace', action='store_true', help='Capture traces without benchmarking.'
    )
    args = parser.parse_args()

    names = [name for name, _ in BENCHMARKS]
    if len(set(names)) != len(names):
        parser.error('Benchmark case names must be unique')

    rows = []
    rounds = (names,) if args.trace else (names, names[::-1])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='tabpack-grid-') as temp:
        result = Path(temp) / 'result.json'
        for round_number, order in enumerate(rounds, 1):
            for name in order:
                worker = (
                    'from pathlib import Path\n'
                    'from dev.nosorozhek.scripts.run_benchmark import _run_case\n'
                    f'_run_case({name!r}, Path({str(result)!r}), {args.trace!r})'
                )
                subprocess.run([sys.executable, '-u', '-c', worker], check=True)
                if not args.trace:
                    rows.extend(
                        {**row, 'round': round_number}
                        for row in json.loads(result.read_text())
                    )
                    pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)
    if args.trace:
        return

    table = pd.DataFrame(rows)
    summary = summarize(table)
    summary.to_csv(SUMMARY_PATH)
    print(summary.to_string())
    print(f'Wrote {OUTPUT_PATH}')
    print(f'Wrote {SUMMARY_PATH}')


if __name__ == '__main__':
    main()
