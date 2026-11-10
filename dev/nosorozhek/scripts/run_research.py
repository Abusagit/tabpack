"""Run trace-only batch sweeps or analyze traces already on disk."""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
from perfetto.trace_processor import TraceProcessor
from tqdm import tqdm

from dev.nosorozhek.scripts import profile_tabpack

DATASETS = [
    'diverse_california',
    'diverse_maps_routing',
]
BATCH_SIZES = [256, 512, 1024, 2048]
COMPILE_MODELS = [False, True]
RESULTS_DIR = Path('dev/nosorozhek/profiling')

_GPU_EVENT_SQL = """
    name GLOB 'ampere_*'
    OR name LIKE 'void at::native%'
    OR name LIKE 'Memcpy %'
    OR name GLOB 'triton_*'
"""

_ANNOTATION_SQL = """
    name LIKE 'inference.%'
    OR name = 'training'
"""


def _trace_cases() -> list[tuple[str, int, bool]]:
    return [
        (dataset, batch, compiled)
        for dataset in DATASETS
        for batch in BATCH_SIZES
        for compiled in COMPILE_MODELS
    ]


def _case_path(dataset: str, batch: int, compiled: bool) -> Path:
    mode = 'compiled' if compiled else 'eager'
    return RESULTS_DIR / dataset / f'train{batch}-predict{batch}-{mode}'


def _prepare_outputs(cases: Iterable[tuple[str, int, bool]]) -> None:
    paths = [_case_path(*case) for case in cases]
    existing = [RESULTS_DIR / 'trace_ratios.csv', *paths]
    existing = [path for path in existing if path.exists()]
    if existing:
        print('This run will replace:\n' + '\n'.join(map(str, existing)))
        try:
            answer = input('Replace these profiling results? [y/N] ')
        except EOFError:
            answer = ''
        if answer.strip().lower() not in ('y', 'yes'):
            raise SystemExit('Cancelled.')
        for path in existing:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _run_cases() -> None:
    cases = _trace_cases()
    _prepare_outputs(cases)
    for dataset, batch, compiled in tqdm(cases, desc='profiling', unit='case'):
        config = json.loads(Path(f'dev/nosorozhek/configs/{dataset}.json').read_text())
        config.update(
            batch_size=batch,
            inference_batch_size=batch,
            max_epochs=1,
            compile_model=compiled,
        )
        output = _case_path(dataset, batch, compiled)
        output.mkdir(parents=True)
        with (
            (output / 'run.log').open('w', buffering=1) as log,
            contextlib.redirect_stdout(log),
            contextlib.redirect_stderr(log),
        ):
            profile_tabpack.capture_traces(config, output)
    _analyze_traces()


def _merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted((s, e) for s, e in intervals if e > s):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _interval_duration(intervals: Iterable[tuple[int, int]]) -> int:
    return sum(end - start for start, end in _merge_intervals(intervals))


def _overlap_duration(left: list[tuple[int, int]], right: list[tuple[int, int]]) -> int:
    left, right = _merge_intervals(left), _merge_intervals(right)
    i = j = total = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        total += max(0, end - start)
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def _dataframe_intervals(frame: pd.DataFrame) -> list[tuple[int, int]]:
    return [
        (int(row.ts), int(row.ts + row.dur))
        for row in frame.itertuples()
        if row.dur > 0
    ]


def _trace_metadata(path: Path) -> dict[str, Any] | None:
    match = re.fullmatch(r'train(\d+)-predict(\d+)-(eager|compiled)', path.parent.name)
    if match is None:
        return None
    return {
        'dataset': path.parent.parent.name,
        'train_batch_size': int(match.group(1)),
        'predict_batch_size': int(match.group(2)),
        'compiled': match.group(3) == 'compiled',
        'mode': 'training' if 'training-trace' in path.name else 'inference',
    }


def _analyze_trace(path: Path) -> list[dict[str, Any]]:
    metadata = _trace_metadata(path)
    if metadata is None:
        return []
    with TraceProcessor(trace=str(path)) as processor:
        bounds = (
            processor.query('select min(ts) start, max(ts + dur) end from slice')
            .as_pandas_dataframe()
            .iloc[0]
        )
        recording_ns = max(0, int(bounds.end - bounds.start))
        candidates = processor.query(
            f"""select track_id, sum(dur) gpu_ns from slice
                where dur > 0 and ({_GPU_EVENT_SQL})
                group by track_id order by gpu_ns desc"""
        ).as_pandas_dataframe()
        gpu_track_id = None if candidates.empty else int(candidates.iloc[0].track_id)
        gpu_intervals = []
        if gpu_track_id is not None:
            gpu_frame = processor.query(
                f"""select ts, dur from slice where track_id = {gpu_track_id}
                    and dur > 0 and not ({_ANNOTATION_SQL})"""
            ).as_pandas_dataframe()
            gpu_intervals = _dataframe_intervals(gpu_frame)
        gpu_ns = _interval_duration(gpu_intervals)
        annotations = processor.query(
            f"""select name, ts, dur from slice where dur > 0 and ({_ANNOTATION_SQL})"""
        ).as_pandas_dataframe()

    rows = []
    names = sorted(annotations.name.unique()) if not annotations.empty else ['']
    for annotation in names:
        frame = annotations[annotations.name == annotation]
        annotation_intervals = _dataframe_intervals(frame)
        annotation_ns = _interval_duration(annotation_intervals)
        under_annotation_ns = _overlap_duration(gpu_intervals, annotation_intervals)
        rows.append(
            {
                **metadata,
                'batch_size': metadata['train_batch_size']
                if metadata['mode'] == 'training'
                else metadata['predict_batch_size'],
                'annotation': annotation,
                'gpu_stream_track_id': gpu_track_id,
                'recording_duration_ns': recording_ns,
                'gpu_activity_ns': gpu_ns,
                'gpu_activity_ratio': gpu_ns / recording_ns if recording_ns else 0.0,
                'annotation_duration_ns': annotation_ns,
                'gpu_time_under_annotation_ns': under_annotation_ns,
                'gpu_time_under_annotation_ratio': (
                    under_annotation_ns / annotation_ns if annotation_ns else None
                ),
                'trace': str(path),
            }
        )
    return rows


def _analyze_traces() -> pd.DataFrame:
    traces = sorted(RESULTS_DIR.rglob('*-trace.json'))
    rows = [
        row
        for trace in tqdm(traces, desc='analyzing traces', unit='trace')
        for row in _analyze_trace(trace)
    ]
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / 'trace_statistics.csv', index=False)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Only rebuild the CSV from existing traces',
    )
    args = parser.parse_args()
    if args.analyze:
        _analyze_traces()
    else:
        _run_cases()


if __name__ == '__main__':
    main()
