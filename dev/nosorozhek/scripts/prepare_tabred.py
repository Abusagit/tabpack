"""Convert a TabReD directory and split into the profiler's NPZ format."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def prepare(source: Path, output: Path, split: str = 'default') -> None:
    """Save one TabReD split as train/validation/test arrays."""
    parts = ('train', 'val', 'test')
    rows = {part: np.load(source / 'splits' / split / f'{part}.npy') for part in parts}
    if len(np.unique(np.concatenate(tuple(rows.values())))) != sum(
        map(len, rows.values())
    ):
        raise ValueError('TabReD splits overlap')
    x_num = np.load(source / 'x_num.npy', mmap_mode='r')
    x_cat = np.load(source / 'x_cat.npy', mmap_mode='r')
    y = np.load(source / 'y.npy')
    means = np.nanmean(x_num[rows['train']], axis=0, dtype=np.float64)
    means = np.nan_to_num(means, nan=0).astype(np.float32)
    arrays = {}
    for part in parts:
        features = x_num[rows[part]].copy()
        np.copyto(features, means, where=np.isnan(features))
        target = y[rows[part]]
        arrays[f'{part}.x_num'] = features
        arrays[f'{part}.x_cat'] = x_cat[rows[part]].copy()
        arrays[f'{part}.y'] = target.copy()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp.npz')
    np.savez(temporary, **arrays)
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--split', default='default')
    args = parser.parse_args()
    prepare(args.source, args.output, args.split)


if __name__ == '__main__':
    main()
