"""Generate the synthetic classification dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler


def generate(
    output_path: Path, classification_kwargs: dict[str, Any], split_sizes: list[int]
) -> None:
    """Generate the synthetic classification dataset."""
    if len(split_sizes) != 3 or sum(split_sizes) != classification_kwargs['n_samples']:
        raise ValueError('split_sizes must contain three sizes summing to n_samples')
    x, y = make_classification(**classification_kwargs)
    x = x.astype(np.float32)
    x = StandardScaler().fit(x[: split_sizes[0]]).transform(x)

    arrays = {}
    start = 0
    for part, size in zip(('train', 'val', 'test'), split_sizes):
        rows = slice(start, start + size)
        arrays[f'{part}.x_num'] = x[rows]
        arrays[f'{part}.y'] = y[rows]
        if not np.array_equal(
            np.unique(y[rows]), np.arange(classification_kwargs['n_classes'])
        ):
            raise ValueError(f'{part} must contain every class')
        start += size
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix('.tmp.npz')
    np.savez(temporary_path, **arrays)
    temporary_path.replace(output_path)
