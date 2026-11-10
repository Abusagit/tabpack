"""Prepare Maps Routing using the original TabPack experiment's data recipe.

Run with PYTHONPATH pointing to the original experiment's src directory and its
Python environment; that repository provides lib.data and its dependencies.
"""
import json
from pathlib import Path

import numpy as np
from lib.data import build_dataset


def main():
    output = Path('.cache/maps-routing-paper')
    output.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset(
        'data/tabred/maps-routing',
        extract_bin_from_num=True,
        bin_policy='convert-to-cat',
        cat_policy='ordinal',
        cache=False,
    )
    stats = dataset.try_standardize_labels_()
    for key, parts in dataset.data.items():
        for name, array in parts.items():
            np.save(output / f'{name}_{key}.npy', array, allow_pickle=False)
    (output / 'metadata.json').write_text(
        json.dumps(
            {
                'n_num': dataset.n_num_features,
                'cat_cardinalities': dataset.compute_cat_cardinalities(),
                'target_mean': stats.mean,
                'target_std': stats.std,
            },
            indent=2,
        )
        + '\n'
    )


if __name__ == '__main__':
    main()
