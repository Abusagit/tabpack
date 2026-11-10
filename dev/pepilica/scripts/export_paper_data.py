"""Run with the paper checkout's Python/PYTHONPATH to freeze its preprocessing."""

import argparse
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    # This is a one-time migration tool. Runtime benchmarks import only TabPackage.
    load_dataset = importlib.import_module('project.real_data').load_dataset

    dataset, metadata = load_dataset(args.config, args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    array_file = args.output / 'churn.npz'
    np.savez_compressed(
        array_file,
        **{
            f'{part}_{kind}': values
            for kind, parts in dataset.data.items()
            for part, values in parts.items()
        },
    )
    metadata.update(
        array_file=array_file.name,
        array_sha256=hashlib.sha256(array_file.read_bytes()).hexdigest(),
        preprocessing_seed=args.seed,
        preprocessing_status='Frozen paper preprocessing; no fitting in benchmark',
    )
    (args.output / 'churn.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
