"""Replay frozen, train-fitted preprocessing without the paper repository."""

import contextlib
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from tabpackage._training import make_shuffled_batch_indices_pack
from tabpack_experimental.benchmark import PHASES, Config, Runner


class RealDataRunner(Runner):
    def __init__(self, config: Config):
        if config.data_config is None:
            raise ValueError('A prepared dataset manifest is required')
        manifest_path = Path(config.data_config).resolve()
        metadata = json.loads(manifest_path.read_text())
        path = manifest_path.parent / metadata['array_file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata['array_sha256']:
            raise ValueError('Prepared dataset checksum mismatch')
        with np.load(path, allow_pickle=False) as arrays:
            self.data = {name: torch.from_numpy(arrays[name].copy()) for name in arrays}
        actual = dataclasses.replace(
            config,
            data_config=None,
            task='binclass',
            n_num_features=metadata['numerical_features'],
            cat_cardinalities=tuple(metadata['categorical_cardinalities']),
        )
        super().__init__(actual)
        self.config = dataclasses.replace(actual, data_config=config.data_config)
        self.data = {name: tensor.to(self.device) for name, tensor in self.data.items()}
        self.dataset_metadata = metadata
        self.batch_generator = torch.Generator(device=self.device).manual_seed(
            config.seed + 3
        )
        self.prepare_epoch(drop_last=True)
        self._bind_batch(self.batches[0])

    @property
    def phases(self):
        return ('batch', *PHASES)

    @property
    def scope(self):
        return (
            'Resident real train batches; includes gathers, '
            'excludes preprocessing and evaluation.'
        )

    def prepare_epoch(self, *, drop_last):
        self.batches = list(
            make_shuffled_batch_indices_pack(
                pack_size=self.config.pack_size,
                total_size=len(self.data['train_y']),
                batch_size=self.config.batch_size,
                generator=self.batch_generator,
                device=self.device,
            )
        )
        if drop_last:
            self.batches = [
                b for b in self.batches if b.shape[1] == self.config.batch_size
            ]
        if not self.batches:
            raise ValueError('Dataset is smaller than the benchmark batch size')
        self.batch_cursor = 0

    def _bind_batch(self, indices):
        self.x_num = (
            self.data['train_x_num'][indices] if 'train_x_num' in self.data else None
        )
        self.x_cat = (
            self.data['train_x_cat'][indices] if 'train_x_cat' in self.data else None
        )
        self.target = self.data['train_y'][indices].float()

    def step(self, scope=None):
        with scope('batch') if scope else contextlib.nullcontext():
            self._bind_batch(self.batches[self.batch_cursor % len(self.batches)])
            self.batch_cursor += 1
        super().step(scope)

    def snapshot(self):
        state = super().snapshot()
        state['batch_cursor'] = self.batch_cursor
        state['batch_rng'] = self.batch_generator.get_state().clone()
        return state

    def restore(self, snapshot):
        super().restore(snapshot)
        self.batch_cursor = snapshot['batch_cursor']
        self.batch_generator.set_state(snapshot['batch_rng'])
