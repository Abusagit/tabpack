"""Measure full-loop CUDA memory by phase at epoch 3 and after pack shrinkage."""
import contextlib
import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import torch

from dev.pepilica.scripts.compact_fixed_tile64 import install
from dev.pepilica.scripts.fixed_slot_tabpack import FixedSlotTabPackBackend
from dev.pepilica.scripts.active_slot_tabpack_benchmark import ActiveSlotBenchmarkBackend
from tabpack_experimental.compact_mlp import CompactFixedSlotMLP
import tabpackage._training as training
import dev.pepilica.scripts.run_maps_routing as runner


class FullColumnBackend(FixedSlotTabPackBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        fixed, cache = self.fixed, {}

        def forward(_fixed, x, active):
            batch = x.shape[0]
            if batch not in cache:
                model = CompactFixedSlotMLP(
                    fixed.widths, batch, tuple(max(1, d) for d in fixed.depths),
                    device='meta', dtype=fixed.weights[0].dtype,
                    dropout=fixed.dropout, fused=False,
                )
                for block, weight, bias in zip(model.blocks, fixed.weights, fixed.biases):
                    block.weight, block.bias = weight, bias
                    block.keep_probabilities = fixed.keep
                cache[batch] = model
            model = cache[batch]
            model.train(fixed.training)
            return model(x, tuple(i for i in active if fixed.depths[i] > 0))

        fixed.forward = types.MethodType(forward, fixed)
        self.compact_models = cache


class MeasurementsDone(Exception):
    pass


class MemoryProfile:
    def __init__(self, output, epoch, steps, hyperparameters, enabled):
        self.output = output
        self.steps_per_epoch = steps
        self.completed = 0
        self.groups = {'epoch3': [], 'shrunken': []}
        self.native = self.layout == 'vanilla'
        self.bridge = (types.SimpleNamespace(ids=list(range(len(hyperparameters))),
                       widths=tuple(h['model']['d_block'] for h in hyperparameters))
                       if self.native else None)
        self.current = None
        self.region = None
        self.shrink_key = None
        self.shrink_seen = 0
        self.captured = False
        self.start_step = self.steps = 0

    def warm(self):
        pass

    def begin(self, name):
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        self.region = (name, allocated, reserved)

    def end(self):
        torch.cuda.synchronize()
        name, start, reserved = self.region
        self.current['phases'][name] = {
            'start_allocated': start,
            'end_allocated': torch.cuda.memory_allocated(),
            'peak_allocated': torch.cuda.max_memory_allocated(),
            'start_reserved': reserved,
            'end_reserved': torch.cuda.memory_reserved(),
            'peak_reserved': torch.cuda.max_memory_reserved(),
        }
        self.region = None

    def select(self, model):
        epoch = self.completed // self.steps_per_epoch + 1
        index = self.completed % self.steps_per_epoch
        if epoch == 3 and 2 <= index < 7:
            return 'epoch3', epoch, index
        if epoch > 3 and model.pack_size <= 32 and not self.groups['shrunken']:
            key = (epoch, tuple(self.bridge.ids))
            if key != self.shrink_key:
                self.shrink_key, self.shrink_seen = key, 0
            self.shrink_seen += 1
            if self.shrink_seen >= 3:
                return 'shrunken', epoch, index
        if epoch > 3 and model.pack_size <= 32 and self.groups['shrunken'] and len(self.groups['shrunken']) < 5:
            return 'shrunken', epoch, index
        return None

    @contextlib.contextmanager
    def instrument(self):
        original_forward = training.forward
        original_factory = training.make_optimizer_pack
        original_backward = torch.Tensor.backward
        original_remove = training.pack_remove
        profile = self

        def forward(model, batch, **kwargs):
            selected = (profile.select(model) if model.training and torch.is_grad_enabled() else None)
            if selected:
                group, epoch, index = selected
                ids = tuple(profile.bridge.ids)
                profile.current = {'group': group, 'epoch': epoch, 'batch_index': index,
                                   'active_ids': ids, 'active_count': len(ids),
                                   'active_width': sum(profile.bridge.widths[i] for i in ids),
                                   'total_width': sum(profile.bridge.widths), 'phases': {}}
                profile.begin('forward')
            result = original_forward(model, batch, **kwargs)
            if profile.current is not None:
                profile.end()
                profile.begin('loss')
            return result

        def backward(tensor, *args, **kwargs):
            if profile.current is not None:
                profile.end()
                profile.begin('backward')
            result = original_backward(tensor, *args, **kwargs)
            if profile.current is not None:
                profile.end()
            return result

        def factory(*args, **kwargs):
            optimizer = original_factory(*args, **kwargs)
            if not profile.native:
                profile.bridge = optimizer.step.__self__
            step, zero = optimizer.step, optimizer.zero_grad

            def measured_step(*a, **kw):
                if profile.current is not None:
                    profile.begin('optimizer')
                result = step(*a, **kw)
                if profile.current is not None:
                    profile.end()
                return result

            def measured_zero(*a, **kw):
                if profile.current is not None:
                    profile.begin('zero_grad')
                result = zero(*a, **kw)
                if profile.current is not None:
                    profile.end()
                    group = profile.current['group']
                    profile.groups[group].append(profile.current)
                    profile.current = None
                    if len(profile.groups[group]) == 5:
                        (profile.output / 'phase-memory.json').write_text(json.dumps(profile.groups, indent=2)+'\n')
                        print(f'Captured {group} phase memory.', flush=True)
                    if len(profile.groups['shrunken']) >= 5 and len(profile.groups['epoch3']) >= 5:
                        (profile.output / 'phase-memory.json').write_text(json.dumps(profile.groups, indent=2)+'\n')
                        print('Captured epoch 3 and shrunken pack phase memory.', flush=True)
                        raise MeasurementsDone()
                profile.completed += 1
                return result

            optimizer.step, optimizer.zero_grad = measured_step, measured_zero
            return optimizer

        def remove(*args, pack_idx, **kwargs):
            if profile.native:
                removed = set(pack_idx.tolist())
                profile.bridge.ids[:] = [member for i, member in enumerate(profile.bridge.ids) if i not in removed]
            return original_remove(*args, pack_idx=pack_idx, **kwargs)

        with (patch.object(training, 'pack_remove', remove),
              patch.object(training, 'forward', forward),
              patch.object(training, 'make_optimizer_pack', factory),
              patch.object(torch.Tensor, 'backward', backward)):
            yield


def main():
    parser = __import__('argparse').ArgumentParser(add_help=False)
    parser.add_argument('--layout', choices=('fixed', 'active', 'vanilla'), required=True)
    options, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    install()
    runner.FixedSlotTabPackBackend = ActiveSlotBenchmarkBackend if options.layout == 'active' else FullColumnBackend
    MemoryProfile.layout = options.layout
    runner.EpochProfile = MemoryProfile
    try:
        runner.main()
    except MeasurementsDone:
        pass

if __name__ == '__main__':
    main()
