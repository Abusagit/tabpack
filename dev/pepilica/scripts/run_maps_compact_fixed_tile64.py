"""Run complete Maps-routing training with unfused compact fixed slots."""
import json
import os
import runpy
import sys
import types
from pathlib import Path

from dev.pepilica.scripts.compact_fixed_tile64 import install

install()

import dev.pepilica.scripts.fixed_slot_tabpack as integration
from tabpack_experimental.compact_mlp import CompactFixedSlotMLP

OriginalBackend = integration.FixedSlotTabPackBackend


class CompactBenchmarkBackend(OriginalBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        fixed = self.fixed
        cache = {}
        fused = os.environ.get('COMPACT_FIXED_FUSED', '1') == '1'

        def forward(_fixed, x, active):
            batch = x.shape[0]
            if batch not in cache:
                model = CompactFixedSlotMLP(
                    fixed.widths, batch, tuple(max(1, d) for d in fixed.depths),
                    device='meta', dtype=fixed.weights[0].dtype,
                    dropout=fixed.dropout, fused=fused,
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


integration.FixedSlotTabPackBackend = CompactBenchmarkBackend
runpy.run_module('dev.pepilica.scripts.run_maps_routing', run_name='__main__')
