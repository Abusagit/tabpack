"""Profile unfused CompactFixedSlotMLP with 64x64 tiles on Maps routing."""
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
runpy.run_module('dev.pepilica.scripts.profile_maps_epoch3', run_name='__main__')

output = Path(sys.argv[sys.argv.index('--output') + 1])
report_path = output / 'report.json'
report = json.loads(report_path.read_text())
report.update(
    hidden_implementation='CompactFixedSlotMLP',
    fused_epilogue=os.environ.get('COMPACT_FIXED_FUSED', '1') == '1',
    tiles='64x64',
    integration='Benchmark-only bridge with native input/head and grouped fixed-slot optimizer',
)
report_path.write_text(json.dumps(report, indent=2) + '\n')
