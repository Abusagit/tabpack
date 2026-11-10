"""Run the California experiment with CUTLASS grouped Muon GEMMs."""

import runpy
import sys
from pathlib import Path

from experiments.cutlass_grouped.backend import CutlassGroupedMuon
from tabpack_experimental import grouped_fixed_optimizer as grouped

backend = CutlassGroupedMuon()
original_init = grouped.GroupedFixedMuonAdamW.__init__


def initialize(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    backend.register(self)


grouped.GroupedFixedMuonAdamW.__init__ = initialize
grouped._mm = backend
sys.argv[0] = str(Path(__file__).with_name('run_diverse_california.py'))
try:
    runpy.run_module('dev.pepilica.scripts.run_diverse_california', run_name='__main__')
finally:
    print(
        f'CUTLASS launches={backend.launches} plan_builds={backend.plan_builds}',
        flush=True,
    )
