"""Full training with active-only hidden activations, profiling epoch 3."""
import json
import sys
import time
from pathlib import Path

import torch

from dev.pepilica.scripts.compact_fixed_tile64 import install
from dev.pepilica.scripts.active_slot_tabpack_benchmark import ActiveSlotBenchmarkBackend
import dev.pepilica.scripts.run_maps_routing as runner

install()
runner.FixedSlotTabPackBackend = ActiveSlotBenchmarkBackend
BaseProfile = runner.EpochProfile


class MeasuredProfile(BaseProfile):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wall_samples = []
        self.start_wall = None
        self.before_peak = 0
        self.before_reserved = 0

    def begin_forward(self, model, batch):
        super().begin_forward(model, batch)
        if self.profiler is not None and self.start_wall is None:
            torch.cuda.synchronize()
            if not self.wall_samples:
                self.before_peak = torch.cuda.max_memory_allocated()
                self.before_reserved = torch.cuda.max_memory_reserved()
                torch.cuda.reset_peak_memory_stats()
            self.start_wall = time.perf_counter()

    def after_zero_grad(self):
        measuring = self.profiler is not None and self.start_wall is not None
        if measuring:
            torch.cuda.synchronize()
            self.wall_samples.append((time.perf_counter() - self.start_wall) * 1000)
            self.start_wall = None
        super().after_zero_grad()
        if measuring and self.captured:
            data = {'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                    'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                    'peak_before_capture_bytes': self.before_peak,
                    'reserved_before_capture_bytes': self.before_reserved,
                    'wall_step_ms': self.wall_samples}
            (self.output / 'epoch03-memory-and-wall.json').write_text(json.dumps(data,indent=2)+'\n')

runner.EpochProfile = MeasuredProfile
runner.main()
output = Path(sys.argv[sys.argv.index('--output') + 1])
report_path = output / 'report.json'
report = json.loads(report_path.read_text())
profile = json.loads((output / 'epoch03-memory-and-wall.json').read_text())
report['peak_allocated_bytes'] = max(report['peak_allocated_bytes'], profile['peak_before_capture_bytes'])
report['peak_reserved_bytes'] = max(report['peak_reserved_bytes'], profile['reserved_before_capture_bytes'])
report.update(hidden_implementation='ActiveFixedSlotMLP', tiles='64x64',
              integration='Benchmark-only bridge with active-only hidden activations')
report_path.write_text(json.dumps(report,indent=2)+'\n')
