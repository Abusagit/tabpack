"""Run the fixed-slot benchmark with 64x64 tiles in the forward GEMM only."""

import runpy
import sys
import torch
import dev.pepilica.scripts.prototype_fixed_slots as fixed

original_init = fixed.FixedSlotPlan.__init__
original_kernel = fixed.fixed_slot_mm


def init(self, widths, batch, active, *, device='cuda'):
    original_init(self, widths, batch, active, device=device)
    tasks = [
        (i, m, n)
        for i in self.active
        for m in range((self.batch + 63) // 64)
        for n in range((self.widths[i] + 63) // 64)
    ]
    tasks.sort(key=lambda item: -self.widths[item[0]])
    self.forward_tasks = torch.tensor(tasks, device=device, dtype=torch.int32).reshape(
        -1, 3
    )


class Kernel64:
    def __getitem__(self, grid):
        def call(x, w, b, out, desc, tasks, stride, batch, bm, bn, bk, **kwargs):
            assert bm == bn == 32
            return original_kernel[grid](
                x, w, b, out, desc, tasks, stride, batch, 64, 64, bk, **kwargs
            )

        return call


fixed.FixedSlotPlan.__init__ = init
fixed.fixed_slot_mm = Kernel64()
sys.argv[0] = 'dev/pepilica/scripts/benchmark_maps_epoch3.py'
runpy.run_module('dev.pepilica.scripts.benchmark_maps_epoch3', run_name='__main__')
