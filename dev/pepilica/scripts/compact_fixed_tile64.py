"""Experimental 64x64 compact fixed-slot and grouped optimizer tiles."""
import torch

import tabpack_experimental.compact_mlp as compact
from dev.pepilica.scripts.tile64_experiment import Kernel64, install as install_optimizer_tiles


def install():
    install_optimizer_tiles()
    original_init = compact.FixedSlotPlan.__init__

    def init(self, widths, batch, active, *, device='cuda'):
        original_init(self, widths, batch, active, device=device)
        forward = [
            (i, m, n)
            for i in self.active
            for m in range((self.batch + 63) // 64)
            for n in range((self.widths[i] + 63) // 64)
        ]
        forward.sort(key=lambda task: -self.widths[task[0]])
        weight = [
            (i, m, n)
            for i in self.active
            for m in range((self.widths[i] + 63) // 64)
            for n in range((self.widths[i] + 63) // 64)
        ]
        self.forward_tasks = torch.tensor(forward, device=device, dtype=torch.int32).reshape(-1, 3)
        self.weight_tasks = torch.tensor(weight, device=device, dtype=torch.int32).reshape(-1, 3)

    compact.FixedSlotPlan.__init__ = init
    for name in ('fixed_slot_mm', 'fixed_slot_fused_mm', 'fixed_slot_dx', 'fixed_slot_dw'):
        setattr(compact, name, Kernel64(getattr(compact, name)))
