"""Experimental 64x64 tiles for all fixed-slot and grouped optimizer GEMMs."""

import torch

import dev.pepilica.scripts.prototype_fixed_slots as fixed
import tabpack_experimental.grouped_fixed_optimizer as grouped


class Kernel64:
    def __init__(self, kernel):
        self.kernel = kernel

    def __getitem__(self, grid):
        def call(*args, **kwargs):
            assert args[-3:-1] == (32, 32)
            return self.kernel[grid](*args[:-3], 64, 64, args[-1], **kwargs)

        return call


def install():
    original_init = fixed.FixedSlotPlan.__init__
    original_tasks = grouped.GroupedFixedMuonAdamW._tasks

    def init(self, widths, batch, active, *, device='cuda'):
        original_init(self, widths, batch, active, device=device)
        forward = [
            (i, m, n)
            for i in self.active
            for m in range((self.batch + 63) // 64)
            for n in range((self.widths[i] + 63) // 64)
        ]
        forward.sort(key=lambda item: -self.widths[item[0]])
        weights = [
            (i, m, n)
            for i in self.active
            for m in range((self.widths[i] + 63) // 64)
            for n in range((self.widths[i] + 63) // 64)
        ]
        self.forward_tasks = torch.tensor(
            forward, device=device, dtype=torch.int32
        ).reshape(-1, 3)
        self.weight_tasks = torch.tensor(
            weights, device=device, dtype=torch.int32
        ).reshape(-1, 3)

    def tasks(self, active):
        active = tuple(active)
        result = original_tasks(self, active)
        if not hasattr(self, '_tile64_cached'):
            self._tile64_cached = set()
        if active not in self._tile64_cached:
            ids, chunks, _ = result
            tiles = [
                (i, m, n)
                for i in active
                for m in range((self.widths[i] + 63) // 64)
                for n in range((self.widths[i] + 63) // 64)
            ]
            self.cache[active] = (
                ids,
                chunks,
                torch.tensor(
                    tiles, device=self.weight.device, dtype=torch.int32
                ).reshape(-1, 3),
            )
            self._tile64_cached.add(active)
        return self.cache[active]

    fixed.FixedSlotPlan.__init__ = init
    for name in ('fixed_slot_mm', 'fixed_slot_dx', 'fixed_slot_dw'):
        setattr(fixed, name, Kernel64(getattr(fixed, name)))
    grouped.GroupedFixedMuonAdamW._tasks = tasks
    grouped._mm = Kernel64(grouped._mm)
