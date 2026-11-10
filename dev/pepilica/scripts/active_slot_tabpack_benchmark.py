"""Benchmark-only active-activation bridge for the native TabPack training loop."""
import types

import torch

from dev.pepilica.scripts.fixed_slot_tabpack import FixedSlotTabPackBackend
from tabpack_experimental.active_fixed_slot import ActiveFixedSlotMLP
from tabpack_experimental.ragged_linear import ActivationTransfer


class ActiveSlotBenchmarkBackend(FixedSlotTabPackBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        fixed = self.fixed
        cache = {}

        def forward(_fixed, x, active):
            batch = x.shape[0]
            if batch not in cache:
                model = ActiveFixedSlotMLP(
                    fixed.widths, batch, fixed.depths, device='meta',
                    dtype=fixed.weights[0].dtype, dropout=fixed.dropout, tile=64,
                )
                for block, weight, bias in zip(model.blocks, fixed.weights, fixed.biases):
                    block.weight, block.bias = weight, bias
                cache[batch] = model
            model = cache[batch]
            model.train(fixed.training)
            return model(x, tuple(active))

        fixed.forward = types.MethodType(forward, fixed)
        self.compact_models = cache

    @torch.inference_mode(False)
    def prepare(self, batch, device):
        key = (tuple(self.ids), batch, device)
        if self.cached_ids != key[0]:
            self.cache.clear()
            self.cached_ids = key[0]
        if key not in self.cache:
            widths = [self.widths[i] for i in self.ids]
            self.cache[key] = ActivationTransfer(
                [(batch, width) for width in widths],
                (len(widths), batch, self.model.backbone.blocks[0].linear.max_out_features),
                device,
            )
            self.rebuilds += 1
        return self.cache[key]
