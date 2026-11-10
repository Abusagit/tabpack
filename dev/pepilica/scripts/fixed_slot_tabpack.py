"""Experimental native TabPack bridge with stable hidden parameter slots."""

from typing import Any

import torch

if __package__:
    from .fixed_slot_mlp import FixedSlotMLP, FixedSlotMLPOptimizer
else:
    from dev.pepilica.scripts.fixed_slot_mlp import FixedSlotMLP, FixedSlotMLPOptimizer
from tabpack_experimental.compact_backbone import _Pack, _Unpack
from tabpack_experimental.ragged_linear import (
    ActivationTransfer,
    MatrixTransfer,
    _pack_matrix,
)


class StableTransfer(ActivationTransfer):
    def __init__(self, widths, ids, batch, max_width, device):
        super().__init__(
            [(batch, widths[i]) for i in ids], (len(ids), batch, max_width), device
        )
        offsets = [sum(widths[:i]) for i in ids]
        self.row_stride = sum(widths)
        self.compact_shape = (batch, self.row_stride)
        self.descriptors = torch.tensor(
            [(offsets[j], batch, widths[i]) for j, i in enumerate(ids)],
            device=device,
            dtype=torch.int64,
        )

    def pack(self, source, dtype):
        output = torch.zeros(self.compact_shape, dtype=dtype, device=source.device)
        kernel: Any = _pack_matrix
        kernel[(self.tasks.shape[0],)](
            source,
            output,
            self.descriptors,
            self.tasks,
            *source.stride(),
            self.row_stride,
            1024,
        )
        return output


class FixedSlotTabPackBackend:
    def __init__(self, model, optimizer, ids, hyperparameters):
        self.model, self.optimizer, self.ids = model, optimizer, ids
        m = model.backbone
        self.original_forward = m.forward
        self.original_step = optimizer.step
        self.original_zero = optimizer.zero_grad
        self.widths = tuple(int(h['model']['d_block']) for h in hyperparameters)
        self.depths = tuple(int(h['model']['n_blocks']) - 1 for h in hyperparameters)
        dropout = [float(h['model']['dropout']) for h in hyperparameters]
        self.fixed = FixedSlotMLP(
            self.widths, self.depths, dropout, device=next(model.parameters()).device
        )
        self.fixed_optimizer = FixedSlotMLPOptimizer(
            self.fixed,
            muon_lr=[h['optimizer']['muon_lr'] for h in hyperparameters],
            adamw_lr=[h['optimizer']['adamw_lr'] for h in hyperparameters],
            weight_decay=[h['optimizer']['weight_decay'] for h in hyperparameters],
        )
        self.columns = [sum(self.widths[:i]) for i in range(len(self.widths))]
        self.woffsets = [
            sum(w * w for w in self.widths[:i]) for i in range(len(self.widths))
        ]
        with torch.no_grad():
            for layer, (w, b) in enumerate(
                zip(self.fixed.weights, self.fixed.biases), start=1
            ):
                native = m.blocks[layer].linear
                for i, width in enumerate(self.widths):
                    w[self.woffsets[i] : self.woffsets[i] + width * width].copy_(
                        native.weight[i, :width, :width].reshape(-1)
                    )
                    b[self.columns[i] : self.columns[i] + width].copy_(
                        native.bias[i, :width]
                    )
        hidden = {p for block in m.blocks[1:] for p in block.linear.parameters()}
        for group in optimizer.param_groups:
            group['params'] = [p for p in group['params'] if p not in hidden]
        optimizer.param_groups[:] = [g for g in optimizer.param_groups if g['params']]
        self.cache = {}
        self.cached_ids = None
        self.calls = self.fallback_calls = self.rebuilds = self.syncs = 0
        self.dirty = True
        m.forward = self.forward
        optimizer.step = self.step
        optimizer.zero_grad = self.zero_grad

    @torch.inference_mode(False)
    def prepare(self, batch, device):
        key = (tuple(self.ids), batch, device)
        if self.cached_ids != key[0]:
            self.cache.clear()
            self.cached_ids = key[0]
        if key not in self.cache:
            self.cache[key] = StableTransfer(
                self.widths,
                self.ids,
                batch,
                self.model.backbone.blocks[0].linear.max_out_features,
                device,
            )
            self.rebuilds += 1
        return self.cache[key]

    @torch.no_grad()
    def sync(self):
        if not self.dirty:
            return
        m = self.model.backbone
        for layer, (w, b) in enumerate(
            zip(self.fixed.weights, self.fixed.biases), start=1
        ):
            native = m.blocks[layer].linear
            selected = [self.widths[i] for i in self.ids]
            wtransfer = MatrixTransfer(
                [(v, v) for v in selected], native.weight.shape, w.device
            )
            wtransfer.descriptors = torch.tensor(
                [(self.woffsets[i], self.widths[i], self.widths[i]) for i in self.ids],
                device=w.device,
                dtype=torch.int64,
            )
            btransfer = MatrixTransfer(
                [(1, v) for v in selected],
                (len(self.ids), 1, native.max_out_features),
                b.device,
            )
            btransfer.descriptors = torch.tensor(
                [(self.columns[i], 1, self.widths[i]) for i in self.ids],
                device=b.device,
                dtype=torch.int64,
            )
            native.weight.copy_(wtransfer.unpack(w, w.dtype))
            native.bias.copy_(btransfer.unpack(b, b.dtype)[:, 0])
        self.dirty = False
        self.syncs += 1

    def forward(self, x, pack_idx=None):
        if pack_idx is not None:
            self.sync()
            self.fallback_calls += 1
            return self.original_forward(x, pack_idx)
        if not self.model.backbone.training:
            self.sync()
        m = self.model.backbone
        x = m.blocks[0](x)
        if not self.fixed.weights:
            return x
        transfer = self.prepare(x.shape[1], x.device)
        x = _Pack.apply(x, transfer, x.dtype)
        self.fixed.train(m.training)
        x = self.fixed(x, tuple(self.ids))
        self.calls += 1
        return _Unpack.apply(x, transfer)

    def step(self, *args, **kwargs):
        result = self.original_step(*args, **kwargs)
        self.fixed_optimizer.step(tuple(self.ids))
        self.dirty = True
        return result

    def zero_grad(self, *args, **kwargs):
        result = self.original_zero(*args, **kwargs)
        self.fixed_optimizer.zero_grad()
        return result

    def remove(self):
        self.model.backbone.forward = self.original_forward
        self.optimizer.step = self.original_step
        self.optimizer.zero_grad = self.original_zero
        self.cache.clear()
