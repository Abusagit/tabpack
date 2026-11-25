"""Connect the active-only MLP to native TabPack training and evaluation."""

from collections import OrderedDict

import torch
from torch import nn

from tabpack_experimental.active_fixed_slot import ActiveFixedSlotMLP
from tabpack_experimental.fixed_slot_matrix import FeatureColumnTransfer
from tabpack_experimental.grouped_fixed_optimizer import GroupedFixedMuonAdamW
from tabpack_experimental.transfers import ActivationTransfer, _Pack, _Unpack


class ActiveTabPackBackend:
    """Group hidden layers and optionally the first layer; keep the native head.

    ``ids`` is the runner's mutable list of surviving original candidate IDs.
    Padded native weights remain as evaluation/checkpoint compatibility copies.
    """

    def __init__(self, model, optimizer, ids, hyperparameters, *, grouped_first=True):
        self.model, self.optimizer, self.ids = model, optimizer, ids
        self.grouped_first = bool(grouped_first)
        backbone = model.backbone
        if any(not isinstance(b.activation, nn.ReLU) for b in backbone.blocks):
            raise ValueError('The active TabPack adapter requires ReLU blocks')
        self.widths = tuple(int(h['model']['d_block']) for h in hyperparameters)
        self.depths = tuple(int(h['model']['n_blocks']) - 1 for h in hyperparameters)
        self.dropout = tuple(float(h['model']['dropout']) for h in hyperparameters)
        first = backbone.blocks[0].linear
        self.input_widths = (
            tuple(
                [first.max_in_features] * len(ids)
                if first.in_features is None
                else first.in_features.tolist()
            )
            if self.grouped_first
            else None
        )
        self.embedding_widths = (
            None
            if model.num_module is None
            else tuple(
                [model.num_module.max_d_embedding] * len(ids)
                if model.num_module.d_embedding is None
                else model.num_module.d_embedding.tolist()
            )
        )
        self.prototype = ActiveFixedSlotMLP(
            self.widths,
            1,
            self.depths,
            input_widths=self.input_widths,
            device='meta',
            dtype=torch.float32,
            dropout=self.dropout,
            tile=64,
        )
        self.layers = []
        if self.grouped_first:
            self.layers.append((self.prototype.input_layer, first, None))
        self.layers.extend(
            (self.prototype.blocks[layer], backbone.blocks[layer + 1].linear, layer)
            for layer in range(max(self.depths))
        )
        self.grouped_optimizers = []
        native_parameters = set()
        columns = self._columns() if self.grouped_first else None
        with torch.no_grad():
            for block, native, layer in self.layers:
                pieces, biases = [], []
                for i, (h, d) in enumerate(block.shapes):
                    value = native.weight[i, :h]
                    value = (
                        value.index_select(
                            1, torch.tensor(columns[i], device=value.device)
                        )
                        if layer is None
                        else value[:, :d]
                    )
                    pieces.append(value.reshape(-1))
                    biases.append(native.bias[i, :h])
                block.weight = nn.Parameter(torch.cat(pieces).clone())
                block.bias = nn.Parameter(torch.cat(biases).clone())
                native_parameters.update(native.parameters())
                self.grouped_optimizers.append(
                    GroupedFixedMuonAdamW(
                        block.weight,
                        block.bias,
                        block.shapes,
                        tile=64,
                        **{
                            k: [h['optimizer'][k] for h in hyperparameters]
                            for k in ('muon_lr', 'adamw_lr', 'weight_decay')
                        },
                    )
                )
        for group in optimizer.param_groups:
            group['params'] = [p for p in group['params'] if p not in native_parameters]
        optimizer.param_groups[:] = [g for g in optimizer.param_groups if g['params']]
        self.original_forward = backbone.forward
        self.original_step, self.original_zero = optimizer.step, optimizer.zero_grad
        self.cache, self.input_transfers, self.models = (
            OrderedDict(),
            OrderedDict(),
            OrderedDict(),
        )
        self.cached_ids = None
        self.calls = self.fallback_calls = self.rebuilds = self.syncs = 0
        self.dirty = True
        backbone.forward, optimizer.step, optimizer.zero_grad = (
            self.forward,
            self.step,
            self.zero_grad,
        )

    def _columns(self):
        count = self.model._n_numerical_features
        cats = sum(self.model._categorical_cardinalities)
        if self.embedding_widths is None:
            return [list(range(count + cats)) for _ in self.widths]
        maximum = self.model.num_module.max_d_embedding
        return [
            [feature * maximum + j for feature in range(count) for j in range(width)]
            + list(range(count * maximum, count * maximum + cats))
            for width in self.embedding_widths
        ]

    def _compact_model(self, batch):
        if batch not in self.models:
            with torch.inference_mode(False):
                model = ActiveFixedSlotMLP(
                    self.widths,
                    batch,
                    self.depths,
                    input_widths=self.input_widths,
                    device='meta',
                    dtype=torch.float32,
                    dropout=self.dropout,
                    tile=64,
                )
                if self.grouped_first:
                    model.input_layer = self.prototype.input_layer
                for new, old in zip(model.blocks, self.prototype.blocks):
                    new.weight, new.bias = old.weight, old.bias
            self.models[batch] = model
            if len(self.models) > 32:
                self.models.popitem(last=False)
        self.models.move_to_end(batch)
        return self.models[batch]

    @torch.inference_mode(False)
    def prepare(self, batch, device):
        key = (tuple(self.ids), batch, device)
        if self.cached_ids != key[0]:
            self.cache.clear()
            self.input_transfers.clear()
            self.cached_ids = key[0]
        if key not in self.cache:
            self.cache[key] = ActivationTransfer(
                [(batch, self.widths[i]) for i in self.ids],
                (
                    len(self.ids),
                    batch,
                    self.model.backbone.blocks[0].linear.max_out_features,
                ),
                device,
            )
            self.rebuilds += 1
            if len(self.cache) > 32:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

    def forward(self, x, pack_idx=None):
        backbone = self.model.backbone
        if pack_idx is not None:
            if backbone.training and torch.is_grad_enabled():
                raise ValueError('Use routed candidate IDs for training')
            self.sync()
            self.fallback_calls += 1
            return self.original_forward(x, pack_idx)
        if not backbone.training:
            self.sync()
        if not self.grouped_first:
            x = backbone.blocks[0](x)
            if not max(self.depths):
                return x
        transfer = self.prepare(x.shape[1], x.device)
        dtype = (
            torch.get_autocast_dtype(x.device.type)
            if torch.is_autocast_enabled(x.device.type)
            else x.dtype
        )
        if self.grouped_first:
            key = tuple(self.ids)
            if key not in self.input_transfers:
                with torch.inference_mode(False):
                    self.input_transfers[key] = FeatureColumnTransfer(
                        self._columns(), key, device=x.device
                    )
            compact = self.input_transfers[key].pack(x, dtype)
        else:
            compact = _Pack.apply(x, transfer, dtype)
        model = self._compact_model(x.shape[1])
        model.train(backbone.training)
        output = model(compact, tuple(self.ids))
        self.calls += 1
        return _Unpack.apply(output, transfer)

    def step(self, *args, **kwargs):
        result = self.original_step(*args, **kwargs)
        for (_, _, layer), opt in zip(self.layers, self.grouped_optimizers):
            active = (
                tuple(self.ids)
                if layer is None
                else tuple(i for i in self.ids if self.depths[i] > layer)
            )
            opt.step(active)
        self.dirty = True
        return result

    def zero_grad(self, *args, **kwargs):
        result = self.original_zero(*args, **kwargs)
        for opt in self.grouped_optimizers:
            opt.zero_grad()
        return result

    @torch.no_grad()
    def sync(self):
        if not self.dirty:
            return
        columns = self._columns() if self.grouped_first else None
        for block, _, layer in self.layers:
            native = self.model.backbone.blocks[
                0 if layer is None else layer + 1
            ].linear
            wo, bo = 0, 0
            offsets = []
            for h, d in block.shapes:
                offsets.append((wo, bo))
                wo += h * d
                bo += h
            for local, i in enumerate(self.ids):
                h, d = block.shapes[i]
                wo, bo = offsets[i]
                value = block.weight[wo : wo + h * d].view(h, d)
                if layer is None:
                    native.weight[local, :h].index_copy_(
                        1, torch.tensor(columns[i], device=value.device), value
                    )
                else:
                    native.weight[local, :h, :d].copy_(value)
                native.bias[local, :h].copy_(block.bias[bo : bo + h])
        self.dirty = False
        self.syncs += 1

    def remove(self):
        self.sync()
        self.model.backbone.forward = self.original_forward
        self.optimizer.step, self.optimizer.zero_grad = (
            self.original_step,
            self.original_zero,
        )
        self.cache.clear()
        self.input_transfers.clear()
        self.models.clear()
