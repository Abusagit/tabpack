"""Compact hidden activations inside native TabPack training.

The first block and output head stay native. Padded FP32 master parameters,
optimizer state and checkpoint schemas are retained. Conversion of hidden
weights/gradients is included in execution; activations are converted only at
the two boundaries of the connected hidden stack.
"""

from typing import Any

import torch
from torch import nn
from torch.autograd.function import once_differentiable

from tabpack_experimental.compact_mlp import CompactBlock, _CompactLayer
from tabpack_experimental.ragged_linear import ActivationTransfer, MatrixTransfer


class _Pack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, source, transfer, dtype):
        ctx.transfer, ctx.dtype = transfer, source.dtype
        return transfer.pack(source, dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        return ctx.transfer.unpack(gradient.contiguous(), ctx.dtype), None, None


class _Unpack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, source, transfer):
        ctx.transfer = transfer
        return transfer.unpack(source, source.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        return ctx.transfer.pack(gradient, gradient.dtype), None


def _stamp(value):
    if not isinstance(value, torch.Tensor):
        return value
    return id(value), None if value.is_inference() else value._version


class CompactBackboneBackend:
    """Instance-local adapter, rebuilt automatically after pack removal/replacement."""

    def __init__(self, backbone, *, fused=False):
        if any(not isinstance(b.activation, nn.ReLU) for b in backbone.blocks):
            raise ValueError('The compact backbone currently requires ReLU')
        self.backbone = backbone
        self.original_forward = backbone.forward
        self.fused = fused
        self.signature = None
        self.cache: dict[Any, Any] = {}
        self.calls = self.fallback_calls = self.rebuilds = 0
        backbone.forward = self.forward

    def _signature(self):
        m = self.backbone
        return (
            _stamp(m.n_blocks),
            tuple(
                (
                    id(b.linear.weight),
                    b.linear.weight.device,
                    tuple(b.linear.weight.shape),
                    _stamp(b.linear.in_features),
                    _stamp(b.linear.out_features),
                    _stamp(b.dropout.p),
                )
                for b in m.blocks
            ),
        )

    @torch.inference_mode(False)
    def prepare(self, batch, dtype):
        m = self.backbone
        signature = self._signature()
        if self.signature != signature:
            self.cache.clear()
            self.signature = signature
            self.widths = tuple(
                m.blocks[0].linear.out_features.tolist()
                if m.blocks[0].linear.out_features is not None
                else [m.blocks[0].linear.max_out_features] * m.pack_size
            )
            self.depths = tuple(
                m.n_blocks.tolist()
                if m.n_blocks is not None
                else [m.max_n_blocks] * m.pack_size
            )
            self.rebuilds += 1
        key = batch, dtype
        if key not in self.cache:
            device = m.blocks[0].linear.weight.device
            transfer = ActivationTransfer(
                [(batch, w) for w in self.widths],
                (m.pack_size, batch, m.blocks[0].linear.max_out_features),
                device,
            )
            layers = []
            for layer in range(1, max(self.depths)):
                native = m.blocks[layer]
                members = tuple(i for i, d in enumerate(self.depths) if d > layer)
                dropout = native.dropout.p
                dropout = (
                    dropout.tolist()
                    if isinstance(dropout, torch.Tensor)
                    else [dropout] * m.pack_size
                )
                block = CompactBlock(
                    self.widths,
                    batch,
                    members,
                    device=device,
                    dtype=dtype,
                    schedule='k_desc',
                    tiled=batch % 16 == 0,
                    large_tile=(64, 64, 32),
                    dropout=dropout,
                    fused=self.fused,
                    scalar_dropout=self.fused,
                    initialize_parameters=False,
                )
                active = set(members)
                weight = MatrixTransfer(
                    [(w if i in active else 0, w) for i, w in enumerate(self.widths)],
                    native.linear.weight.shape,
                    device,
                )
                bias = MatrixTransfer(
                    [(int(i in active), w) for i, w in enumerate(self.widths)],
                    (m.pack_size, 1, native.linear.max_out_features),
                    device,
                )
                layers.append((native, block, weight, bias))
            self.cache[key] = transfer, layers
        return self.cache[key]

    def forward(self, x, pack_idx=None):
        m = self.backbone
        dtype = (
            torch.get_autocast_dtype('cuda')
            if torch.is_autocast_enabled('cuda')
            else x.dtype
        )
        if (
            pack_idx is not None
            or x.device.type != 'cuda'
            or dtype not in (torch.bfloat16, torch.float16)
        ):
            self.fallback_calls += 1
            return self.original_forward(x, pack_idx)
        transfer, layers = self.prepare(x.shape[1], dtype)
        x = m.blocks[0](x)
        if not layers:
            return x
        x = _Pack.apply(x, transfer, dtype)
        count = sum(m.training and b.stochastic for _, b, _, _ in layers)
        seeds = iter(
            torch.randint(2**63 - 1, (count,), device=x.device).unbind()
            if count
            else ()
        )
        for native, block, weight, bias in layers:
            block.train(m.training)
            cw = _Pack.apply(native.linear.weight, weight, dtype)
            cb = _Pack.apply(native.linear.bias[:, None], bias, dtype)
            seed = next(seeds) if m.training and block.stochastic else None
            x = _CompactLayer.apply(x, cw, cb, seed, block)
        self.calls += 1
        return _Unpack.apply(x, transfer)

    def remove(self):
        self.backbone.forward = self.original_forward
        self.cache.clear()
