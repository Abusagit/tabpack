"""LinearPack variants using explicit COO-like tile coordinates."""

from __future__ import annotations

import torch
from torch import Tensor

from .kernels.linear_coo import grouped_linear_coo, make_specs
from .triton_linear_pack import TritonGroupedLinearPack


class TritonCOOLinearPack(TritonGroupedLinearPack):
    swizzle = 'row'

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._coo_cache: dict[tuple, tuple] = {}

    def _invalidate_prefix_sums(self) -> None:
        super()._invalidate_prefix_sums()
        self._coo_cache.clear()

    @torch._dynamo.disable
    @torch.inference_mode(False)
    def _coo_specs(self, batch_size: int):
        in_sizes, out_sizes, *_ = self._get_prefix_sums(batch_size)
        key = (batch_size, self.swizzle)
        cached = self._coo_cache.get(key)
        if cached is None:
            cached = (
                in_sizes,
                out_sizes,
                *make_specs(
                    in_sizes,
                    out_sizes,
                    batch_size,
                    self.block_m,
                    self.block_n,
                    self.swizzle,
                ),
            )
            self._coo_cache[key] = cached
        return cached

    def forward(self, x: Tensor, pack_idx: Tensor | None = None) -> Tensor:
        if pack_idx is not None or self.pack_size > 128:
            return super().forward(x, pack_idx)
        if x.ndim != 3 or x.shape[0] != self.pack_size:
            raise ValueError('TritonCOOLinearPack expects a full packed input')
        if x.shape[-1] != self.max_in_features:
            raise ValueError('The input width must match the packed weight width')
        if x.device.type != 'cuda' or x.shape[1] == 0:
            return super().forward(x, pack_idx)
        in_sizes, out_sizes, forward, grad_x, grad_weight = self._coo_specs(x.shape[1])
        bias = self.bias if self.bias is not None else x.new_empty(0)
        return grouped_linear_coo(
            x,
            self.weight,
            bias,
            (forward, grad_x, grad_weight),
            (self.block_m, self.block_n, self.block_k, self.num_warps, self.num_stages),
        )


class TritonCOOLinearPackZ(TritonCOOLinearPack):
    swizzle = 'z'


class TritonCOOLinearPackHilbert(TritonCOOLinearPack):
    swizzle = 'hilbert'
