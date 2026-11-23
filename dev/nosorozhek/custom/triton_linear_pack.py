"""Triton grouped GEMM replacement for TabPack's LinearPack."""

from __future__ import annotations

import torch
from torch import Tensor

from tabpackage import nn

from .kernels.linear import grouped_linear


class TritonGroupedLinearPack(nn.LinearPack):
    """Compute all packed linear layers through one Triton tile stream."""

    block_m = 64
    block_n = 64
    block_k = 64
    num_warps = 4
    num_stages = 2

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._prefix_layout_key: tuple | None = None
        self._prefix_buffers: tuple[Tensor | None, ...] = ()
        self._prefix_cache: dict[
            int, tuple[Tensor, Tensor, Tensor, Tensor, Tensor, int, int, int]
        ] = {}

    def _invalidate_prefix_sums(self) -> None:
        self._prefix_cache.clear()

    def replace(self, **kwargs) -> None:
        super().replace(**kwargs)
        self._invalidate_prefix_sums()

    def _sizes(self) -> tuple[Tensor, Tensor]:
        in_sizes = (
            torch.full(
                (self.pack_size,),
                self.max_in_features,
                dtype=torch.int32,
                device=self.weight.device,
            )
            if self.in_features is None
            else self.in_features.to(dtype=torch.int32)
        )
        out_sizes = (
            torch.full(
                (self.pack_size,),
                self.max_out_features,
                dtype=torch.int32,
                device=self.weight.device,
            )
            if self.out_features is None
            else self.out_features.to(dtype=torch.int32)
        )
        return in_sizes, out_sizes

    @torch._dynamo.disable
    @torch.inference_mode(False)
    def _get_prefix_sums(
        self, batch_size: int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, int, int, int]:
        # Removal replaces buffers, while loading state can mutate them in place.
        buffers = (self.in_features, self.out_features)
        versions = tuple(
            None if t is None else tuple(t.tolist()) if t.is_inference() else t._version
            for t in buffers
        )
        key = (
            self.weight.device,
            self.weight.shape,
            self.block_m,
            self.block_n,
            versions,
        )
        if self._prefix_layout_key != key or any(
            old is not new for old, new in zip(self._prefix_buffers, buffers)
        ):
            self._invalidate_prefix_sums()
            self._prefix_layout_key = key
            self._prefix_buffers = buffers
        cached = self._prefix_cache.get(batch_size)
        if cached is None:
            in_sizes, out_sizes = (
                t.clone() if t.is_inference() else t for t in self._sizes()
            )
            in_values = in_sizes.cpu().tolist()
            out_values = out_sizes.cpu().tolist()
            if not all(0 < i <= self.max_in_features for i in in_values) or not all(
                0 < o <= self.max_out_features for o in out_values
            ):
                raise ValueError('Packed feature counts exceed the linear layout')
            in_sizes = torch.full_like(in_sizes, self.max_in_features)
            in_values = [self.max_in_features] * self.pack_size

            def prefix(rows: int | list[int], columns: list[int]) -> tuple[Tensor, int]:
                values = [0]
                total = 0
                for rows_i, columns_i in zip(
                    [rows] * len(columns) if isinstance(rows, int) else rows,
                    columns,
                ):
                    total += ((rows_i + self.block_m - 1) // self.block_m) * (
                        (columns_i + self.block_n - 1) // self.block_n
                    )
                    values.append(total)
                return (
                    torch.tensor(values, dtype=torch.int32, device=self.weight.device),
                    total,
                )

            prefix_out, total_out = prefix(batch_size, out_values)
            prefix_x, total_x = prefix(batch_size, in_values)
            prefix_weight, total_weight = prefix(out_values, in_values)
            cached = (
                in_sizes,
                out_sizes,
                prefix_out,
                prefix_x,
                prefix_weight,
                total_out,
                total_x,
                total_weight,
            )
            self._prefix_cache[batch_size] = cached
        return cached

    def forward(self, x: Tensor, pack_idx: Tensor | None = None) -> Tensor:
        if pack_idx is not None:
            return super().forward(x, pack_idx)
        if x.ndim != 3 or x.shape[0] != self.pack_size:
            raise ValueError('TritonGroupedLinearPack expects a full packed input')
        if x.shape[-1] != self.max_in_features:
            raise ValueError('The input width must match the packed weight width')
        layout = self._get_prefix_sums(x.shape[1])
        bias = self.bias if self.bias is not None else x.new_empty(0)
        return grouped_linear(
            x,
            self.weight,
            bias,
            layout,
            (
                self.block_m,
                self.block_n,
                self.block_k,
                self.num_warps,
                self.num_stages,
            ),
        )
