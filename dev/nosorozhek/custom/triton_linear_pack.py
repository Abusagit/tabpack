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
        self._prefix_cache: dict[
            tuple[int, torch.device, int, int],
            tuple[Tensor, Tensor, Tensor, int, int, int],
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
    def _get_prefix_sums(
        self, batch_size: int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, int, int, int]:
        key = (batch_size, self.weight.device, self.block_m, self.block_n)
        cached = self._prefix_cache.get(key)
        if cached is None:
            in_sizes, out_sizes = self._sizes()
            in_values = in_sizes.detach().cpu().tolist()
            out_values = out_sizes.detach().cpu().tolist()

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
                prefix_out,
                prefix_x,
                prefix_weight,
                total_out,
                total_x,
                total_weight,
            )
            self._prefix_cache[key] = cached
        else:
            in_sizes, out_sizes = self._sizes()
        prefix_out, prefix_x, prefix_weight, total_out, total_x, total_weight = cached
        return (
            in_sizes,
            out_sizes,
            prefix_out,
            prefix_x,
            prefix_weight,
            total_out,
            total_x,
            total_weight,
        )

    def forward(self, x: Tensor, pack_idx: Tensor | None = None) -> Tensor:
        if pack_idx is not None or self.pack_size > 128:
            return super().forward(x, pack_idx)
        if x.ndim != 3 or x.shape[0] != self.pack_size:
            raise ValueError('TritonGroupedLinearPack expects a full packed input')
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
