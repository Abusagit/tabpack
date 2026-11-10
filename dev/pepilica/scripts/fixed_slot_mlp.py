"""Fixed-slot square hidden stacks for training experiments."""

import math
from typing import Any

import torch
from torch import nn

if __package__:
    from .prototype_fixed_slots import FixedSlotPlan
else:
    from dev.pepilica.scripts.prototype_fixed_slots import FixedSlotPlan
from tabpack_experimental.grouped_fixed_optimizer import GroupedFixedMuonAdamW


class FixedSlotMLP(nn.Module):
    keep: torch.Tensor

    def __init__(self, widths, depths, dropout, *, device='cuda', seed=42):
        super().__init__()
        self.widths = tuple(map(int, widths))
        self.depths = tuple(map(int, depths))
        self.dropout = tuple(map(float, dropout))
        if (
            not self.widths
            or min(self.widths) < 1
            or len(self.widths) != len(self.depths)
            or len(self.widths) != len(self.dropout)
            or min(self.depths) < 0
            or any(not 0 <= p < 1 for p in self.dropout)
        ):
            raise ValueError('Invalid member widths, depths or dropout')
        self.stride = sum(self.widths)
        self.weight_size = sum(w * w for w in self.widths)
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        generator = torch.Generator(device=device).manual_seed(seed)
        for _ in range(max(self.depths)):
            weight = torch.empty(self.weight_size, device=device, dtype=torch.float32)
            offset = 0
            for w in self.widths:
                weight[offset : offset + w * w].uniform_(
                    -1 / math.sqrt(w), 1 / math.sqrt(w), generator=generator
                )
                offset += w * w
            self.weights.append(nn.Parameter(weight))
            self.biases.append(nn.Parameter(torch.zeros(self.stride, device=device)))
        self._plans = {}
        self._masks = {}
        self.register_buffer(
            'keep',
            torch.tensor(
                [1 - p for p, w in zip(self.dropout, self.widths) for _ in range(w)],
                device=device,
            ),
        )

    def layer_active(self, active, layer):
        return tuple(i for i in active if self.depths[i] > layer)

    def forward(self, x, active):  # ty: ignore[invalid-mutable-override]
        active = tuple(int(i) for i in active)
        if len(set(active)) != len(active) or any(
            i < 0 or i >= len(self.widths) for i in active
        ):
            raise ValueError('Invalid active set')
        if (
            x.ndim != 2
            or x.shape[1] != self.stride
            or x.dtype not in (torch.bfloat16, torch.float16)
        ):
            raise ValueError('Expected low-precision batch-major fixed-slot input')
        for layer, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            selected = self.layer_active(active, layer)
            if not selected:
                continue
            key = (x.shape[0], selected, x.device)
            if key not in self._plans:
                with torch.inference_mode(False):
                    self._plans[key] = FixedSlotPlan(
                        self.widths, x.shape[0], selected, device=x.device
                    )
            y = torch.relu(self._plans[key](x, weight.to(x.dtype), bias.to(x.dtype)))
            if self.training and any(self.dropout[i] > 0 for i in selected):
                y = y * (
                    (torch.rand(y.shape, device=y.device) < self.keep) / self.keep
                ).to(y.dtype)
            if key not in self._masks:
                with torch.inference_mode(False):
                    self._masks[key] = torch.tensor(
                        [
                            i in selected
                            for i, w in enumerate(self.widths)
                            for _ in range(w)
                        ],
                        device=x.device,
                        dtype=torch.bool,
                    )
            x = torch.where(self._masks[key], y, x)
        return x


class FixedSlotMLPOptimizer:
    def __init__(self, model, **kwargs):
        self.model = model
        self.optimizers: list[Any] = [
            GroupedFixedMuonAdamW(w, b, model.widths, **kwargs)
            for w, b in zip(model.weights, model.biases)
        ]

    def step(self, active):
        active = tuple(active)
        for layer, optimizer in enumerate(self.optimizers):
            optimizer.step(self.model.layer_active(active, layer))

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()
