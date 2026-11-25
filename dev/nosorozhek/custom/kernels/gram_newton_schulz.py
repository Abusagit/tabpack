"""Gram Newton-Schulz orthogonalization using grouped GEMMs."""

from __future__ import annotations

import torch
from torch import Tensor

from ..torch_gram_muon import (
    GRAM_COEFFICIENTS,
    GRAM_RESET_ITERATION,
    validate_gram_steps,
)
from .newton_schulz import NewtonSchulz


class GramNewtonSchulz(NewtonSchulz):
    """Orthogonalize by iterating on the smaller symmetric Gram matrix."""

    @torch._dynamo.disable
    def __call__(self, update: Tensor, steps: int) -> Tensor:
        validate_gram_steps(steps)
        if update.ndim != 3:
            raise ValueError('Gram Newton-Schulz expects a batch of matrices')
        x = update.float()
        x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        x = x.half()
        if self.transpose:
            x = x.mT
        if steps == 0:
            return x.mT if self.transpose else x

        count, rows, columns = x.shape
        gram = x.new_zeros(count, rows, rows)
        q = x.new_zeros(count, rows, rows)
        q_next = x.new_zeros(count, rows, rows)
        z = x.new_zeros(count, rows, rows)
        rz = x.new_zeros(count, rows, rows)
        result = x.new_zeros(count, rows, columns)
        eye = torch.eye(rows, device=x.device, dtype=x.dtype).expand(count, -1, -1)

        self._gemm(x, x.mT, z, gram, 0)

        for step, (a, b, c) in enumerate(GRAM_COEFFICIENTS[:steps]):
            if step == GRAM_RESET_ITERATION:
                self._gemm(q.mT if self.transpose else q, x, x, result, 2)
                x, result = result, x
                self._gemm(x, x.mT, z, gram, 0)

            self._gemm(gram, gram, gram, z, 1, alpha=c, beta=b)
            if step in (0, GRAM_RESET_ITERATION):
                q.copy_(z + a * eye)
            else:
                self._gemm(q, z, q, q_next, 1, alpha=1.0, beta=a)
                q, q_next = q_next, q

            if step + 1 < steps and step + 1 != GRAM_RESET_ITERATION:
                self._gemm(gram, z, gram, rz, 1, beta=a)
                self._gemm(z, rz, rz, gram, 1, beta=a)

        self._gemm(q.mT if self.transpose else q, x, x, result, 2)
        x = result
        return x.mT if self.transpose else x
