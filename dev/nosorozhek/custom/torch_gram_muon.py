"""Muon with the PyTorch Gram Newton-Schulz reference.

Adapted from https://github.com/Dao-AILab/gram-newton-schulz
"""

import torch
from torch import Tensor
from tabpackage import optim

GRAM_COEFFICIENTS = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)
GRAM_RESET_ITERATION = 2


def validate_gram_steps(steps: int) -> None:
    if not isinstance(steps, int) or not 0 <= steps <= len(GRAM_COEFFICIENTS):
        raise ValueError(
            f'Gram Newton-Schulz requires 0 to {len(GRAM_COEFFICIENTS)} steps'
        )

# fmt: off
# ruff: noqa: E501
def torch_gram_newton_schulz(X: Tensor, steps: int) -> Tensor:
    """Orthogonalize a batch of updates whose padded entries are zero."""
    validate_gram_steps(steps)
    if X.ndim != 3:
        raise ValueError('Gram Newton-Schulz expects a batch of matrices')
    ns_coefficients = GRAM_COEFFICIENTS[:steps]
    gram_newton_schulz_reset_iterations = {GRAM_RESET_ITERATION}
    ns_epsilon = 1e-7

    tall_skinny = X.size(-2) > X.size(-1)
    X = X.to(torch.float32)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + ns_epsilon)
    X = X.to(torch.float16)
    if steps == 0:
        return X

    if tall_skinny:
        R = X.mT @ X
    else:
        R = X @ X.mT

    batch_size = R.size(0)
    I = torch.eye(R.size(-1), device=X.device, dtype=X.dtype).unsqueeze(0).expand(batch_size, -1, -1).contiguous()
    Q = None

    for i, (a, b, c) in enumerate(ns_coefficients):
        if i in gram_newton_schulz_reset_iterations and i != 0:
            if tall_skinny:
                X = X @ Q
                R = X.mT @ X
            else:
                X = Q @ X
                R = X @ X.mT
            Q = None

        Z = torch.baddbmm(R, R, R, alpha=c, beta=b)
        if i == 0 or i in gram_newton_schulz_reset_iterations:
            Q = Z + a * I
        else:
            Q = torch.baddbmm(Q, Q, Z, beta=a)
        if i < len(ns_coefficients) - 1 and i + 1 not in gram_newton_schulz_reset_iterations:
            RZ = torch.baddbmm(R, R, Z, beta=a)
            R = torch.baddbmm(RZ, Z, RZ, beta=a)

    if tall_skinny:
        X = X @ Q
    else:
        X = Q @ X
    return X
# fmt: on


class TorchGramMuonAdamWPack(optim.MuonAdamWPack):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._zeropower_via_newtonschulz5 = torch_gram_newton_schulz

    def step(self, closure=None):
        for group in self.param_groups:
            if group['muon']:
                validate_gram_steps(group['muon_newton_schulz_steps'])
        return super().step(closure)
