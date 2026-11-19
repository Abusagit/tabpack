"""TabPack Muon optimizer with grouped Triton orthogonalization."""

import logging

from torch import Tensor

from tabpackage import nn, optim

from .kernels.newton_schulz import NewtonSchulz


_logger = logging.getLogger(__name__)

class TritonMuonAdamWPack(optim.MuonAdamWPack):
    """Reuse TabPack's optimizer updates with a grouped CUDA orthogonalization."""

    block_m = 64
    block_n = 64
    block_k = 64
    num_warps = 4
    num_stages = 2

    @staticmethod
    def make_muon_parameter_group(module: nn.LinearPack) -> dict:
        group = optim.MuonAdamWPack.make_muon_parameter_group(module)
        group['_triton_module'] = module
        return group

    def __init__(self, *args, **kwargs) -> None:
        self._muon_modules: list[nn.LinearPack] = []
        self._plans = {}
        super().__init__(*args, **kwargs)

    def add_param_group(self, param_group: dict) -> None:
        module = param_group.pop('_triton_module', None)
        super().add_param_group(param_group)
        if module is not None:
            self._muon_modules.append(module)

    def replace(self, **kwargs) -> None:
        super().replace(**kwargs)
        self._plans.clear()

    def _step_muon(self, group: dict) -> None:
        (weight,) = group['params']
        # Pack removal replaces the Parameter but keeps the module.
        module = next((m for m in self._muon_modules if m.weight is weight), None)
        if (
            module is None
            or weight.grad is None
            or weight.device.type != 'cuda'
        ):
            _logger.warning(
                'Triton Muon falling back to the base optimizer '
                '(weight_shape=%s, device=%s, module_found=%s, gradient_present=%s)',
                tuple(weight.shape),
                weight.device,
                module is not None,
                weight.grad is not None,
            )
            return super()._step_muon(group)
        options = (
            self.block_m,
            self.block_n,
            self.block_k,
            self.num_warps,
            self.num_stages,
        )
        features = tuple(
            (t.data_ptr(), tuple(t.tolist()) if t.is_inference() else t._version)
            for t in (module.in_features, module.out_features)
            if t is not None
        )
        key = (weight.device, weight.shape, features, options)
        cached = self._plans.get(module)
        if cached is None or cached[0] != key:
            cached = key, NewtonSchulz(module, weight.device, options)
            self._plans[module] = cached
        _, plan = cached
        stock = self._zeropower_via_newtonschulz5

        def orthogonalize(update: Tensor, steps: int) -> Tensor:
            # Stock Muon can compact active rows without returning their indices.
            if update.shape != weight.shape:
                _logger.warning(
                    'Triton Muon falling back to the base orthogonalization '
                    'for a compacted update (update_shape=%s, weight_shape=%s)',
                    tuple(update.shape),
                    tuple(weight.shape),
                )
                return stock(update, steps)
            return plan(update, steps)

        self._zeropower_via_newtonschulz5 = orthogonalize
        try:
            super()._step_muon(group)
        finally:
            self._zeropower_via_newtonschulz5 = stock
