"""TabPack Muon optimizer with grouped Triton orthogonalization."""

from torch import Tensor

from tabpackage import nn, optim

from .kernels.newton_schulz import NewtonSchulz


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
        self._modules_by_weight = {}
        self._plans = {}
        super().__init__(*args, **kwargs)

    def add_param_group(self, param_group: dict) -> None:
        module = param_group.pop('_triton_module', None)
        super().add_param_group(param_group)
        if module is not None:
            self._modules_by_weight[module.weight] = module

    def replace(self, **kwargs) -> None:
        super().replace(**kwargs)
        self._plans.clear()

    def _step_muon(self, group: dict) -> None:
        (weight,) = group['params']
        module = self._modules_by_weight.get(weight)
        if (
            module is None
            or weight.grad is None
            or weight.device.type != 'cuda'
            or weight.shape[0] > 128
        ):
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
        cached = self._plans.get(weight)
        if cached is None or cached[0] != key:
            cached = key, NewtonSchulz(module, weight.device, options)
            self._plans[weight] = cached
        _, plan = cached
        stock = self._zeropower_via_newtonschulz5

        def orthogonalize(update: Tensor, steps: int) -> Tensor:
            # Stock Muon can compact active rows without returning their indices.
            if update.shape != weight.shape:
                return stock(update, steps)
            return plan(update, steps)

        self._zeropower_via_newtonschulz5 = orthogonalize
        try:
            super()._step_muon(group)
        finally:
            self._zeropower_via_newtonschulz5 = stock
