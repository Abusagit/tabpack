"""Compare original and Triton packs using identical random data."""

import pytest
import torch

from dev.nosorozhek.custom.triton_linear_pack import TritonGroupedLinearPack
from dev.nosorozhek.custom.triton_muon import TritonMuonAdamWPack
from tabpackage import nn, optim

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')

PACK_SHAPES = [
    (1, 17, 33),
    (4, 65, 97),
    (64, 128, 64),
    (200, 32, 48),
]


@pytest.fixture(params=[False, True])
def allow_tf32(request):
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = request.param
    try:
        yield request.param
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.parametrize('pack_size,max_in_features,max_out_features', PACK_SHAPES)
@pytest.mark.parametrize('batch_size', [1, 33, 128, 1024])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_linear_pack(
    pack_size,
    max_in_features,
    max_out_features,
    batch_size,
    dtype,
    allow_tf32,
):
    torch.manual_seed(42)
    in_features = torch.randint(1, max_in_features + 1, (pack_size,)).tolist()
    out_features = torch.randint(1, max_out_features + 1, (pack_size,)).tolist()
    kwargs = {
        'in_features': in_features,
        'out_features': out_features,
        'max_in_features': max_in_features,
        'max_out_features': max_out_features,
        'pack_size': pack_size,
        'device': 'cuda',
    }
    original = nn.LinearPack(**kwargs).to(dtype)
    custom = TritonGroupedLinearPack(**kwargs).to(dtype)
    custom.load_state_dict(original.state_dict())

    source = torch.randn(
        pack_size, batch_size, max_in_features, device='cuda', dtype=dtype
    )
    original_input = source.clone().requires_grad_()
    custom_input = source.clone().requires_grad_()
    mask = (
        torch.arange(max_in_features, device='cuda')[None, None, :]
        < torch.tensor(in_features, device='cuda')[:, None, None]
    )
    # LinearPack expects padded inputs to be masked by its caller.
    expected = original(original_input * mask)
    actual = custom(custom_input * mask)
    gradient = torch.randn(
        pack_size, batch_size, 2 * max_out_features, device='cuda', dtype=dtype
    )[..., ::2]
    expected.backward(gradient)
    actual.backward(gradient)

    full_precision = dtype == torch.float32 and not allow_tf32
    rtol, atol = (1e-4, 2e-4) if full_precision else (0.01, 0.02)
    for value, reference in (
        (actual, expected),
        (custom_input.grad, original_input.grad),
        (custom.bias.grad, original.bias.grad),
    ):
        torch.testing.assert_close(value, reference, rtol=rtol, atol=atol)
    # Reduced-precision batch reductions need a larger absolute tolerance.
    reduction_tolerance = 0.003 if dtype == torch.float32 and allow_tf32 else 0.002
    weight_atol = (
        atol if full_precision else max(atol, reduction_tolerance * batch_size**0.5)
    )
    torch.testing.assert_close(
        custom.weight.grad, original.weight.grad, rtol=rtol, atol=weight_atol
    )


@pytest.mark.parametrize('pack_size,max_in_features,max_out_features', PACK_SHAPES)
@pytest.mark.parametrize('newton_schulz_steps', [1, 5])
def test_muon_pack(pack_size, max_in_features, max_out_features, newton_schulz_steps):
    torch.manual_seed(42)
    in_features = torch.randint(1, max_in_features + 1, (pack_size,)).tolist()
    out_features = torch.randint(1, max_out_features + 1, (pack_size,)).tolist()
    kwargs = {
        'in_features': in_features,
        'out_features': out_features,
        'max_in_features': max_in_features,
        'max_out_features': max_out_features,
        'pack_size': pack_size,
        'device': 'cuda',
    }
    original = nn.LinearPack(**kwargs)
    custom = nn.LinearPack(**kwargs)
    custom.load_state_dict(original.state_dict())
    optimizers = [
        cls(
            [
                cls.make_muon_parameter_group(layer),
                {'params': [layer.bias]},
            ],
            adamw_lr=0.001,
            muon_lr=0.01,
            weight_decay=0.01,
            muon_newton_schulz_steps=newton_schulz_steps,
            muon_skip_zero_updates=False,
            allow_inplace_grad_mutation=False,
            pack_size=pack_size,
        )
        for cls, layer in (
            (optim.MuonAdamWPack, original),
            (TritonMuonAdamWPack, custom),
        )
    ]
    rows = torch.arange(max_out_features, device='cuda')[None, :, None]
    columns = torch.arange(max_in_features, device='cuda')[None, None, :]
    mask = (rows < torch.tensor(out_features, device='cuda')[:, None, None]) & (
        columns < torch.tensor(in_features, device='cuda')[:, None, None]
    )
    for _ in range(3):
        gradient = torch.randn_like(original.weight) * mask
        bias_gradient = torch.randn_like(original.bias)
        original_before = original.weight.detach().clone()
        custom_before = custom.weight.detach().clone()
        for layer, optimizer in zip((original, custom), optimizers, strict=True):
            layer.weight.grad = gradient.clone()
            layer.bias.grad = bias_gradient.clone()
            optimizer.step()
        # BF16 Newton-Schulz arithmetic can round differently across GEMM kernels.
        torch.testing.assert_close(
            custom.weight - custom_before,
            original.weight - original_before,
            rtol=0.05,
            atol=2e-4,
        )
        torch.testing.assert_close(custom.bias, original.bias, rtol=0, atol=0)
        torch.testing.assert_close(
            optimizers[1].state[custom.weight]['muon_momentum_buffer'],
            optimizers[0].state[original.weight]['muon_momentum_buffer'],
            rtol=0,
            atol=0,
        )
    assert custom in optimizers[1]._plans
