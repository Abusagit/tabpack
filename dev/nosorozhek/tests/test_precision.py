"""Dtype policy and autocast checks for the two grouped Triton kernels."""

import pytest
import torch
import triton.language as tl

from dev.nosorozhek.custom.kernels.newton_schulz import NewtonSchulz
from dev.nosorozhek.custom.kernels.utils import resolve_matmul_dtypes
from dev.nosorozhek.custom.triton_linear_pack import TritonGroupedLinearPack
from tabpackage import nn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('allow_tf32', [False, True])
def test_matmul_dtypes(dtype, allow_tf32):
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        x = torch.empty(0, device='cuda', dtype=dtype)
        expected = 'tf32' if dtype == torch.float32 and allow_tf32 else 'ieee'
        assert resolve_matmul_dtypes(x, x) == (dtype, tl.float32, expected)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.parametrize('amp_dtype', [torch.float16, torch.bfloat16])
def test_matmul_autocast_policy(amp_dtype):
    lhs = torch.empty(0, device='cuda', dtype=torch.float32)
    rhs = lhs.to(torch.bfloat16)
    with torch.autocast('cuda', dtype=amp_dtype):
        assert resolve_matmul_dtypes(lhs, rhs, autocast=True) == (
            amp_dtype,
            tl.float32,
            'ieee',
        )
        # Muon retains its explicit BF16 arithmetic even inside an AMP context.
        assert resolve_matmul_dtypes(rhs, rhs) == (torch.bfloat16, tl.float32, 'ieee')


@pytest.mark.parametrize('rhs_dtype', [torch.float16, torch.float64, torch.int32])
def test_matmul_rejects_invalid_dtypes(rhs_dtype):
    lhs = torch.empty(0, device='cuda')
    with pytest.raises(TypeError):
        resolve_matmul_dtypes(lhs, lhs.to(rhs_dtype))


def test_matmul_rejects_cpu_operands():
    with pytest.raises(ValueError):
        resolve_matmul_dtypes(torch.empty(0), torch.empty(0, device='cuda'))


@pytest.mark.parametrize('amp_dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('input_dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_linear_autocast(amp_dtype, input_dtype):
    torch.manual_seed(42)
    kwargs = {
        'in_features': [5, 16],
        'out_features': [7, 24],
        'max_in_features': 16,
        'max_out_features': 24,
        'pack_size': 2,
        'device': 'cuda',
    }
    original = nn.LinearPack(**kwargs)
    custom = TritonGroupedLinearPack(**kwargs)
    custom.load_state_dict(original.state_dict())
    source = torch.randn(2, 33, 16, device='cuda', dtype=input_dtype)
    x = source.clone().requires_grad_()
    y = source.clone().requires_grad_()
    mask = (
        torch.arange(16, device='cuda')[None, None, :]
        < torch.tensor([5, 16], device='cuda')[:, None, None]
    )
    with torch.autocast('cuda', dtype=amp_dtype):
        expected = original(x * mask)
        actual = custom(y * mask)
    assert actual.dtype == expected.dtype == amp_dtype
    grad = torch.randn_like(expected)
    expected.backward(grad)
    actual.backward(grad)
    assert y.grad.dtype == x.grad.dtype == input_dtype
    assert custom.weight.grad.dtype == original.weight.grad.dtype == torch.float32
    for value, reference in (
        (actual, expected),
        (y.grad, x.grad),
        (custom.weight.grad, original.weight.grad),
        (custom.bias.grad, original.bias.grad),
    ):
        torch.testing.assert_close(value, reference, rtol=0.01, atol=0.02)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_newton_schulz_gemm_dtypes(dtype):
    torch.manual_seed(42)
    module = nn.LinearPack(in_features=16, out_features=16, pack_size=2, device='cuda')
    plan = NewtonSchulz(module, module.weight.device, (32, 32, 32, 4, 2))
    lhs = torch.randn(2, 16, 16, device='cuda', dtype=dtype)
    rhs = torch.randn_like(lhs)
    actual = torch.empty_like(lhs)
    plan._gemm(lhs, rhs, actual, actual, 0)
    torch.testing.assert_close(actual, torch.bmm(lhs, rhs), rtol=0.01, atol=0.02)
