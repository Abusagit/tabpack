"""Compare original and Triton packs using identical random data."""

import contextlib

import pytest
import torch

from dev.nosorozhek.custom.kernels.gram_newton_schulz import GramNewtonSchulz
from dev.nosorozhek.custom.kernels.linear_coo import _tile_coordinates
from dev.nosorozhek.custom.torch_gram_muon import (
    TorchGramMuonAdamWPack,
    torch_gram_newton_schulz,
)
from dev.nosorozhek.custom.triton_linear_coo import (
    TritonCOOLinearPack,
    TritonCOOLinearPackHilbert,
    TritonCOOLinearPackZ,
)
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
COO_TYPES = [TritonCOOLinearPack, TritonCOOLinearPackZ, TritonCOOLinearPackHilbert]


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
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize('linear_type', [TritonGroupedLinearPack, *COO_TYPES])
def test_linear_pack(
    pack_size,
    max_in_features,
    max_out_features,
    batch_size,
    dtype,
    allow_tf32,
    linear_type,
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
    custom = linear_type(**kwargs).to(dtype)
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
@pytest.mark.parametrize('algorithm', ['standard', 'gram'])
def test_muon_pack(
    pack_size, max_in_features, max_out_features, newton_schulz_steps, algorithm
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
            **(
                {'newton_schulz_algorithm': algorithm}
                if cls is TritonMuonAdamWPack
                else {}
            ),
        )
        for cls, layer in (
            (
                TorchGramMuonAdamWPack if algorithm == 'gram' else optim.MuonAdamWPack,
                original,
            ),
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


@pytest.mark.parametrize('shape', [(1, 17, 33), (4, 65, 17), (64, 32, 48), (4, 33, 33)])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('steps', [0, 1, 2, 3, 5])
@pytest.mark.parametrize('layout', ['slice', 'transpose'])
def test_gram_newton_schulz(shape, dtype, steps, layout):
    torch.manual_seed(42)
    count, rows, columns = shape
    in_features = torch.randint(1, columns + 1, (count,)).tolist()
    out_features = torch.randint(1, rows + 1, (count,)).tolist()
    module = nn.LinearPack(
        in_features=in_features,
        out_features=out_features,
        max_in_features=columns,
        max_out_features=rows,
        pack_size=count,
        device='cuda',
    )
    if layout == 'slice':
        update = torch.randn(count, rows, columns * 2, device='cuda', dtype=dtype)[
            ..., ::2
        ]
    else:
        update = torch.randn(count, columns, rows, device='cuda', dtype=dtype).mT
    mask = (
        torch.arange(rows, device='cuda')[None, :, None]
        < torch.tensor(out_features, device='cuda')[:, None, None]
    ) & (
        torch.arange(columns, device='cuda')[None, None, :]
        < torch.tensor(in_features, device='cuda')[:, None, None]
    )
    update.masked_fill_(~mask, 0)
    if count > 1:
        update[0].zero_()
    before = update.clone()
    plan = GramNewtonSchulz(module, module.weight.device, (32, 32, 32, 4, 2))
    expected = torch_gram_newton_schulz(update, steps)
    actual = plan(update, steps)
    assert actual.dtype == expected.dtype == torch.float16
    torch.testing.assert_close(
        actual, expected, rtol=0.01, atol=0.02 if steps == 5 else 0.002
    )
    error = (actual.float() - expected.float()).flatten(1).norm(dim=1)
    reference_norm = expected.float().flatten(1).norm(dim=1).clamp_min(1e-7)
    assert (error / reference_norm).max() < 0.03
    torch.testing.assert_close(update, before, rtol=0, atol=0)
    assert not actual.masked_select(~mask).any()
    if count > 1:
        assert not actual[0].any()


@pytest.mark.parametrize('linear_type', COO_TYPES)
@pytest.mark.parametrize(
    'mutation', ['buffers', 'state_dict', 'replace', 'remove', 'select', 'inference']
)
def test_coo_layout_changes(linear_type, mutation):
    torch.manual_seed(42)
    kwargs = dict(
        in_features=[3, 8], out_features=[2, 9], pack_size=2, device='cuda'
    )
    original = nn.LinearPack(**kwargs)
    custom = linear_type(**kwargs)
    custom.load_state_dict(original.state_dict())

    def check():
        source = torch.randn(
            custom.pack_size, 7, 8, device='cuda'
        )
        source *= (
            torch.arange(8, device='cuda')[None, None, :]
            < custom.in_features[:, None, None]
        )
        with torch.autocast('cuda', dtype=torch.bfloat16):
            expected = original(source)
            actual = custom(source)
        torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.02)
        return actual

    check()
    idx = torch.tensor([0], device='cuda')
    if mutation == 'buffers':
        for module in (original, custom):
            module.in_features = nn.BufferPack(module.in_features.flip(0))
            module.out_features = nn.BufferPack(module.out_features.flip(0))
    elif mutation == 'state_dict':
        state = original.state_dict()
        state['in_features'] = state['in_features'].flip(0)
        state['out_features'] = state['out_features'].flip(0)
        for module in (original, custom):
            module.load_state_dict(state)
    elif mutation == 'replace':
        for module in (original, custom):
            module.replace(pack_idx=idx, in_features=[7], out_features=[8])
        custom.load_state_dict(original.state_dict())
    elif mutation == 'remove':
        for module in (original, custom):
            module.remove(pack_idx=idx)
    elif mutation == 'select':
        original.eval()
        custom.eval()
        idx = torch.tensor([1, 0], device='cuda')
        with original.temporarily_select(pack_idx=idx):
            with custom.temporarily_select(pack_idx=idx):
                check()
    else:
        custom._invalidate_prefix_sums()
        with torch.inference_mode():
            check()
    actual = check()
    actual.sum().backward()
    assert custom.weight.grad is not None
    assert len(custom._coo_cache) == 1


@pytest.mark.parametrize('linear_type', COO_TYPES)
@pytest.mark.parametrize('field,value', [('in_features', 9), ('out_features', 0)])
def test_coo_rejects_invalid_sizes(linear_type, field, value):
    module = linear_type(in_features=[3, 8], out_features=[2, 9], pack_size=2)
    getattr(module, field)[0] = value
    with pytest.raises(ValueError, match='feature counts'):
        module._coo_specs(7)


@pytest.mark.parametrize('linear_type', COO_TYPES)
def test_coo_no_bias_strided_input(linear_type):
    torch.manual_seed(42)
    kwargs = dict(
        in_features=17, out_features=33, bias=False, pack_size=4, device='cuda'
    )
    original = nn.LinearPack(**kwargs)
    custom = linear_type(**kwargs)
    custom.load_state_dict(original.state_dict())
    source = torch.randn(4, 17, 7, device='cuda').mT
    x = source.detach().requires_grad_()
    y = source.detach().requires_grad_()
    expected = original(x)
    actual = custom(y)
    expected.sum().backward()
    actual.sum().backward()
    for value, reference in (
        (actual, expected),
        (y.grad, x.grad),
        (custom.weight.grad, original.weight.grad),
    ):
        torch.testing.assert_close(value, reference, rtol=1e-4, atol=2e-4)


@pytest.mark.parametrize('swizzle', ['row', 'z', 'hilbert'])
def test_coo_tile_coverage(swizzle):
    rows, columns = [1, 65, 4096], [129, 1, 1]
    coordinates = _tile_coordinates(rows, columns, 64, 64, swizzle, torch.device('cpu'))
    actual = list(zip(*(t.tolist() for t in coordinates)))
    expected = [
        (matrix, row, column)
        for matrix, (m, n) in enumerate(zip(rows, columns))
        for row in range((m + 63) // 64)
        for column in range((n + 63) // 64)
    ]
    assert sorted(actual) == sorted(expected)
    if swizzle == 'hilbert':
        coordinates = _tile_coordinates([2], [2], 1, 1, swizzle, torch.device('cpu'))
        assert list(zip(*(t.tolist() for t in coordinates))) == [
            (0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)
        ]


@pytest.mark.parametrize('linear_type', [TritonGroupedLinearPack, *COO_TYPES])
@pytest.mark.parametrize('amp_dtype', [None, torch.bfloat16, torch.float16])
def test_linear_interleaved_embeddings(linear_type, amp_dtype):
    torch.manual_seed(42)
    embeddings = [
        nn.CosineEmbeddingsPack(
            n_features=2, d_embedding=[2, 4], init_scale=1.0, pack_size=2
        ).cuda()
        for _ in range(2)
    ]
    layers = [
        cls(in_features=[4, 8], out_features=1, pack_size=2, device='cuda')
        for cls in (nn.LinearPack, linear_type)
    ]
    with torch.no_grad():
        for embedding, layer in zip(embeddings, layers):
            embedding.weight.zero_()
            embedding.bias.zero_()
            embedding.elementwise_affine_weight.fill_(1)
            embedding.elementwise_affine_bias.zero_()
            layer.weight.fill_(1)
            layer.bias.zero_()
    inputs = [torch.ones(2, 3, 2, device='cuda', requires_grad=True) for _ in layers]
    context = (
        contextlib.nullcontext()
        if amp_dtype is None
        else torch.autocast('cuda', dtype=amp_dtype)
    )
    with context:
        outputs = [
            layer(embedding(source).flatten(-2))
            for layer, embedding, source in zip(layers, embeddings, inputs)
        ]
    torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0)
    torch.testing.assert_close(outputs[1][0], torch.full_like(outputs[1][0], 4))
    gradient = torch.randn_like(outputs[0])
    for output in outputs:
        output.backward(gradient)
    pairs = [
        (inputs[1].grad, inputs[0].grad),
        (layers[1].weight.grad, layers[0].weight.grad),
        (layers[1].bias.grad, layers[0].bias.grad),
        *[
            (actual.grad, expected.grad)
            for actual, expected in zip(
                embeddings[1].parameters(), embeddings[0].parameters()
            )
        ],
    ]
    for actual, expected in pairs:
        torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.02)
