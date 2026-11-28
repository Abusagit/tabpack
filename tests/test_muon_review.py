"""Regression checks for the grouped Muon review findings."""

import copy

import pytest
import torch

from tabpackage import nn, optim
from tabpackage._muon_grouped import GroupedNewtonSchulz
from tabpackage._training import make_optimizer_pack


@pytest.mark.parametrize('backend', [None, 'padded'])
def test_padded_backend_preserves_subclass_hook(backend):
    class LegacyMuon(optim.MuonAdamWPack):
        calls = 0

        @staticmethod
        def make_muon_parameter_group(module):
            LegacyMuon.calls += 1
            return optim.MuonAdamWPack.make_muon_parameter_group(module)

    model = nn.TabPack(
        n_numerical_features=8,
        categorical_cardinalities=[],
        d_out=1,
        n_blocks=2,
        d_block=[8, 16],
        dropout=0.0,
        pack_size=2,
    )
    opt = make_optimizer_pack(
        model,
        type=LegacyMuon,
        adamw_lr=0.001,
        muon_lr=0.01,
        weight_decay=0.1,
        **({} if backend is None else {'muon_backend': backend}),
    )
    assert LegacyMuon.calls == 2
    x = torch.randn(2, 4, 8)
    model(x, None).sum().backward()
    opt.step()


@pytest.mark.parametrize('inplace', [False, True])
@pytest.mark.parametrize('nesterov', [False, True])
@pytest.mark.parametrize('initialized', [False, True])
@pytest.mark.parametrize('bad_source', ['grad', 'momentum'])
def test_padding_failure_preserves_entire_optimizer(
    inplace, nesterov, initialized, bad_source
):
    if bad_source == 'momentum' and not initialized:
        pytest.skip('no momentum exists before the first step')
    layers = [nn.LinearPack(9, [4, 8], bias=False, pack_size=2) for _ in range(3)]
    groups = [{'params': [layers[0].weight], 'muon': False}]
    groups += [
        optim.MuonAdamWPack.make_muon_parameter_group(
            layer,
            grouped_rectangular=True,
        )
        for layer in layers[1:]
    ]
    opt = optim.MuonAdamWPack(
        groups,
        adamw_lr=0.001,
        muon_lr=0.01,
        weight_decay=0.1,
        pack_size=2,
        allow_inplace_grad_mutation=inplace,
        muon_nesterov=nesterov,
        muon_backend='grouped',
    )
    for layer in layers:
        layer.weight.grad = torch.ones_like(layer.weight)
        layer.weight.grad[0, 4:] = 0
    if initialized:
        # Seed a valid state using native execution without needing a CUDA build.
        opt._muon_backend = 'padded'
        opt.step()
        opt._muon_backend = 'grouped'
    if bad_source == 'grad':
        layers[-1].weight.grad[0, -1, 0] = 1
    else:
        opt.state[layers[-1].weight]['muon_momentum_buffer'][0, -1, 0] = 1
    weights = [p.weight.detach().clone() for p in layers]
    grads = [p.weight.grad.clone() for p in layers]
    state = copy.deepcopy(opt.state_dict())
    with pytest.raises(ValueError, match='zero update padding'):
        opt.step()
    for layer, weight, grad in zip(layers, weights, grads):
        torch.testing.assert_close(layer.weight, weight, rtol=0, atol=0)
        torch.testing.assert_close(layer.weight.grad, grad, rtol=0, atol=0)
    after = opt.state_dict()
    assert state['state'].keys() == after['state'].keys()
    for key in state['state']:
        for name, value in state['state'][key].items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(
                    after['state'][key][name], value, rtol=0, atol=0
                )
            else:
                assert after['state'][key][name] == value


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')


@CUDA
@pytest.mark.parametrize('shape', [(17, 97), (17, 17), (16, 98)])
@pytest.mark.parametrize('strategy', ['grouped', 'lt'])
def test_odd_shapes_and_reused_active_sets(monkeypatch, shape, strategy):
    monkeypatch.setenv('TABPACK_MUON_BX_BACKEND', strategy)
    rows, cols = shape
    widths = torch.tensor([4, 8, rows], device='cuda')
    rectangular = rows != cols
    runtime = GroupedNewtonSchulz()
    plan = None
    for members in ([0, 1], [1, 2], [0, 2], None, [1, 2]):
        x = torch.zeros(3, rows, cols, device='cuda')
        for i in range(3) if members is None else members:
            width = int(widths[i])
            x[i, :width, : cols if rectangular else width] = torch.randn(
                width,
                cols if rectangular else width,
                device='cuda',
            )
        active = None if members is None else torch.tensor(members, device='cuda')
        actual = runtime(
            x, widths, active_pack_idx=active, rectangular=rectangular
        ).clone()
        torch.testing.assert_close(
            actual,
            optim._zeropower_via_newtonschulz5(x, 5),
            rtol=0.05,
            atol=0.02,
        )
        if plan is not None:
            assert runtime.plan is plan
        plan = runtime.plan
        for i in set(range(3)) - set(range(3) if members is None else members):
            assert actual[i].count_nonzero() == 0
