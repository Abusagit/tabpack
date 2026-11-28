"""GPU correctness checks for row-trimmed rectangular Newton-Schulz."""

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch

from tabpackage import nn, optim
from tabpackage._muon_grouped import GroupedNewtonSchulz


def error(actual, expected):
    d = actual.detach().float() - expected.detach().float()
    return {
        'max_abs': float(d.abs().max()),
        'relative_l2': float(
            d.norm() / expected.detach().float().norm().clamp_min(1e-12)
        ),
    }


def main():
    out = Path('../output')
    out.mkdir(exist_ok=True)
    report = {
        'torch': torch.__version__,
        'gpu': torch.cuda.get_device_name(),
        'ns': [],
        'tolerance': {'rtol': 0.05, 'atol': 0.02, 'relative_l2_max': 0.03},
        'optimizers': [],
    }
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        for m, n, rectangular in (
            (16, 97, True),
            (17, 97, True),
            (17, 17, False),
            (16, 98, True),
            (32, 31506, True),
            (512, 31506, True),
            (512, 20061, True),
            (32, 32, True),
            (32, 32, False),
        ):
            widths = torch.tensor([64 if m == 512 else 4, m // 2, m], device='cuda')
            x = torch.zeros(3, m, n, device='cuda')
            for i, w in enumerate(widths.tolist()):
                x[i, :w, : n if rectangular else w] = torch.randn(
                    w, n if rectangular else w, device='cuda'
                )
            x[:, :, 1::3] = 0  # non-contiguous zero columns, like masked embeddings
            x[0] = 0
            runtime = GroupedNewtonSchulz()
            for active in (None, torch.tensor([1, 2], device='cuda')):
                result = runtime(
                    x, widths, active_pack_idx=active, rectangular=rectangular
                )
                assert result.shape == x.shape
                if n == 20061:
                    assert runtime.plan.workspace_columns == 20064
                    assert not result.is_contiguous()
                if n == 31506:
                    assert runtime.plan.workspace_columns == 31512
                if m == 512 and torch.cuda.get_device_properties(0).major >= 8:
                    assert runtime.plan.wide_backend == 'lt'
                y = result.clone()
                expected = optim._zeropower_via_newtonschulz5(x, 5)
                assert bool(y.isfinite().all())
                assert error(y, expected)['relative_l2'] <= 0.03
                assert y[0].count_nonzero() == 0
                torch.testing.assert_close(y, expected, rtol=0.05, atol=0.02)
                for i, w in enumerate(widths.tolist()):
                    assert y[i, w:].count_nonzero() == 0
                assert y[:, :, 1::3].count_nonzero() == 0
                report['ns'].append(
                    dict(
                        seed=seed,
                        shape=list(x.shape),
                        rectangular=rectangular,
                        workspace_columns=runtime.plan.workspace_columns,
                        wide_backend=runtime.plan.wide_backend,
                        **error(y, expected),
                    )
                )
            bad = x.clone()
            bad[0, -1, 0] = 1
            try:
                runtime(bad, widths, rectangular=rectangular)
            except ValueError:
                pass
            else:
                raise AssertionError('nonzero row padding accepted')
            widths[1] = m // 4
            x[1, m // 4 :] = 0
            if not rectangular:
                x[1, :, m // 4 :] = 0
            y = runtime(x, widths, rectangular=rectangular).clone()
            torch.testing.assert_close(
                y, optim._zeropower_via_newtonschulz5(x, 5), rtol=0.05, atol=0.02
            )
            if rectangular and n % 2 and seed == 0:
                # A non-finite call must not poison internal zero columns forever.
                poisoned = x.clone()
                poisoned[1, 0, 0] = float('nan')
                runtime(poisoned, widths, rectangular=True)
                recovered = runtime(x, widths, rectangular=True).clone()
                torch.testing.assert_close(
                    recovered,
                    optim._zeropower_via_newtonschulz5(x, 5),
                    rtol=0.05,
                    atol=0.02,
                )
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                y = runtime(x, widths, rectangular=rectangular).clone()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                y, optim._zeropower_via_newtonschulz5(x, 5), rtol=0.05, atol=0.02
            )
    for skip in (False, True):
        torch.manual_seed(0)
        layer = nn.LinearPack(97, [4, 8, 16], bias=False, pack_size=3, device='cuda')
        twin = copy.deepcopy(layer)
        layers = {'padded': layer, 'grouped': twin}
        opts = {
            name: optim.MuonAdamWPack(
                [
                    optim.MuonAdamWPack.make_muon_parameter_group(
                        mod, grouped_rectangular=True
                    )
                ],
                adamw_lr=0.001,
                muon_lr=0.01,
                weight_decay=0.1,
                muon_skip_zero_updates=skip,
                pack_size=3,
                allow_inplace_grad_mutation=False,
                support_replacement=True,
                muon_backend=name,
            )
            for name, mod in layers.items()
        }
        for step in range(5):
            grad = torch.zeros_like(layer.weight)
            for i, w in enumerate([4, 8, 16]):
                if i:
                    grad[i, :w] = torch.randn(w, 97, device='cuda')
            for name, mod in layers.items():
                mod.weight.grad = grad.clone()
                opts[name].step()
            assert torch.equal(
                opts['padded'].state[layer.weight]['muon_momentum_buffer'],
                opts['grouped'].state[twin.weight]['muon_momentum_buffer'],
            )
            torch.testing.assert_close(twin.weight, layer.weight, rtol=0.05, atol=0.02)
            assert torch.equal(twin.weight[0, 4:], layer.weight[0, 4:])
        five_step_error = error(twin.weight, layer.weight)
        idx = torch.tensor([1], device='cuda')
        for name, mod in layers.items():
            mod.replace(pack_idx=idx, out_features=[4])
            opts[name].replace(pack_idx=idx, module=mod)
        assert opts['grouped'].param_groups[0]['muon_grouped_widths'][1] == 4
        twin.weight.data.copy_(layer.weight)
        paired_grad = torch.randn(2, 16, 97, device='cuda')
        paired_grad[0, 4:] = 0
        for name, mod in layers.items():
            mapping = mod.remove(pack_idx=idx)
            optim.optimizer_pack_remove(
                opts[name], pack_idx=idx, old_to_new_parameters=mapping
            )
            mod.weight.grad = paired_grad.clone()
            opts[name].step()
        assert len(opts['grouped']._grouped_runtimes) == 1
        torch.testing.assert_close(twin.weight, layer.weight, rtol=0.05, atol=0.02)
        report['optimizers'].append(
            {
                'skip': skip,
                'five_step_error': five_step_error,
                'post_removal_error': error(twin.weight, layer.weight),
                'lifecycle_passed': True,
            }
        )
    tall = nn.LinearPack(8, [16, 32], pack_size=2, device='cuda')
    assert (
        optim.MuonAdamWPack.make_muon_parameter_group(tall, grouped_rectangular=True)[
            'muon_grouped_widths'
        ]
        is None
    )
    report['tall_fallback'] = True
    (out / 'correctness.json').write_text(json.dumps(report, indent=2))
    print('Rectangular/square NS and optimizer lifecycle checks passed', flush=True)


if __name__ == '__main__':
    main()
