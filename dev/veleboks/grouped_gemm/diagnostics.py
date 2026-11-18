"""Numerical reference, preparation breakdown and lifecycle regressions."""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from experiment import differences, summary, timed

from tabpackage._muon_grouped import GroupedNewtonSchulz, extension
from tabpackage.optim import _zeropower_via_newtonschulz5

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-run', required=True)
args = parser.parse_args()
os.environ['PATH'] = str(Path.home() / 'tabpack/.venv/bin') + ':' + os.environ['PATH']
torch.set_num_threads(4)
source = json.loads(
    (
        Path.home() / 'tabpack-runs' / args.source_run / 'output/experiment.json'
    ).read_text()
)
sampled = source['sampled_models']
report = {
    'source_run': args.source_run,
    'gpu': torch.cuda.get_device_name(),
    'uuid': str(torch.cuda.get_device_properties(0).uuid),
    'torch': torch.__version__,
    'cuda': torch.version.cuda,
}
out = Path('../output')
out.mkdir(exist_ok=True)
start = time.perf_counter()
extension()
report['extension_load_build_seconds'] = time.perf_counter() - start

# Higher-precision reference starts with the identical rounded normalization.
report['reference_checks'] = []
widths = torch.tensor([32, 64, 128, 256, 384, 512], device='cuda')
for seed in (42, 43, 44):
    torch.manual_seed(seed)
    g = torch.zeros(6, 512, 512, device='cuda')
    for i, w in enumerate(widths.tolist()):
        g[i, :w, :w] = torch.randn(w, w, device='cuda')
    runtime = GroupedNewtonSchulz()
    got = runtime(g, widths).clone()
    padded = _zeropower_via_newtonschulz5(g, 5)
    x = g.bfloat16()
    normalized = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    assert torch.equal(runtime(g, widths, steps=0), normalized)
    for i, w in enumerate(widths.tolist()):
        ref = normalized[i, :w, :w].double()
        for _ in range(5):
            a = ref @ ref.mT
            b = -4.7750 * a + 2.0315 * (a @ a)
            ref = 3.4445 * ref + b @ ref
        # A separate native invocation on an actual-width, contiguous matrix
        # helps distinguish shape-dependent cuBLAS rounding from masking errors.
        standalone = _zeropower_via_newtonschulz5(g[i : i + 1, :w, :w].contiguous(), 5)[
            0
        ]
        report['reference_checks'].append(
            {
                'seed': seed,
                'width': w,
                'grouped_vs_padded': differences(got[i, :w, :w], padded[i, :w, :w]),
                'grouped_vs_standalone': differences(got[i, :w, :w], standalone),
                'padded_vs_fp64': differences(padded[i, :w, :w], ref),
                'grouped_vs_fp64': differences(got[i, :w, :w], ref),
            }
        )
    # Mutating widths in place must invalidate the plan even if shape is unchanged.
    widths_copy = widths.clone()
    zero = torch.zeros_like(g)
    runtime(zero, widths_copy)
    previous_plan = runtime.plan
    widths_copy[0] = 64
    assert torch.count_nonzero(runtime(zero, widths_copy)) == 0
    assert runtime.plan is not previous_plan

widths = torch.tensor([s['width'] for s in sampled], device='cuda')
g = torch.zeros(len(sampled), 512, 512, device='cuda')
for i, member in enumerate(sampled):
    if member['depth'] > 1:
        w = member['width']
        g[i, :w, :w] = torch.randn(w, w, device='cuda')
runtime = GroupedNewtonSchulz()
runtime(g, widths)
x = g.bfloat16()
normalized = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)


def guard():
    assert not torch.any((g != 0) & runtime.padding_mask).item()


def normalize():
    x = g.bfloat16()
    return x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)


def padded_iterations():
    x = normalized
    for _ in range(5):
        a = x @ x.mT
        b = torch.baddbmm(a, a, a, beta=-4.7750, alpha=2.0315)
        x = torch.baddbmm(x, b, x, beta=3.4445)
    return x


functions = {
    'padding_guard': guard,
    'normalization': normalize,
    'grouped_iterations_including_workspace_copies': lambda: runtime.plan.run(
        normalized, 5
    ),
    'padded_iterations': padded_iterations,
}
report['breakdown'] = {}
for name, fn in functions.items():
    for _ in range(8):
        fn()
    samples = [timed(fn, 3)[0] for _ in range(15)]
    report['breakdown'][name] = summary(samples)
# Repeated optimizer state loads must keep the number of workspaces bounded.
from tabpackage import nn, optim

layer = nn.LinearPack([32, 64], [32, 64], bias=False, pack_size=2, device='cuda')
opt = optim.MuonAdamWPack(
    [optim.MuonAdamWPack.make_muon_parameter_group(layer, grouped_square=True)],
    adamw_lr=0.001,
    muon_lr=0.01,
    weight_decay=0.01,
    pack_size=2,
    allow_inplace_grad_mutation=False,
    muon_backend='grouped',
)
grad = torch.zeros_like(layer.weight)
grad[0, :32, :32] = torch.randn(32, 32, device='cuda')
grad[1] = torch.randn(64, 64, device='cuda')
layer.weight.grad = grad
opt.step()
state = opt.state_dict()
for _ in range(20):
    opt.load_state_dict(state)
    opt.step()
    assert len(opt._grouped_runtimes) == 1
report['metadata_mutation_and_repeated_state_load_passed'] = True
setup_samples = []
for _ in range(15):
    torch.cuda.synchronize()
    start = time.perf_counter()
    plan = extension().Plan(g, widths.tolist(), list(range(len(widths))))
    torch.cuda.synchronize()
    setup_samples.append((time.perf_counter() - start) * 1000)
report['metadata_and_workspace_setup_wall_ms'] = summary(setup_samples)
report['cublas_version'] = runtime.plan.version
(out / 'diagnostics.json').write_text(json.dumps(report, indent=2) + '\n')
print('Diagnostics complete', report['breakdown'], flush=True)
