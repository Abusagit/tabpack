"""Focused PyTorch allocated-memory checks, separate from performance timing."""

import argparse
import collections
import contextlib
import gc
import json
from pathlib import Path

import torch


def storage_bytes(values):
    return sum(t.numel() * t.element_size() for t in values)


def model_memory(args):
    from tabpack_experimental.benchmark import Config, make_runner
    from tabpack_experimental.ragged_linear import install_ragged_linears
    from tabpack_experimental.ragged_muon import install_ragged_muon

    raw = json.loads(args.config.read_text())
    for key in ('widths', 'cat_cardinalities'):
        raw[key] = tuple(raw[key])
    runner = make_runner(Config(**raw))
    for _ in range(10):
        runner.step()
    muon = None
    if args.variant in ('ragged_linear', 'ragged_both'):
        install_ragged_linears(runner.model, scope='all', schedule='k_desc', tiled=True)
    if args.variant in ('ragged_muon', 'ragged_both'):
        muon = install_ragged_muon(runner.optimizer, runner.model)
    if args.variant == 'compile_both':
        runner.forward = torch.compile(runner.forward)
        runner.optimizer.step = torch.compile(runner.optimizer.step)
    for _ in range(5):
        runner.step()
    runner.check_finite()
    runner.sync()
    gc.collect()
    torch.cuda.empty_cache()
    samples = []
    for _ in range(3):
        runner.sync()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        runner.step()
        runner.sync()
        samples.append(
            {
                'before_allocated_bytes': baseline,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'after_allocated_bytes': torch.cuda.memory_allocated(),
                'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            }
        )
    phases = {}

    @contextlib.contextmanager
    def scope(name):
        runner.sync()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        yield
        runner.sync()
        phases[name] = {
            'before_allocated_bytes': baseline,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'after_allocated_bytes': torch.cuda.memory_allocated(),
        }

    runner.step(scope)
    state = collections.defaultdict(int)
    for values in runner.optimizer.state.values():
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                state[key] += value.numel() * value.element_size()
    return {
        'variant': args.variant,
        'torch': torch.__version__,
        'gpu': torch.cuda.get_device_name(),
        'parameter_bytes': storage_bytes(runner.model.parameters()),
        'gradient_bytes': storage_bytes(
            p.grad for p in runner.model.parameters() if p.grad is not None
        ),
        'optimizer_state_bytes_by_key': dict(state),
        'muon_workspace_bytes': sum(
            storage_bytes((plan.x, plan.y, plan.a, plan.b, plan.partials, plan.norms))
            for plan in muon.plans.values()
        )
        if muon
        else 0,
        'samples': samples,
        'phases': phases,
        'scope': (
            'Fresh process containing only this implementation. Ten native updates '
            'initialize state, then five variant updates warm it. Historical free '
            'allocator blocks released before three consecutive complete steps. '
            'Previous-step gradients remain live until the original loop clears '
            'them after forward/loss. GPU context/library allocations outside the '
            'PyTorch allocator and other users are excluded.'
        ),
    }


def foreach_memory(args):
    from tabpack_experimental.ragged import LinearPlan

    widths, batch = (64,) * 99 + (1024,), 256
    xs = [torch.randn(batch, h, device='cuda', dtype=torch.bfloat16) for h in widths]
    ws = [torch.randn(h, h, device='cuda', dtype=torch.bfloat16) for h in widths]
    dys = [torch.randn_like(x) for x in xs]
    x, dy = torch.cat(xs, dim=1), torch.cat(dys, dim=1)
    w = torch.cat([value.flatten() for value in ws])
    xv, dyv = x.split(widths, dim=1), dy.split(widths, dim=1)
    result = {
        'torch': torch.__version__,
        'gpu': torch.cuda.get_device_name(),
        'compact_input_activation_bytes': x.numel() * x.element_size(),
        'compact_weight_bytes': w.numel() * w.element_size(),
        'cases': {},
        'scope': (
            'Warmed eager pure GEMMs; additional PyTorch allocations above all '
            'prepared inputs/metadata/library workspaces. Outputs stay alive until '
            'peak is read. Lists and compact copies coexist only as benchmark '
            'inputs, so their common baseline is not a standalone input footprint.'
        ),
    }
    for mode in ('forward', 'input_grad', 'weight_grad'):
        if mode == 'forward':
            a, b, av, bv, ca, cb = xs, [v.mT for v in ws], xv, [v.mT for v in ws], x, w
        elif mode == 'input_grad':
            a, b, av, bv, ca, cb = dys, ws, dyv, ws, dy, w
        else:
            a, b, av, bv, ca, cb = (
                [v.mT for v in dys],
                xs,
                [v.mT for v in dyv],
                xv,
                dy,
                x,
            )
        plan = LinearPlan([(batch, h, h) for h in widths], mode, schedule='k_desc')

        def packed(av=av, bv=bv, mode=mode):
            outputs = torch.foreach.mm(av, bv)
            return (
                torch.cat([v.flatten() for v in outputs])
                if mode == 'weight_grad'
                else torch.cat(outputs, dim=1)
            )

        functions = {
            'foreach_contiguous_lists': lambda a=a, b=b: torch.foreach.mm(a, b),
            'foreach_views_to_packed': packed,
            'ragged_default': lambda plan=plan, ca=ca, cb=cb: plan.run(ca, cb),
        }
        measurements = {}
        for name, function in functions.items():
            for _ in range(3):
                outputs = function()
                torch.cuda.synchronize()
                del outputs
            gc.collect()
            samples = []
            for _ in range(3):
                torch.cuda.synchronize()
                baseline = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                outputs = function()
                torch.cuda.synchronize()
                samples.append(
                    {
                        'additional_peak_bytes': torch.cuda.max_memory_allocated()
                        - baseline,
                        'retained_output_bytes': torch.cuda.memory_allocated()
                        - baseline,
                    }
                )
                del outputs
            measurements[name] = samples
        result['cases'][mode] = measurements
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--variant',
        required=True,
        choices=(
            'native',
            'compile_both',
            'ragged_linear',
            'ragged_muon',
            'ragged_both',
            'foreach',
        ),
    )
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    result = foreach_memory(args) if args.variant == 'foreach' else model_memory(args)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
