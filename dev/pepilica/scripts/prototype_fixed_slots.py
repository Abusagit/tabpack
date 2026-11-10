"""Fixed-slot ragged GEMM prototype and forward benchmark."""

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

from tabpack_experimental.compact_mlp import width_tiles
from tabpack_experimental.ragged import LinearPlan


@triton.jit
def fixed_slot_mm(
    X,
    W,
    B,
    Y,
    D,
    T,
    STRIDE: tl.constexpr,
    BATCH: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    task = tl.program_id(0)
    member = tl.load(T + task * 3)
    mi = tl.load(T + task * 3 + 1)
    ni = tl.load(T + task * 3 + 2)
    d = D + member * 3
    col = tl.load(d)
    wo = tl.load(d + 1)
    width = tl.load(d + 2)
    rows = mi * BM + tl.arange(0, BM)
    cols = ni * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(width, BK)):
        k = start * BK + kk
        x = tl.load(
            X + rows[:, None] * STRIDE + col + k[None, :],
            (rows[:, None] < BATCH) & (k[None, :] < width),
            0,
        )
        w = tl.load(
            W + wo + cols[None, :] * width + k[:, None],
            (cols[None, :] < width) & (k[:, None] < width),
            0,
        )
        acc += tl.dot(x, w)
    bias = tl.load(B + col + cols, cols < width, 0)
    tl.store(
        Y + rows[:, None] * STRIDE + col + cols[None, :],
        acc + bias[None, :],
        (rows[:, None] < BATCH) & (cols[None, :] < width),
    )


@triton.jit
def fixed_slot_dx(
    DY,
    W,
    DX,
    D,
    T,
    STRIDE: tl.constexpr,
    BATCH: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    task = tl.program_id(0)
    member, mi, ni = (
        tl.load(T + task * 3),
        tl.load(T + task * 3 + 1),
        tl.load(T + task * 3 + 2),
    )
    col, wo, width = (
        tl.load(D + member * 3),
        tl.load(D + member * 3 + 1),
        tl.load(D + member * 3 + 2),
    )
    rows = mi * BM + tl.arange(0, BM)
    cols = ni * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(width, BK)):
        k = start * BK + kk
        dy = tl.load(
            DY + rows[:, None] * STRIDE + col + k[None, :],
            (rows[:, None] < BATCH) & (k[None, :] < width),
            0,
        )
        w = tl.load(
            W + wo + k[:, None] * width + cols[None, :],
            (k[:, None] < width) & (cols[None, :] < width),
            0,
        )
        acc += tl.dot(dy, w)
    tl.store(
        DX + rows[:, None] * STRIDE + col + cols[None, :],
        acc,
        (rows[:, None] < BATCH) & (cols[None, :] < width),
    )


@triton.jit
def fixed_slot_dw(
    DY,
    X,
    DW,
    D,
    T,
    STRIDE: tl.constexpr,
    BATCH: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    task = tl.program_id(0)
    member, mi, ni = (
        tl.load(T + task * 3),
        tl.load(T + task * 3 + 1),
        tl.load(T + task * 3 + 2),
    )
    col, wo, width = (
        tl.load(D + member * 3),
        tl.load(D + member * 3 + 1),
        tl.load(D + member * 3 + 2),
    )
    rows = mi * BM + tl.arange(0, BM)
    cols = ni * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(BATCH, BK)):
        k = start * BK + kk
        dy = tl.load(
            DY + k[None, :] * STRIDE + col + rows[:, None],
            (k[None, :] < BATCH) & (rows[:, None] < width),
            0,
        )
        x = tl.load(
            X + k[:, None] * STRIDE + col + cols[None, :],
            (k[:, None] < BATCH) & (cols[None, :] < width),
            0,
        )
        acc += tl.dot(dy, x)
    tl.store(
        DW + wo + rows[:, None] * width + cols[None, :],
        acc,
        (rows[:, None] < width) & (cols[None, :] < width),
    )


@triton.jit
def fixed_slot_db(
    DY,
    DB,
    D,
    T,
    STRIDE: tl.constexpr,
    BATCH: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    task = tl.program_id(0)
    member, ni = tl.load(T + task * 3), tl.load(T + task * 3 + 2)
    col, width = tl.load(D + member * 3), tl.load(D + member * 3 + 2)
    cols = ni * BN + tl.arange(0, BN)
    rows = tl.arange(0, BM)
    acc = tl.full((BN,), 0, tl.float32)
    for start in range(tl.cdiv(BATCH, BM)):
        row = start * BM + rows
        values = tl.load(
            DY + row[:, None] * STRIDE + col + cols[None, :],
            (row[:, None] < BATCH) & (cols[None, :] < width),
            0,
        )
        acc += tl.sum(values.to(tl.float32), axis=0)
    tl.store(DB + col + cols, acc, cols < width)


class FixedSlotPlan:
    """Prepared active-set plan. Retain it unchanged until backward completes."""

    def __init__(self, widths, batch, active, *, device='cuda'):
        self.widths = tuple(int(w) for w in widths)
        self.batch = int(batch)
        self.active = tuple(int(i) for i in active)
        if (
            not self.widths
            or min(self.widths) < 1
            or self.batch < 1
            or len(set(self.active)) != len(self.active)
            or any(i < 0 or i >= len(self.widths) for i in self.active)
        ):
            raise ValueError('Invalid widths, batch or active members')
        self.stride = sum(self.widths)
        self.weight_size = sum(w * w for w in self.widths)
        descriptors, forward_tasks, weight_tasks, bias_tasks = [], [], [], []
        col = weight = 0
        for width in self.widths:
            descriptors.append((col, weight, width))
            col += width
            weight += width * width
        for i in self.active:
            width = self.widths[i]
            forward_tasks.extend(
                (i, m, n)
                for m in range((self.batch + 31) // 32)
                for n in range((width + 31) // 32)
            )
            weight_tasks.extend(
                (i, m, n)
                for m in range((width + 31) // 32)
                for n in range((width + 31) // 32)
            )
            bias_tasks.extend((i, 0, n) for n in range((width + 31) // 32))
        forward_tasks.sort(key=lambda task: -self.widths[task[0]])
        self.descriptors = torch.tensor(descriptors, device=device, dtype=torch.int64)
        self.forward_tasks = torch.tensor(
            forward_tasks, device=device, dtype=torch.int32
        ).reshape(-1, 3)
        self.weight_tasks = torch.tensor(
            weight_tasks, device=device, dtype=torch.int32
        ).reshape(-1, 3)
        self.bias_tasks = torch.tensor(
            bias_tasks, device=device, dtype=torch.int32
        ).reshape(-1, 3)

    def __call__(self, x, weight, bias):
        if (
            x.device.type != 'cuda'
            or x.device != self.descriptors.device
            or weight.device != x.device
            or bias.device != x.device
            or x.dtype not in (torch.bfloat16, torch.float16)
            or weight.dtype != x.dtype
            or bias.dtype != x.dtype
            or x.shape != (self.batch, self.stride)
            or weight.shape != (self.weight_size,)
            or bias.shape != (self.stride,)
            or not all(t.is_contiguous() for t in (x, weight, bias))
        ):
            raise ValueError(
                'Expected contiguous fixed-slot CUDA inputs with matching FP16/BF16'
            )
        return _FixedSlotLinear.apply(x, weight, bias, self)


class _FixedSlotLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, plan):
        output = torch.zeros_like(x)
        if plan.active:
            kernel: Any = fixed_slot_mm
            kernel[(len(plan.forward_tasks),)](
                x,
                weight,
                bias,
                output,
                plan.descriptors,
                plan.forward_tasks,
                plan.stride,
                plan.batch,
                32,
                32,
                32,
                num_warps=4,
            )
        ctx.save_for_backward(x, weight)
        ctx.plan = plan
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        plan = ctx.plan
        dy = grad_output.contiguous()
        dx = torch.zeros_like(x) if ctx.needs_input_grad[0] else None
        dw = torch.zeros_like(weight) if ctx.needs_input_grad[1] else None
        db = (
            torch.zeros((plan.stride,), device=x.device, dtype=x.dtype)
            if ctx.needs_input_grad[2]
            else None
        )
        if plan.active:
            if dx is not None:
                kernel_dx: Any = fixed_slot_dx
                kernel_dx[(len(plan.forward_tasks),)](
                    dy,
                    weight,
                    dx,
                    plan.descriptors,
                    plan.forward_tasks,
                    plan.stride,
                    plan.batch,
                    32,
                    32,
                    32,
                    num_warps=4,
                )
            if dw is not None:
                kernel_dw: Any = fixed_slot_dw
                kernel_dw[(len(plan.weight_tasks),)](
                    dy,
                    x,
                    dw,
                    plan.descriptors,
                    plan.weight_tasks,
                    plan.stride,
                    plan.batch,
                    32,
                    32,
                    32,
                    num_warps=4,
                )
            if db is not None:
                kernel_db: Any = fixed_slot_db
                kernel_db[(len(plan.bias_tasks),)](
                    dy,
                    db,
                    plan.descriptors,
                    plan.bias_tasks,
                    plan.stride,
                    plan.batch,
                    128,
                    32,
                    num_warps=4,
                )
        return dx, dw, db, None


def timed(fn, repeats=7, steps=100):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    samples = []
    for _ in range(repeats):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        for _ in range(steps):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / steps)
    return {'median_us': statistics.median(samples), 'samples_us': samples}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_per_process_memory_fraction(0.25)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    widths = [32, 64, 128, 256, 384, 128, 64, 256, 32, 384, 128, 64]
    batch = 128
    stride = sum(widths)
    offsets, weight_offsets = [], []
    for i, width in enumerate(widths):
        offsets.append(sum(widths[:i]))
        weight_offsets.append(sum(w * w for w in widths[:i]))
    device = 'cuda'
    x = torch.randn(batch, stride, device=device, dtype=torch.bfloat16)
    weights = [
        torch.randn(w, w, device=device, dtype=torch.bfloat16) / w**0.5 for w in widths
    ]
    w = torch.cat([v.flatten() for v in weights])
    b = torch.randn(stride, device=device, dtype=torch.bfloat16)
    y = torch.empty_like(x)
    descriptors = torch.tensor(
        list(zip(offsets, weight_offsets, widths)), device=device, dtype=torch.int64
    )
    result = {
        'scope': 'Synthetic forward only; no backward, optimizer or integration',
        'gpu': torch.cuda.get_device_name(),
        'torch': torch.__version__,
        'triton': triton.__version__,
        'widths': widths,
        'batch': batch,
        'stride': stride,
        'cases': [],
    }
    for count in (12, 8, 4):
        active = list(range(count))
        start = time.perf_counter()
        tasks = [
            (i, m, n)
            for i in active
            for m in range((batch + 31) // 32)
            for n in range((widths[i] + 31) // 32)
        ]
        tasks.sort(key=lambda t: -widths[t[0]])
        t = torch.tensor(tasks, dtype=torch.int32, device=device)
        torch.cuda.synchronize()
        setup_fixed = time.perf_counter() - start

        def fixed(tasks=tasks, t=t):
            kernel: Any = fixed_slot_mm
            kernel[(len(tasks),)](
                x, w, b, y, descriptors, t, stride, batch, 32, 32, 32, num_warps=4
            )

        start = time.perf_counter()
        fixed()
        torch.cuda.synchronize()
        first_fixed = time.perf_counter() - start
        for i in active:
            ref = (
                x[:, offsets[i] : offsets[i] + widths[i]].float() @ weights[i].float().T
                + b[offsets[i] : offsets[i] + widths[i]].float()
            ).to(torch.bfloat16)
            out = y[:, offsets[i] : offsets[i] + widths[i]]
            rel = (
                (out.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-10)
            ).item()
            if rel > 0.01:
                raise AssertionError((i, rel))
        start = time.perf_counter()
        ws = widths[:count]
        old_x = torch.cat(
            [x[:, offsets[i] : offsets[i] + widths[i]] for i in active], dim=1
        )
        old_w = torch.cat([weights[i].flatten() for i in active])
        old_b = torch.cat([b[offsets[i] : offsets[i] + widths[i]] for i in active])
        plan = LinearPlan(
            [(batch, v, v) for v in ws],
            'forward',
            device=device,
            schedule='k_desc',
            tiles_by_width=width_tiles(ws, batch, 'forward'),
        )
        old_y = torch.empty((batch, sum(ws)), device=device, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        setup_old = time.perf_counter() - start

        def old(plan=plan, old_x=old_x, old_w=old_w, old_b=old_b, old_y=old_y):
            plan.run(old_x, old_w, old_b, out=old_y)

        start = time.perf_counter()
        old()
        torch.cuda.synchronize()
        first_old = time.perf_counter() - start
        cursor = 0
        for i in active:
            torch.testing.assert_close(
                old_y[:, cursor : cursor + widths[i]],
                y[:, offsets[i] : offsets[i] + widths[i]],
                rtol=0.02,
                atol=0.04,
            )
            cursor += widths[i]
        case = {
            'active_models': count,
            'fixed_tasks': len(tasks),
            'old_tasks': plan.n_tasks,
            'fixed_setup_s': setup_fixed,
            'old_setup_s_including_repacking': setup_old,
            'fixed_first_call_s': first_fixed,
            'old_first_call_s': first_old,
            'fixed_timing': timed(fixed),
            'old_timing': timed(old),
        }
        result['cases'].append(case)
        (args.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(case), flush=True)


if __name__ == '__main__':
    main()
