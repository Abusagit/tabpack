"""Experimental CUDA linear primitives with compact per-member storage.

Uses Triton's tiled dot primitive, with a precomputed tile-to-problem map.
References: https://triton-lang.org/main/getting-started/tutorials/
08-grouped-gemm.html and 03-matrix-multiplication.html.
"""

import dataclasses
import math
from typing import Any

import torch
import triton as _triton
import triton.language as _tl

# Triton annotations describe GPU compilation, not Python call signatures.
triton: Any = _triton
tl: Any = _tl


HIDDEN_SIZES = (16, 32, 64, 128, 256, 384, 512, 1024, 2048, 4096)


@triton.jit
def _store_output(
    Values,
    C,
    Offsets,
    Columns,
    ColumnBase,
    Valid,
    KeepProbabilities,
    Seed,
    FUSED: tl.constexpr,
    DROPOUT: tl.constexpr,
    RANDOM: tl.constexpr,
    SCALAR_PROBABILITY: tl.constexpr,
    Addend=None,
    ALPHA: tl.constexpr = 1.0,
    BETA: tl.constexpr = 0.0,
):
    if ALPHA != 1.0:
        Values = Values * ALPHA
    if BETA != 0.0:
        Values = Values + BETA * tl.load(Addend + Offsets, Valid, 0).to(tl.float32)
    if FUSED:
        values = Values.to(C.dtype.element_ty).to(tl.float32)
        values = tl.maximum(values, 0)
        if DROPOUT:
            if SCALAR_PROBABILITY:
                q = tl.load(KeepProbabilities + ColumnBase)
                inverse_q = 1.0 / tl.where(q > 0, q, 1)
            else:
                columns = tl.broadcast_to(Columns, Offsets.shape)
                q = tl.load(KeepProbabilities + columns, Valid, 1)
            keep = q > 0
            if RANDOM:
                seed = tl.load(Seed).to(tl.uint64)
                keep = keep & (tl.rand(seed, Offsets.to(tl.uint32)) < q)
            if SCALAR_PROBABILITY:
                values = tl.where(keep, values * inverse_q, 0)
            else:
                scale = tl.where(keep, 1.0 / tl.where(q > 0, q, 1), 0)
                values = tl.where(keep, values * scale, 0)
        tl.store(C + Offsets, values, Valid)
    else:
        tl.store(C + Offsets, Values, Valid)


@triton.jit
def _copy_member(
    A,
    C,
    Descriptors,
    Tasks,
    Task,
    X_ROW_STRIDE: tl.constexpr,
    Y_ROW_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    member = tl.load(Tasks + Task * 3)
    chunk = tl.load(Tasks + Task * 3 + 1)
    desc = Descriptors + member * 4
    xo, yo = tl.load(desc), tl.load(desc + 1)
    batch, width = tl.load(desc + 2), tl.load(desc + 3)
    index = chunk * BLOCK + tl.arange(0, BLOCK)
    row, column = index // width, index % width
    valid = index < batch * width
    value = tl.load(A + xo + row * X_ROW_STRIDE + column, valid, 0)
    tl.store(C + yo + row * Y_ROW_STRIDE + column, value, valid)


@triton.jit
def _specialized_tile(
    A,
    B,
    Bias,
    C,
    desc,
    tile_m,
    tile_n,
    KeepProbabilities,
    Seed,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    MODE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    BIAS: tl.constexpr,
    FUSED: tl.constexpr,
    DROPOUT: tl.constexpr,
    RANDOM: tl.constexpr,
    SCALAR_PROBABILITY: tl.constexpr,
    X_ROW_STRIDE: tl.constexpr,
    Y_ROW_STRIDE: tl.constexpr,
    Addend=None,
    ALPHA: tl.constexpr = 1.0,
    BETA: tl.constexpr = 0.0,
):
    ao = tl.multiple_of(tl.load(desc), 16)
    bo = tl.multiple_of(tl.load(desc + 1), 16)
    co = tl.multiple_of(tl.load(desc + 2), 16)
    if MODE == 'forward':
        am = X_ROW_STRIDE
        ak = 1
        bk = 1
        bn = K
        cm = Y_ROW_STRIDE
    elif MODE == 'input_grad':
        am = Y_ROW_STRIDE
        ak = 1
        bk = N
        bn = 1
        cm = X_ROW_STRIDE
    elif MODE == 'weight_grad':
        am = 1
        ak = Y_ROW_STRIDE
        bk = X_ROW_STRIDE
        bn = 1
        cm = N
    else:
        am = tl.load(desc + 6)
        ak = tl.load(desc + 7)
        bk = tl.load(desc + 8)
        bn = tl.load(desc + 9)
        cm = tl.load(desc + 10)
    rows = tile_m * BM + tl.arange(0, BM)
    cols = tile_n * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        red = start * BK + kk
        a = tl.load(
            A + ao + rows[:, None] * am + red[None, :] * ak,
            (rows[:, None] < M) & (red[None, :] < K),
            0,
        )
        b = tl.load(
            B + bo + red[:, None] * bk + cols[None, :] * bn,
            (red[:, None] < K) & (cols[None, :] < N),
            0,
        )
        acc = tl.dot(a, b, acc)
    if BIAS:
        acc += tl.load(Bias + tl.load(desc + 12) + cols, cols < N, 0)[None, :]
    _store_output(
        acc,
        C,
        co + rows[:, None] * cm + cols[None, :],
        co + cols[None, :],
        co,
        (rows[:, None] < M) & (cols[None, :] < N),
        KeepProbabilities,
        Seed,
        FUSED,
        DROPOUT,
        RANDOM,
        SCALAR_PROBABILITY,
        Addend,
        ALPHA,
        BETA,
    )


@triton.jit
def _specialized_group(
    A,
    B,
    Bias,
    C,
    Descriptors,
    Tasks,
    KeepProbabilities,
    Seed,
    CopyDescriptors,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    BIAS: tl.constexpr,
    FUSED: tl.constexpr,
    DROPOUT: tl.constexpr,
    RANDOM: tl.constexpr,
    SCALAR_PROBABILITY: tl.constexpr,
    COPY: tl.constexpr,
    GEMM_TASKS: tl.constexpr,
    MODE: tl.constexpr,
    COUNT: tl.constexpr,
    SHAPES: tl.constexpr,
    X_ROW_STRIDE: tl.constexpr,
    Y_ROW_STRIDE: tl.constexpr,
):
    task = tl.program_id(0)
    if COPY:
        if task >= GEMM_TASKS:
            _copy_member(
                A, C, CopyDescriptors, Tasks, task, X_ROW_STRIDE, Y_ROW_STRIDE, 1024
            )
    if not COPY or task < GEMM_TASKS:
        problem = tl.load(Tasks + task * 3)
        mi = tl.load(Tasks + task * 3 + 1)
        ni = tl.load(Tasks + task * 3 + 2)
        desc = Descriptors + problem * 13
        signature = tl.load(desc + 11)
        for index in tl.static_range(COUNT):
            if signature == index:
                _specialized_tile(
                    A,
                    B,
                    Bias,
                    C,
                    desc,
                    mi,
                    ni,
                    KeepProbabilities,
                    Seed,
                    SHAPES[6 * index],
                    SHAPES[6 * index + 1],
                    SHAPES[6 * index + 2],
                    MODE,
                    SHAPES[6 * index + 3],
                    SHAPES[6 * index + 4],
                    SHAPES[6 * index + 5],
                    BIAS,
                    FUSED,
                    DROPOUT,
                    RANDOM,
                    SCALAR_PROBABILITY,
                    X_ROW_STRIDE,
                    Y_ROW_STRIDE,
                )


@triton.jit
def _grouped_dot(
    A,
    B,
    Bias,
    C,
    Descriptors,
    Tasks,
    KeepProbabilities,
    Seed,
    CopyDescriptors,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    BIAS: tl.constexpr,
    FUSED: tl.constexpr,
    DROPOUT: tl.constexpr,
    RANDOM: tl.constexpr,
    SCALAR_PROBABILITY: tl.constexpr,
    COPY: tl.constexpr,
    GEMM_TASKS: tl.constexpr,
    X_ROW_STRIDE: tl.constexpr,
    Y_ROW_STRIDE: tl.constexpr,
    Addend=None,
    ALPHA: tl.constexpr = 1.0,
    BETA: tl.constexpr = 0.0,
):
    task = tl.program_id(0)
    if COPY:
        if task >= GEMM_TASKS:
            _copy_member(
                A, C, CopyDescriptors, Tasks, task, X_ROW_STRIDE, Y_ROW_STRIDE, 1024
            )
    if not COPY or task < GEMM_TASKS:
        problem = tl.load(Tasks + task * 3)
        tile_m = tl.load(Tasks + task * 3 + 1)
        tile_n = tl.load(Tasks + task * 3 + 2)
        desc = Descriptors + problem * 13
        ao = tl.load(desc)
        bo = tl.load(desc + 1)
        co = tl.load(desc + 2)
        m = tl.load(desc + 3)
        n = tl.load(desc + 4)
        k = tl.load(desc + 5)
        am = tl.load(desc + 6)
        ak = tl.load(desc + 7)
        bk = tl.load(desc + 8)
        bn = tl.load(desc + 9)
        cm = tl.load(desc + 10)
        bias_offset = tl.load(desc + 12)
        rows = tile_m * BM + tl.arange(0, BM)
        cols = tile_n * BN + tl.arange(0, BN)
        kk = tl.arange(0, BK)
        accumulator = tl.zeros((BM, BN), tl.float32)
        for start in range(tl.cdiv(k, BK)):
            reduction = start * BK + kk
            a = tl.load(
                A + ao + rows[:, None] * am + reduction[None, :] * ak,
                (rows[:, None] < m) & (reduction[None, :] < k),
                0,
            )
            b = tl.load(
                B + bo + reduction[:, None] * bk + cols[None, :] * bn,
                (reduction[:, None] < k) & (cols[None, :] < n),
                0,
            )
            accumulator = tl.dot(a, b, accumulator)
        if BIAS:
            accumulator += tl.load(Bias + bias_offset + cols, cols < n, 0)[None, :]
        _store_output(
            accumulator,
            C,
            co + rows[:, None] * cm + cols[None, :],
            co + cols[None, :],
            co,
            (rows[:, None] < m) & (cols[None, :] < n),
            KeepProbabilities,
            Seed,
            FUSED,
            DROPOUT,
            RANDOM,
            SCALAR_PROBABILITY,
            Addend,
            ALPHA,
            BETA,
        )


@triton.jit
def _grouped_vector(
    A,
    B,
    Bias,
    C,
    Descriptors,
    Tasks,
    KeepProbabilities,
    Seed,
    CopyDescriptors,
    BO: tl.constexpr,
    RK: tl.constexpr,
    BIAS: tl.constexpr,
    FUSED: tl.constexpr,
    DROPOUT: tl.constexpr,
    RANDOM: tl.constexpr,
    SCALAR_PROBABILITY: tl.constexpr,
    COPY: tl.constexpr,
    GEMM_TASKS: tl.constexpr,
    X_ROW_STRIDE: tl.constexpr,
    Y_ROW_STRIDE: tl.constexpr,
):
    task = tl.program_id(0)
    if COPY:
        if task >= GEMM_TASKS:
            _copy_member(
                A, C, CopyDescriptors, Tasks, task, X_ROW_STRIDE, Y_ROW_STRIDE, 1024
            )
    if not COPY or task < GEMM_TASKS:
        problem = tl.load(Tasks + task * 3)
        tile = tl.load(Tasks + task * 3 + 1)
        desc = Descriptors + problem * 13
        ao = tl.load(desc)
        bo = tl.load(desc + 1)
        co = tl.load(desc + 2)
        m = tl.load(desc + 3)
        n = tl.load(desc + 4)
        k = tl.load(desc + 5)
        am = tl.load(desc + 6)
        ak = tl.load(desc + 7)
        bk = tl.load(desc + 8)
        bn = tl.load(desc + 9)
        cm = tl.load(desc + 10)
        offsets = tile * BO + tl.arange(0, BO)
        rows = offsets // n
        cols = offsets % n
        kk = tl.arange(0, RK)
        acc = tl.zeros((BO, RK), tl.float32)
        for start in range(tl.cdiv(k, RK)):
            reduction = start * RK + kk
            a = tl.load(
                A + ao + rows[:, None] * am + reduction[None, :] * ak,
                (offsets[:, None] < m * n) & (reduction[None, :] < k),
                0,
            )
            b = tl.load(
                B + bo + reduction[None, :] * bk + cols[:, None] * bn,
                (offsets[:, None] < m * n) & (reduction[None, :] < k),
                0,
            )
            acc += a.to(tl.float32) * b.to(tl.float32)
        result = tl.sum(acc, 1)
        if BIAS:
            result += tl.load(Bias + tl.load(desc + 12) + cols, offsets < m * n, 0)
        _store_output(
            result,
            C,
            co + rows * cm + cols,
            co + cols,
            co,
            offsets < m * n,
            KeepProbabilities,
            Seed,
            FUSED,
            DROPOUT,
            RANDOM,
            SCALAR_PROBABILITY,
        )


@dataclasses.dataclass(frozen=True)
class Tile:
    m: int = 32
    n: int = 64
    k: int = 32
    warps: int = 4
    stages: int = 3


class LinearPlan:
    """Prepared forward/dX/dW with batch-major FP16/BF16 activations.

    Each shape is (batch rows, input features, output features). Metadata and
    output allocation can be reused; packing and dtype conversion are external.
    Aligned hidden layers share a fixed specialization vocabulary, including 384.
    Set specialize=False to compare against the runtime-dimension dot kernel.
    With members=..., activations keep the complete layout_shapes footprint;
    only active-member weights/biases are packed. Inactive output slots are left
    untouched by default; fused_relu=True copies their identity paths. Schedules
    reorder independent tiles without moving data.
    tiles_by_width selects per-branch tiles within one launch, using common
    warp/stage settings. It is keyed by max(input features, output features).
    Activations have shape [batch, sum(widths)]; every member shares the batch
    size. Weights and biases are contiguous vectors containing each matrix/vector
    in sequence. No activation layout conversion occurs inside the plan.
    """

    def __init__(
        self,
        shapes,
        mode='forward',
        *,
        device='cuda:0',
        tile=None,
        specialize=True,
        members=None,
        schedule='member',
        tiles_by_width=None,
        fused_relu=False,
    ):
        self.layout_shapes = [tuple(map(int, shape)) for shape in shapes]
        if not self.layout_shapes or any(
            len(s) != 3 or min(s) < 1 for s in self.layout_shapes
        ):
            raise ValueError('Expected nonempty positive (batch, input, output) shapes')
        if len({b for b, _, _ in self.layout_shapes}) != 1:
            raise ValueError('Batch-major activations require a common batch size')
        self.x_row_stride = sum(k for _, k, _ in self.layout_shapes)
        self.y_row_stride = sum(n for _, _, n in self.layout_shapes)
        self.x_shape = (self.layout_shapes[0][0], self.x_row_stride)
        self.y_shape = (self.layout_shapes[0][0], self.y_row_stride)
        self.member_ids = tuple(
            range(len(self.layout_shapes)) if members is None else members
        )
        if (
            not self.member_ids
            or len(set(self.member_ids)) != len(self.member_ids)
            or any(i < 0 or i >= len(self.layout_shapes) for i in self.member_ids)
        ):
            raise ValueError('Expected unique, valid active member indices')
        self.shapes = [self.layout_shapes[i] for i in self.member_ids]
        if mode not in ('forward', 'input_grad', 'weight_grad'):
            raise ValueError(mode)
        if schedule not in ('member', 'k_desc', 'round_robin'):
            raise ValueError(f'Unknown tile schedule: {schedule}')
        self.schedule = schedule
        self.mode = mode
        self.fused_relu = bool(fused_relu)
        if self.fused_relu and mode != 'forward':
            raise ValueError('The fused ReLU epilogue requires a forward plan')
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('This prototype requires CUDA')
        self.tile = tile or Tile()
        self.tiles_by_width = dict(tiles_by_width or {})
        for candidate in (self.tile, *self.tiles_by_width.values()):
            if any(
                size < 16 or size & (size - 1)
                for size in (candidate.m, candidate.n, candidate.k)
            ):
                raise ValueError('GEMM tile dimensions must be powers of two >= 16')
            if (candidate.warps, candidate.stages) != (
                self.tile.warps,
                self.tile.stages,
            ):
                raise ValueError('One launch requires common warp and stage settings')
        self.x_offsets, self.w_offsets, self.y_offsets, self.bias_offsets = (
            [],
            [],
            [],
            [],
        )
        x_size = y_size = 0
        layout_x_offsets, layout_y_offsets = [], []
        x_columns = y_columns = 0
        for batch, inputs, outputs in self.layout_shapes:
            layout_x_offsets.append(x_columns)
            layout_y_offsets.append(y_columns)
            x_size += batch * inputs
            y_size += batch * outputs
            x_columns += inputs
            y_columns += outputs
        w_size = bias_size = 0
        descs = []
        problems = []
        for member, (batch, inputs, outputs) in zip(self.member_ids, self.shapes):
            xo = layout_x_offsets[member]
            yo = layout_y_offsets[member]
            xs, ys = self.x_row_stride, self.y_row_stride
            self.x_offsets.append(xo)
            self.w_offsets.append(w_size)
            self.y_offsets.append(yo)
            self.bias_offsets.append(bias_size)
            if mode == 'forward':
                m, n, k = batch, outputs, inputs
                ao, bo, co = xo, w_size, yo
                am, ak, bk, bn, cm = xs, 1, 1, inputs, ys
            elif mode == 'input_grad':
                m, n, k = batch, inputs, outputs
                ao, bo, co = yo, w_size, xo
                am, ak, bk, bn, cm = ys, 1, inputs, 1, xs
            else:
                m, n, k = outputs, inputs, batch
                ao, bo, co = yo, xo, w_size
                am, ak, bk, bn, cm = 1, ys, xs, 1, inputs
            descs.append([ao, bo, co, m, n, k, am, ak, bk, bn, cm, 1, bias_size])
            problems.append((m, n, k))
            w_size += outputs * inputs
            bias_size += outputs
        self.x_size, self.w_size, self.y_size = x_size, w_size, y_size
        self.w_shape = (w_size,)
        self.bias_size = bias_size
        self.a_size, self.b_size, self.c_size = {
            'forward': (x_size, w_size, y_size),
            'input_grad': (y_size, w_size, x_size),
            'weight_grad': (y_size, x_size, w_size),
        }[mode]
        self.a_shape, self.b_shape, self.c_shape = {
            'forward': (self.x_shape, self.w_shape, self.y_shape),
            'input_grad': (self.y_shape, self.w_shape, self.x_shape),
            'weight_grad': (self.y_shape, self.x_shape, self.w_shape),
        }[mode]
        self.vector = all(min(shape) == 1 for shape in problems)
        self.signatures = tuple(sorted(set(problems)))
        self.specialized = (
            specialize
            and not self.vector
            and all(size % 16 == 0 for shape in problems for size in shape)
            and all(offset % 16 == 0 for desc in descs for offset in desc[:3])
            and self.x_row_stride % 16 == 0
            and self.y_row_stride % 16 == 0
        )
        self.hidden_vocabulary = (
            self.specialized
            and len({batch for batch, _, _ in self.shapes}) == 1
            and all(
                inputs in HIDDEN_SIZES and outputs in HIDDEN_SIZES
                for _, inputs, outputs in self.shapes
            )
        )
        self.kernel_signatures = self.signatures
        if self.hidden_vocabulary:
            batch = self.shapes[0][0]
            squares = tuple(
                (width, width, batch)
                if mode == 'weight_grad'
                else (batch, width, width)
                for width in HIDDEN_SIZES
            )
            self.kernel_signatures = squares + tuple(
                shape for shape in self.signatures if shape not in squares
            )
        self.kernel_kind = (
            'vector'
            if self.vector
            else 'hidden_vocabulary'
            if self.hidden_vocabulary
            else 'aligned_shapes'
            if self.specialized
            else 'generic'
        )
        if self.tiles_by_width and not self.specialized:
            raise ValueError('Width-specific tiles require aligned specialized GEMMs')
        self.kernel_tiles = tuple(
            self.tiles_by_width.get(
                max(shape[:2]) if mode == 'weight_grad' else max(shape[1:]),
                self.tile,
            )
            for shape in self.kernel_signatures
        )
        self.problem_tiles = tuple(
            self.kernel_tiles[self.kernel_signatures.index(shape)] for shape in problems
        )
        for desc, shape in zip(descs, problems):
            desc[11] = self.kernel_signatures.index(shape)
        self.vector_block = 1024 if all(k == 1 for _, _, k in problems) else 16
        self.reduction_block = min(
            128, triton.next_power_of_2(max(k for _, _, k in problems))
        )
        tasks = []
        self.useful_flops = 0
        self.scheduled_flops = 0
        for i, (m, n, k) in enumerate(problems):
            self.useful_flops += 2 * m * n * k
            if self.vector:
                tasks.extend(
                    (i, j, 0) for j in range(triton.cdiv(m * n, self.vector_block))
                )
                self.scheduled_flops += (
                    2
                    * triton.cdiv(m * n, self.vector_block)
                    * self.vector_block
                    * triton.cdiv(k, self.reduction_block)
                    * self.reduction_block
                )
            else:
                problem_tile = self.problem_tiles[i]
                tasks.extend(
                    (i, mi, ni)
                    for mi in range(triton.cdiv(m, problem_tile.m))
                    for ni in range(triton.cdiv(n, problem_tile.n))
                )
                self.scheduled_flops += 2 * math.prod(
                    triton.cdiv(size, block) * block
                    for size, block in zip(
                        (m, n, k),
                        (problem_tile.m, problem_tile.n, problem_tile.k),
                    )
                )
        if schedule == 'k_desc':
            tasks.sort(key=lambda task: -problems[task[0]][2])
        elif schedule == 'round_robin':
            counters = [0] * len(problems)
            ranked = []
            for task in tasks:
                problem = task[0]
                ranked.append((counters[problem], -problems[problem][2], task))
                counters[problem] += 1
            tasks = [task for _, _, task in sorted(ranked)]
        self.n_gemm_tasks = len(tasks)
        copies = []
        if self.fused_relu:
            for member, (batch, inputs, outputs) in enumerate(self.layout_shapes):
                if member in self.member_ids:
                    continue
                if inputs != outputs:
                    raise ValueError('Inactive identity paths require equal widths')
                copies.append(
                    (layout_x_offsets[member], layout_y_offsets[member], batch, inputs)
                )
                tasks.extend(
                    (len(copies) - 1, chunk, 0)
                    for chunk in range(triton.cdiv(batch * inputs, 1024))
                )
        self.copy_descriptors = torch.tensor(
            copies, dtype=torch.int64, device=self.device
        ).reshape(-1, 4)
        self.has_copies = bool(copies)
        self.descriptors = torch.tensor(descs, dtype=torch.int64, device=self.device)
        self.tasks = torch.tensor(tasks, dtype=torch.int32, device=self.device)
        self.n_tasks = len(tasks)

    def empty_output(self, dtype):
        return torch.empty(self.c_shape, device=self.device, dtype=dtype)

    def run(
        self,
        a,
        b,
        bias=None,
        *,
        out=None,
        keep_probabilities=None,
        seed=None,
        stochastic=False,
        scalar_probability=False,
    ):
        if a.dtype not in (torch.bfloat16, torch.float16) or a.dtype != b.dtype:
            raise ValueError('Expected equal FP16/BF16 input dtypes')
        if (
            any(
                t.device != self.descriptors.device or not t.is_contiguous()
                for t in (a, b)
            )
            or tuple(a.shape) != self.a_shape
            or tuple(b.shape) != self.b_shape
        ):
            raise ValueError('Input buffers do not match this prepared CUDA plan')
        if bias is not None and (
            self.mode != 'forward'
            or bias.numel() != self.bias_size
            or bias.device != a.device
            or bias.dtype != a.dtype
            or bias.ndim != 1
            or not bias.is_contiguous()
        ):
            raise ValueError('Invalid bias buffer')
        if keep_probabilities is not None and (
            not self.fused_relu
            or tuple(keep_probabilities.shape) != (self.y_row_stride,)
            or keep_probabilities.dtype != torch.float32
            or keep_probabilities.device != a.device
            or not keep_probabilities.is_contiguous()
        ):
            raise ValueError('Invalid fused keep-probability buffer')
        if scalar_probability and (not self.fused_relu or keep_probabilities is None):
            raise ValueError('Scalar probabilities require a fused dropout epilogue')
        if stochastic and (keep_probabilities is None or seed is None):
            raise ValueError('Stochastic dropout requires probabilities and a seed')
        if seed is not None and (
            not self.fused_relu
            or not stochastic
            or seed.numel() != 1
            or seed.dtype != torch.int64
            or seed.device != a.device
            or not seed.is_contiguous()
        ):
            raise ValueError('Invalid fused dropout seed')
        if out is None:
            out = self.empty_output(a.dtype)
        if (
            out.device != a.device
            or out.dtype != a.dtype
            or not out.is_contiguous()
            or tuple(out.shape) != self.c_shape
        ):
            raise ValueError('Invalid output buffer')
        if out.untyped_storage().data_ptr() in (
            a.untyped_storage().data_ptr(),
            b.untyped_storage().data_ptr(),
            *([bias.untyped_storage().data_ptr()] if bias is not None else []),
        ):
            raise ValueError('Output must not alias an input')
        epilogue = {
            'FUSED': self.fused_relu,
            'DROPOUT': keep_probabilities is not None,
            'RANDOM': stochastic,
            'SCALAR_PROBABILITY': bool(scalar_probability),
            'COPY': self.has_copies,
            'GEMM_TASKS': self.n_gemm_tasks if self.has_copies else 0,
            'X_ROW_STRIDE': self.x_row_stride,
            'Y_ROW_STRIDE': self.y_row_stride,
        }
        if self.vector:
            vector_kernel: Any = _grouped_vector
            vector_kernel[(self.n_tasks,)](
                a,
                b,
                bias if bias is not None else a,
                out,
                self.descriptors,
                self.tasks,
                keep_probabilities,
                seed,
                self.copy_descriptors,
                self.vector_block,
                self.reduction_block,
                bias is not None,
                **epilogue,
                num_warps=4,
            )
        else:
            tile = self.tile
            kernel: Any = _specialized_group if self.specialized else _grouped_dot
            specialization = {}
            if self.specialized:
                specialization = {
                    'MODE': self.mode,
                    'COUNT': len(self.kernel_signatures),
                    'SHAPES': tuple(
                        size
                        for shape, shape_tile in zip(
                            self.kernel_signatures, self.kernel_tiles
                        )
                        for size in (*shape, shape_tile.m, shape_tile.n, shape_tile.k)
                    ),
                }
            kernel[(self.n_tasks,)](
                a,
                b,
                bias if bias is not None else a,
                out,
                self.descriptors,
                self.tasks,
                keep_probabilities,
                seed,
                self.copy_descriptors,
                tile.m,
                tile.n,
                tile.k,
                bias is not None,
                **epilogue,
                **specialization,
                num_warps=tile.warps,
                num_stages=tile.stages,
            )
        return out
