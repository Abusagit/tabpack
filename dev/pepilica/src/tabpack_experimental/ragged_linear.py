"""Opt-in ragged GEMM adapter for the existing padded LinearPack interface."""

from typing import Any

import torch
import triton as _triton
import triton.language as _tl
from torch.autograd.function import once_differentiable
from torch.utils.hooks import RemovableHandle

from tabpack_experimental.compact_mlp import width_tiles
from tabpack_experimental.ragged import LinearPlan
from tabpackage.nn import LinearPack, MLPBackbonePack, _get_tensor_pack

# Triton annotations describe GPU compilation, not Python call signatures.
triton: Any = _triton
tl: Any = _tl


@triton.jit
def _pack_matrix(
    S,
    D,
    Descriptors,
    Tasks,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    task = tl.program_id(0)
    member = tl.load(Tasks + task * 2)
    chunk = tl.load(Tasks + task * 2 + 1)
    desc = Descriptors + member * 3
    base = tl.load(desc)
    rows = tl.load(desc + 1)
    columns = tl.load(desc + 2)
    index = chunk * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(
        S + member * S0 + (index // columns) * S1 + (index % columns) * S2,
        index < rows * columns,
        0,
    )
    stride = ROW_STRIDE if ROW_STRIDE else columns
    destination = base + (index // columns) * stride + index % columns
    tl.store(D + destination, values, index < rows * columns)


@triton.jit
def _unpack_matrix(
    S,
    D,
    Descriptors,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    member = tl.program_id(1)
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    desc = Descriptors + member * 3
    base = tl.load(desc)
    rows = tl.load(desc + 1)
    columns = tl.load(desc + 2)
    r, c = index // COLS, index % COLS
    stride = ROW_STRIDE if ROW_STRIDE else columns
    values = tl.load(
        S + base + r * stride + c,
        (index < ROWS * COLS) & (r < rows) & (c < columns),
        0,
    )
    tl.store(D + member * ROWS * COLS + index, values, index < ROWS * COLS)


@triton.jit
def _bias_grad(DY, DB, Descriptors, Tasks, BM: tl.constexpr, BN: tl.constexpr):
    task = tl.program_id(0)
    member = tl.load(Tasks + task * 2)
    column_tile = tl.load(Tasks + task * 2 + 1)
    desc = Descriptors + member * 13
    base = tl.load(desc + 2)
    batch = tl.load(desc + 3)
    width = tl.load(desc + 4)
    stride = tl.load(desc + 10)
    bias_base = tl.load(desc + 12)
    rows = tl.arange(0, BM)
    columns = column_tile * BN + tl.arange(0, BN)
    total = tl.zeros((BN,), tl.float32)
    for start in range(tl.cdiv(batch, BM)):
        r = start * BM + rows
        values = tl.load(
            DY + base + r[:, None] * stride + columns[None, :],
            (r[:, None] < batch) & (columns[None, :] < width),
            0,
        )
        total += tl.sum(values.to(tl.float32), 0)
    tl.store(DB + bias_base + columns, total, columns < width)


class MatrixTransfer:
    def __init__(self, shapes, padded_shape, device):
        self.padded_shape = tuple(padded_shape)
        descs, tasks = [], []
        total = 0
        for member, (rows, columns) in enumerate(shapes):
            descs.append((total, rows, columns))
            tasks.extend(
                (member, chunk) for chunk in range(triton.cdiv(rows * columns, 1024))
            )
            total += rows * columns
        self.size = total
        self.compact_shape = (total,)
        self.row_stride = 0
        self.descriptors = torch.tensor(descs, dtype=torch.int64, device=device)
        self.tasks = torch.tensor(tasks, dtype=torch.int32, device=device)

    def pack(self, source, dtype):
        if tuple(source.shape) != self.padded_shape:
            raise ValueError('Packed-linear operand has an unexpected shape')
        output = torch.empty(self.compact_shape, dtype=dtype, device=source.device)
        kernel: Any = _pack_matrix
        kernel[(self.tasks.shape[0],)](
            source,
            output,
            self.descriptors,
            self.tasks,
            *source.stride(),
            self.row_stride,
            1024,
        )
        return output

    def unpack(self, source, dtype):
        output = torch.empty(self.padded_shape, dtype=dtype, device=source.device)
        _, rows, columns = self.padded_shape
        kernel: Any = _unpack_matrix
        kernel[(triton.cdiv(rows * columns, 1024), self.padded_shape[0])](
            source, output, self.descriptors, rows, columns, self.row_stride, 1024
        )
        return output


class ActivationTransfer(MatrixTransfer):
    """Transfer padded member matrices to/from batch-major feature columns."""

    def __init__(self, shapes, padded_shape, device):
        self.padded_shape = tuple(padded_shape)
        self.row_stride = sum(columns for _, columns in shapes)
        self.compact_shape = (padded_shape[1], self.row_stride)
        self.size = self.compact_shape[0] * self.row_stride
        descriptors, tasks = [], []
        offset = 0
        for member, (rows, columns) in enumerate(shapes):
            descriptors.append((offset, rows, columns))
            tasks.extend(
                (member, chunk) for chunk in range(triton.cdiv(rows * columns, 1024))
            )
            offset += columns
        self.descriptors = torch.tensor(descriptors, dtype=torch.int64, device=device)
        self.tasks = torch.tensor(tasks, dtype=torch.int32, device=device)


class PackedLinearState:
    def __init__(self, shapes, max_input, max_output, device, *, schedule, tiled):
        count, batch = len(shapes), shapes[0][0]
        self.x = ActivationTransfer(
            [(b, k) for b, k, _ in shapes], (count, batch, max_input), device
        )
        self.w = MatrixTransfer(
            [(n, k) for _, k, n in shapes], (count, max_output, max_input), device
        )
        self.y = ActivationTransfer(
            [(b, n) for b, _, n in shapes], (count, batch, max_output), device
        )
        self.bias = MatrixTransfer(
            [(1, n) for _, _, n in shapes], (count, 1, max_output), device
        )
        widths = [max(k, n) for _, k, n in shapes]
        self.plans = {}
        for mode in ('forward', 'input_grad', 'weight_grad'):
            aligned = all(size % 16 == 0 for shape in shapes for size in shape)
            self.plans[mode] = LinearPlan(
                shapes,
                mode,
                device=device,
                schedule=schedule,
                tiles_by_width=width_tiles(widths, batch, mode, large=(64, 128, 32))
                if tiled and aligned and all(k == n for _, k, n in shapes)
                else None,
            )
        self.bias_tasks = torch.tensor(
            [
                (i, j)
                for i, (_, _, n) in enumerate(shapes)
                for j in range(triton.cdiv(n, 128))
            ],
            dtype=torch.int32,
            device=device,
        )


class _RaggedLinear(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(ctx, x, weight, bias, state, dtype):
        cx, cw = state.x.pack(x, dtype), state.w.pack(weight, dtype)
        cb = state.bias.pack(bias[:, None], dtype) if bias is not None else None
        cy = state.plans['forward'].run(cx, cw, cb)
        ctx.save_for_backward(cx, cw)
        ctx.state = state
        ctx.x_dtype, ctx.w_dtype = x.dtype, weight.dtype
        ctx.bias_dtype = bias.dtype if bias is not None else None
        return state.y.unpack(cy, dtype)

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    @once_differentiable
    def backward(ctx, grad_output):
        cx, cw = ctx.saved_tensors
        state = ctx.state
        dy = state.y.pack(grad_output, cx.dtype)
        dx = dw = db = None
        if ctx.needs_input_grad[0]:
            dx = state.x.unpack(state.plans['input_grad'].run(dy, cw), ctx.x_dtype)
        if ctx.needs_input_grad[1]:
            dw = state.w.unpack(state.plans['weight_grad'].run(dy, cx), ctx.w_dtype)
        if ctx.needs_input_grad[2] and ctx.bias_dtype is not None:
            cb = torch.empty(state.bias.size, dtype=cx.dtype, device=cx.device)
            kernel: Any = _bias_grad
            kernel[(state.bias_tasks.shape[0],)](
                dy, cb, state.plans['forward'].descriptors, state.bias_tasks, 32, 128
            )
            db = state.bias.unpack(cb, ctx.bias_dtype)[:, 0]
        return dx, dw, db, None, None


class RaggedLinearBackend:
    """Preserves padded parameters, masks, outputs, and optimizer references.

    Like LinearPack, requires masked input padding. Gradients in discarded input
    padding are zero; upstream input masking makes these slots unobservable.
    Plans for member selections/batch sizes are cached outside timed warm runs.
    """

    def __init__(self, module, *, schedule='member', tiled=False):
        self.module = module
        self.original_forward = module.forward
        self.load_hook: RemovableHandle | None = None
        self._refresh_dimensions()
        self.schedule, self.tiled = schedule, tiled
        self.calls = self.fallback_calls = 0

    def _dimension_signature(self):
        module = self.module
        return (
            id(module.weight),
            tuple(module.weight.shape),
            module.weight.device,
            *(
                (id(t), t._version) if t is not None else None
                for t in (module.in_features, module.out_features)
            ),
        )

    def _refresh_dimensions(self):
        module = self.module
        geometry = (tuple(module.weight.shape), module.weight.device)
        widths_in = (
            tuple(module.in_features.tolist())
            if module.in_features is not None
            else (module.max_in_features,) * module.pack_size
        )
        widths_out = (
            tuple(module.out_features.tolist())
            if module.out_features is not None
            else (module.max_out_features,) * module.pack_size
        )
        if getattr(self, 'geometry', None) != geometry or (
            getattr(self, 'widths_in', None),
            getattr(self, 'widths_out', None),
        ) != (widths_in, widths_out):
            self.cache = {}
            self.plans = {}
        self.geometry = geometry
        self.widths_in, self.widths_out = widths_in, widths_out
        self.dimension_signature = self._dimension_signature()

    def _after_load(self, module, incompatible_keys):
        self._refresh_dimensions()

    def prepare(self, batch_size, pack_idx=None):
        """Prepare selection metadata without running a layer or consuming RNG."""
        module = self.module
        if self.dimension_signature != self._dimension_signature():
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    'Warm updated linear dimensions before graph capture'
                )
            self._refresh_dimensions()
        inference_members = (
            tuple(pack_idx.tolist())
            if pack_idx is not None and torch.is_inference(pack_idx)
            else None
        )
        key = (
            inference_members if inference_members is not None else pack_idx,
            pack_idx._version
            if pack_idx is not None and inference_members is None
            else 0,
            batch_size,
            module.weight.device,
        )
        if key not in self.cache:
            members = (
                tuple(range(module.pack_size))
                if pack_idx is None
                else inference_members
                if inference_members is not None
                else tuple(pack_idx.tolist())
            )
            plan_key = (members, batch_size)
            if plan_key not in self.plans:
                shapes = [
                    (batch_size, self.widths_in[i], self.widths_out[i]) for i in members
                ]
                self.plans[plan_key] = PackedLinearState(
                    shapes,
                    module.max_in_features,
                    module.max_out_features,
                    module.weight.device,
                    schedule=self.schedule,
                    tiled=self.tiled,
                )
            self.cache[key] = self.plans[plan_key]
        return self.cache[key]

    def forward(self, x, pack_idx=None):
        module = self.module
        dtype = (
            torch.get_autocast_dtype('cuda')
            if torch.is_autocast_enabled('cuda')
            else x.dtype
        )
        if (
            x.device.type != 'cuda'
            or dtype not in (torch.float16, torch.bfloat16)
            or x.shape[0] == 0
            or x.dtype == torch.float64
            or module.weight.dtype == torch.float64
        ):
            self.fallback_calls += 1
            return self.original_forward(x, pack_idx)
        if not torch.is_autocast_enabled('cuda') and (
            module.weight.dtype != x.dtype
            or (module.bias is not None and module.bias.dtype != x.dtype)
        ):
            self.fallback_calls += 1
            return self.original_forward(x, pack_idx)
        if x.ndim != 3 or x.shape[-1] != module.max_in_features:
            raise ValueError('Unexpected LinearPack input dimensions')
        state = self.prepare(x.shape[1], pack_idx)
        self.calls += 1
        weight = _get_tensor_pack(module.weight, pack_idx)
        bias = _get_tensor_pack(module.bias, pack_idx)
        out_features = _get_tensor_pack(module.out_features, pack_idx)
        y = _RaggedLinear.apply(x, weight, bias, state, dtype)
        if out_features is not None:
            output_mask = (
                torch.arange(module.max_out_features, device=x.device)[None]
                < out_features[:, None]
            )
            y = y * output_mask.to(y.dtype).unsqueeze(1)
        return y

    def remove(self):
        self.module.forward = self.original_forward
        if self.load_hook is not None:
            self.load_hook.remove()


def install_ragged_linears(model, *, scope='hidden', schedule='member', tiled=False):
    """Patch linear execution only; parameter objects/state_dict stay unchanged."""
    if scope not in ('hidden', 'all'):
        raise ValueError(scope)
    backends = {}
    hidden_layers = {
        id(block.linear)
        for module in model.modules()
        if isinstance(module, MLPBackbonePack)
        for block in list(module.blocks)[1:]
    }
    for name, module in model.named_modules():
        if not isinstance(module, LinearPack):
            continue
        hidden = id(module) in hidden_layers
        if scope == 'hidden' and not hidden:
            continue
        if hasattr(module, '_ragged_linear_backend'):
            raise ValueError(f'Already patched: {name}')
        backend = RaggedLinearBackend(module, schedule=schedule, tiled=tiled)
        backend.load_hook = module.register_load_state_dict_post_hook(
            backend._after_load
        )
        module._ragged_linear_backend = backend

        module.forward = backend.forward
        backends[name] = backend
    return backends


def remove_ragged_linears(backends):
    for backend in backends.values():
        backend.remove()
        del backend.module._ragged_linear_backend
