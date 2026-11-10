"""Opt-in compact Newton-Schulz backend for the existing Muon optimizer.

The original optimizer's parameters, state, momentum and AdamW path stay padded.
Only Newton-Schulz workspaces and products use actual matrix sizes.
"""

from typing import Any

import torch
import triton as _triton
import triton.language as _tl

from tabpack_experimental.matrix_gemm import MatrixPlan
from tabpack_experimental.ragged_linear import MatrixTransfer
from tabpackage.nn import LinearPack
from tabpackage.optim import MuonAdamWPack

# Triton annotations describe GPU compilation, not Python call signatures.
triton: Any = _triton
tl: Any = _tl


@triton.jit
def _norm_partials(X, P, D, Tasks, BLOCK: tl.constexpr):
    task = tl.program_id(0)
    member, chunk = tl.load(Tasks + 2 * task), tl.load(Tasks + 2 * task + 1)
    desc = D + member * 5
    base, rows, cols = tl.load(desc), tl.load(desc + 1), tl.load(desc + 2)
    partial = tl.load(desc + 3)
    index = chunk * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + base + index, index < rows * cols, 0).to(tl.float32)
    tl.store(P + partial + chunk, tl.sum(x * x, 0))


@triton.jit
def _norm_finish(P, Norms, D, BLOCK: tl.constexpr):
    member = tl.program_id(0)
    desc = D + member * 5
    base, count = tl.load(desc + 3), tl.load(desc + 4)
    index = tl.arange(0, BLOCK)
    total = tl.sum(tl.load(P + base + index, index < count, 0), 0)
    norm = tl.sqrt(total).to(tl.bfloat16).to(tl.float32)
    denominator = (norm + 1e-7).to(tl.bfloat16)
    tl.store(Norms + member, denominator)


@triton.jit
def _normalize(X, Y, Norms, D, Tasks, BLOCK: tl.constexpr):
    task = tl.program_id(0)
    member, chunk = tl.load(Tasks + 2 * task), tl.load(Tasks + 2 * task + 1)
    desc = D + member * 5
    base, rows, cols = tl.load(desc), tl.load(desc + 1), tl.load(desc + 2)
    index = chunk * BLOCK + tl.arange(0, BLOCK)
    source = tl.where(rows > cols, (index % rows) * cols + index // rows, index)
    x = tl.load(X + base + source, index < rows * cols, 0).to(tl.float32)
    denominator = tl.load(Norms + member).to(tl.float32)
    tl.store(Y + base + index, x / denominator, index < rows * cols)


@triton.jit
def _restore_orientation(X, Y, D, Tasks, BLOCK: tl.constexpr):
    task = tl.program_id(0)
    member, chunk = tl.load(Tasks + 2 * task), tl.load(Tasks + 2 * task + 1)
    desc = D + member * 5
    base, rows, cols = tl.load(desc), tl.load(desc + 1), tl.load(desc + 2)
    index = chunk * BLOCK + tl.arange(0, BLOCK)
    source = tl.where(rows > cols, (index % cols) * rows + index // cols, index)
    value = tl.load(X + base + source, index < rows * cols, 0)
    tl.store(Y + base + index, value, index < rows * cols)


class MuonPlan:
    """Prepared BF16 Newton-Schulz workspaces; not reentrant or differentiable.

    Shapes are original (output, input) matrix dimensions. Normalization and
    orientation are independent per member. The returned public output owns
    its storage. Warm construction/compilation before CUDA Graph capture.
    """

    def __init__(
        self,
        shapes,
        *,
        device='cuda:0',
        steps=5,
        specialize=True,
        schedule='k_desc',
        tiled=True,
        large_tile=(64, 64, 32),
    ):
        self.shapes = tuple(tuple(map(int, s)) for s in shapes)
        if not self.shapes or any(len(s) != 2 or min(s) < 1 for s in self.shapes):
            raise ValueError('Expected positive matrix shapes')
        if not isinstance(steps, int) or steps < 0:
            raise ValueError('Expected nonnegative Newton-Schulz steps')
        self.steps, self.device = steps, torch.device(device)
        options = {
            'device': device,
            'specialize': specialize,
            'schedule': schedule,
            'tiled': tiled,
            'large_tile': large_tile,
        }
        oriented = [(min(r, c), max(r, c)) for r, c in self.shapes]
        self.gram = MatrixPlan(
            [(r, r, s) for r, s in oriented], transpose_b=True, **options
        )
        self.square = MatrixPlan([(r, r, r) for r, _ in oriented], **options)
        self.product = MatrixPlan([(r, s, r) for r, s in oriented], **options)
        self.size = sum(r * c for r, c in self.shapes)
        self.gram_size = sum(r * r for r, _ in oriented)
        self.transpose = any(r > c for r, c in self.shapes)
        descs, tasks, base, partial = [], [], 0, 0
        for member, (r, c) in enumerate(self.shapes):
            count = triton.cdiv(r * c, 2048)
            descs.append((base, r, c, partial, count))
            tasks.extend((member, chunk) for chunk in range(count))
            base += r * c
            partial += count
        self.descriptors = torch.tensor(descs, device=device, dtype=torch.int64)
        self.tasks = torch.tensor(tasks, device=device, dtype=torch.int32)
        self.norm_block = triton.next_power_of_2(max(d[4] for d in descs))
        self.partials = torch.empty(partial, device=device, dtype=torch.float32)
        self.norms = torch.empty(len(self.shapes), device=device, dtype=torch.bfloat16)
        self.x, self.y = [
            torch.empty(self.size, device=device, dtype=torch.bfloat16)
            for _ in range(2)
        ]
        self.a, self.b = [
            torch.empty(self.gram_size, device=device, dtype=torch.bfloat16)
            for _ in range(2)
        ]
        self.transfers = {}

    def _orthogonalize(self, update):
        if (
            update.device != self.descriptors.device
            or update.dtype != torch.bfloat16
            or update.ndim != 1
            or update.numel() != self.size
            or not update.is_contiguous()
        ):
            raise ValueError('Expected a matching flat contiguous CUDA BF16 update')
        partials: Any = _norm_partials
        finish: Any = _norm_finish
        normalize: Any = _normalize
        partials[(self.tasks.shape[0],)](
            update, self.partials, self.descriptors, self.tasks, 2048
        )
        finish[(len(self.shapes),)](
            self.partials, self.norms, self.descriptors, self.norm_block
        )
        normalize[(self.tasks.shape[0],)](
            update, self.x, self.norms, self.descriptors, self.tasks, 2048
        )
        x, y = self.x, self.y
        for _ in range(self.steps):
            self.gram.run(x, x, out=self.a)
            self.square.run(
                self.a, self.a, out=self.b, addend=self.a, alpha=2.0315, beta=-4.7750
            )
            self.product.run(self.b, x, out=y, addend=x, beta=3.4445)
            x, y = y, x
        return x

    @torch.no_grad()
    def run(self, update):
        x = self._orthogonalize(update)
        if not self.transpose:
            return x.clone()
        output = torch.empty_like(x)
        restore: Any = _restore_orientation
        restore[(self.tasks.shape[0],)](x, output, self.descriptors, self.tasks, 2048)
        return output

    @torch.no_grad()
    def run_padded(self, update):
        if (
            update.ndim != 3
            or update.shape[0] != len(self.shapes)
            or update.device != self.descriptors.device
            or update.dtype not in (torch.float32, torch.float16, torch.bfloat16)
            or any(r > update.shape[1] or c > update.shape[2] for r, c in self.shapes)
        ):
            raise ValueError('Padded operand does not match the prepared Muon plan')
        key = tuple(update.shape)
        if key not in self.transfers:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('Warm the Muon transfer before graph capture')
            self.transfers[key] = MatrixTransfer(self.shapes, key, update.device)
        transfer = self.transfers[key]
        compact = transfer.pack(update, torch.bfloat16)
        x = self._orthogonalize(compact)
        if self.transpose:
            original = torch.empty_like(x)
            restore: Any = _restore_orientation
            restore[(self.tasks.shape[0],)](
                x, original, self.descriptors, self.tasks, 2048
            )
            x = original
        return transfer.unpack(x, torch.bfloat16)


class RaggedMuonAdapter:
    """Replace only one optimizer instance's Muon orthogonalization backend.

    Matrix shapes, membership and parameter identities are static. Discard/rebuild
    after pack removal. Padding in gradients and existing momentum must be zero.
    This compatibility adapter intentionally retains padded FP32 optimizer state.
    """

    def __init__(
        self, optimizer, shapes_by_parameter, *, modules_by_parameter=None, **options
    ):
        if not isinstance(optimizer, MuonAdamWPack):
            raise TypeError('Expected the repository MuonAdamWPack optimizer')
        if getattr(optimizer, '_ragged_muon_adapter', None) is not None:
            raise ValueError('Optimizer already has a ragged Muon adapter')
        self.optimizer = optimizer
        self.original_step = optimizer._step_muon
        self.original_zeropower = optimizer._zeropower_via_newtonschulz5
        self.modules = modules_by_parameter or {}
        self.module_signatures = {
            p: self._module_signature(m) for p, m in self.modules.items()
        }
        self.fallback_calls = 0
        self.plans = {}
        self.padded_shapes = {}
        self.calls = 0
        for group in optimizer.param_groups:
            if not group['muon']:
                continue
            for p in group['params']:
                if p not in shapes_by_parameter:
                    raise ValueError('Missing logical matrix shapes')
                shapes = tuple(shapes_by_parameter[p])
                if (
                    p.device.type != 'cuda'
                    or p.ndim != 3
                    or p.dtype != torch.float32
                    or not p.is_contiguous()
                    or len(shapes) != p.shape[0]
                    or any(
                        min(s) < 1 or s[0] > p.shape[1] or s[1] > p.shape[2]
                        for s in shapes
                    )
                ):
                    raise ValueError('Invalid padded FP32 parameter or logical shapes')
                mask = torch.zeros_like(p, dtype=torch.bool)
                for member, (rows, cols) in enumerate(shapes):
                    mask[member, :rows, :cols] = True
                buffers = [optimizer.state[p].get('muon_momentum_buffer'), p.grad]
                if any(
                    t is not None and torch.count_nonzero(t[~mask]).item()
                    for t in buffers
                ):
                    raise ValueError('Gradient/momentum padding must be zero')
                self.plans[p] = MuonPlan(
                    shapes,
                    device=p.device,
                    steps=group['muon_newton_schulz_steps'],
                    **options,
                )
                self.padded_shapes[p] = tuple(p.shape)
        if not self.plans:
            raise ValueError('Optimizer has no Muon matrix groups')
        self.enable()

    def enable(self):
        owner = getattr(self.optimizer, '_ragged_muon_adapter', None)
        if owner is not None and owner is not self:
            raise ValueError('Another ragged Muon adapter owns this optimizer')
        self.optimizer._step_muon = self.step_group
        self.optimizer._ragged_muon_adapter = self

    def disable(self):
        self.optimizer._step_muon = self.original_step
        self.optimizer._ragged_muon_adapter = None

    @staticmethod
    def _module_signature(module):
        return (
            id(module.weight),
            tuple(module.weight.shape),
            module.weight.device,
            *(
                (id(t), t._version) if t is not None else None
                for t in (module.in_features, module.out_features)
            ),
        )

    def refresh_metadata(self):
        for p in self.modules:
            if p in self.plans:
                self._refresh_module_metadata(p)

    def _refresh_module_metadata(self, p):
        module = self.modules[p]
        signature = self._module_signature(module)
        if signature == self.module_signatures[p]:
            return
        rows = (
            module.out_features.tolist()
            if module.out_features is not None
            else [module.max_out_features] * module.pack_size
        )
        cols = (
            module.in_features.tolist()
            if module.in_features is not None
            else [module.max_in_features] * module.pack_size
        )
        if module.weight is p and tuple(zip(rows, cols)) == self.plans[p].shapes:
            self.module_signatures[p] = signature

    def step_group(self, group):
        for p in group['params']:
            if p in self.modules and p in self.plans:
                self._refresh_module_metadata(p)
            if (
                p not in self.plans
                or tuple(p.shape) != self.padded_shapes[p]
                or (
                    p in self.modules
                    and self._module_signature(self.modules[p])
                    != self.module_signatures[p]
                )
            ):
                raise ValueError(
                    'Rebuild the adapter after changing pack membership or widths'
                )
            if group['muon_newton_schulz_steps'] != self.plans[p].steps:
                raise ValueError(
                    'Rebuild the adapter after changing Newton-Schulz steps'
                )
        # Native skipping passes a dynamically selected subset into orthogonalization.
        # Preserve that policy rather than guessing which member each row represents.
        if group['muon_skip_zero_updates']:
            self.fallback_calls += 1
            return self.original_step(group)
        parameters = iter(p for p in group['params'] if p.grad is not None)

        def orthogonalize(update, steps):
            p = next(parameters)
            self.calls += 1
            return self.plans[p].run_padded(update)

        optimizer = self.optimizer
        optimizer._zeropower_via_newtonschulz5 = orthogonalize
        try:
            return self.original_step(group)
        finally:
            optimizer._zeropower_via_newtonschulz5 = self.original_zeropower


def install_ragged_muon(optimizer, model, **options):
    shapes, modules = {}, {}
    for module in model.modules():
        if isinstance(module, LinearPack):
            rows = (
                module.out_features.tolist()
                if module.out_features is not None
                else [module.max_out_features] * module.pack_size
            )
            cols = (
                module.in_features.tolist()
                if module.in_features is not None
                else [module.max_in_features] * module.pack_size
            )
            shapes[module.weight] = tuple(zip(rows, cols))
            modules[module.weight] = module
    return RaggedMuonAdapter(optimizer, shapes, modules_by_parameter=modules, **options)
