"""Grouped parameter-matrix products using the existing ragged GEMM kernels."""

from typing import Any

import torch
import triton as _triton
import triton.language as _tl

from tabpack_experimental import ragged

# Triton annotations describe GPU compilation, not Python call signatures.
triton: Any = _triton
tl: Any = _tl


@triton.jit
def _matrix_tile(
    A,
    B,
    C,
    desc,
    mi,
    ni,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    TA: tl.constexpr,
    TB: tl.constexpr,
    Addend,
    ALPHA: tl.constexpr,
    BETA: tl.constexpr,
):
    if not TA and TB:
        ragged._specialized_tile(
            A,
            B,
            A,
            C,
            desc,
            mi,
            ni,
            None,
            None,
            M,
            N,
            K,
            'forward',
            BM,
            BN,
            BK,
            False,
            False,
            False,
            False,
            False,
            K,
            N,
            Addend,
            ALPHA,
            BETA,
        )
    elif not TA and not TB:
        ragged._specialized_tile(
            A,
            B,
            A,
            C,
            desc,
            mi,
            ni,
            None,
            None,
            M,
            N,
            K,
            'input_grad',
            BM,
            BN,
            BK,
            False,
            False,
            False,
            False,
            False,
            N,
            K,
            Addend,
            ALPHA,
            BETA,
        )
    elif TA and not TB:
        ragged._specialized_tile(
            A,
            B,
            A,
            C,
            desc,
            mi,
            ni,
            None,
            None,
            M,
            N,
            K,
            'weight_grad',
            BM,
            BN,
            BK,
            False,
            False,
            False,
            False,
            False,
            N,
            M,
            Addend,
            ALPHA,
            BETA,
        )
    else:
        ragged._specialized_tile(
            A,
            B,
            A,
            C,
            desc,
            mi,
            ni,
            None,
            None,
            M,
            N,
            K,
            'matrix',
            BM,
            BN,
            BK,
            False,
            False,
            False,
            False,
            False,
            0,
            0,
            Addend,
            ALPHA,
            BETA,
        )


@triton.jit
def _matrix_group(
    A,
    B,
    C,
    Descriptors,
    Tasks,
    COUNT: tl.constexpr,
    SHAPES: tl.constexpr,
    TA: tl.constexpr,
    TB: tl.constexpr,
    Addend,
    ALPHA: tl.constexpr,
    BETA: tl.constexpr,
):
    task = tl.program_id(0)
    member = tl.load(Tasks + task * 3)
    mi, ni = tl.load(Tasks + task * 3 + 1), tl.load(Tasks + task * 3 + 2)
    desc = Descriptors + member * 13
    signature = tl.load(desc + 11)
    for index in tl.static_range(COUNT):
        if signature == index:
            _matrix_tile(
                A,
                B,
                C,
                desc,
                mi,
                ni,
                SHAPES[6 * index],
                SHAPES[6 * index + 1],
                SHAPES[6 * index + 2],
                SHAPES[6 * index + 3],
                SHAPES[6 * index + 4],
                SHAPES[6 * index + 5],
                TA,
                TB,
                Addend,
                ALPHA,
                BETA,
            )


class MatrixPlan:
    """C_i = op(A_i) op(B_i), with compact row-major matrix buffers.

    Shapes specify logical (M, N, K). Each operand's matrices are concatenated
    in member order; transpose flags describe its physical row-major storage.
    This planner is independent of the model's batch-major activation layout.
    """

    def __init__(
        self,
        shapes,
        *,
        transpose_a=False,
        transpose_b=False,
        device='cuda:0',
        specialize=True,
        schedule='k_desc',
        tiled=True,
        large_tile=(64, 64, 32),
    ):
        self.shapes = tuple(tuple(map(int, s)) for s in shapes)
        if not self.shapes or any(len(s) != 3 or min(s) < 1 for s in self.shapes):
            raise ValueError('Expected positive (M, N, K) shapes')
        if schedule not in ('member', 'k_desc', 'round_robin'):
            raise ValueError('Unknown tile schedule')
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('MatrixPlan requires CUDA')
        self.transpose_a = bool(transpose_a)
        self.transpose_b = bool(transpose_b)
        descs, offsets = [], [0, 0, 0]
        for m, n, k in self.shapes:
            am, ak = (1, m) if transpose_a else (k, 1)
            bk, bn = (1, k) if transpose_b else (n, 1)
            descs.append([*offsets, m, n, k, am, ak, bk, bn, n, 0, 0])
            offsets = [offsets[0] + m * k, offsets[1] + k * n, offsets[2] + m * n]
        self.sizes = tuple(offsets)
        self.specialized = bool(
            specialize
            and all(size % 16 == 0 for s in self.shapes for size in s)
            and all(o % 16 == 0 for d in descs for o in d[:3])
        )
        self.tile = ragged.Tile()
        self.signatures = tuple(sorted(set(self.shapes)))
        if self.specialized and all(
            m == n == k and m in ragged.HIDDEN_SIZES for m, n, k in self.shapes
        ):
            self.signatures = tuple((h, h, h) for h in ragged.HIDDEN_SIZES)
        self.tiles = []
        for shape in self.signatures:
            m, n, k = shape
            base = (
                (16, 32, 32)
                if max(shape) <= 64
                else large_tile
                if min(m, n) >= 1024
                else (32, 64, 32)
            )
            candidate = (
                ragged.Tile(
                    *(
                        max(16, min(b, triton.next_power_of_2(s)))
                        for b, s in zip(base, shape)
                    )
                )
                if tiled and self.specialized and shape in self.shapes
                else self.tile
            )
            if any(
                v < 16 or v & (v - 1) for v in (candidate.m, candidate.n, candidate.k)
            ):
                raise ValueError('Tile dimensions must be powers of two >=16')
            self.tiles.append(candidate)
        tasks = []
        self.useful_flops = self.scheduled_flops = 0
        for member, (desc, shape) in enumerate(zip(descs, self.shapes)):
            index = self.signatures.index(shape)
            desc[11] = index
            tile = self.tiles[index]
            m, n, k = shape
            tasks.extend(
                (member, mi, ni)
                for mi in range(triton.cdiv(m, tile.m))
                for ni in range(triton.cdiv(n, tile.n))
            )
            self.useful_flops += 2 * m * n * k
            self.scheduled_flops += (
                2
                * triton.cdiv(m, tile.m)
                * tile.m
                * triton.cdiv(n, tile.n)
                * tile.n
                * triton.cdiv(k, tile.k)
                * tile.k
            )
        if schedule == 'k_desc':
            tasks.sort(key=lambda task: -self.shapes[task[0]][2])
        elif schedule == 'round_robin':
            counts = [0] * len(self.shapes)
            ranked = []
            for task in tasks:
                member = task[0]
                ranked.append((counts[member], -self.shapes[member][2], task))
                counts[member] += 1
            tasks = [task for _, _, task in sorted(ranked)]
        self.descriptors = torch.tensor(descs, device=self.device, dtype=torch.int64)
        self.tasks = torch.tensor(tasks, device=self.device, dtype=torch.int32)

    @torch.no_grad()
    def run(self, a, b, *, out=None, addend=None, alpha=1.0, beta=0.0):
        if a.dtype not in (torch.bfloat16, torch.float16) or b.dtype != a.dtype:
            raise ValueError('Expected matching FP16/BF16 operands')
        if any(
            t.device != self.descriptors.device
            or t.ndim != 1
            or not t.is_contiguous()
            or t.numel() != size
            for t, size in zip((a, b), self.sizes[:2])
        ):
            raise ValueError('Operand does not match the prepared matrix plan')
        if out is None:
            out = torch.empty(self.sizes[2], device=a.device, dtype=a.dtype)
        if (
            out.device != a.device
            or out.dtype != a.dtype
            or out.ndim != 1
            or not out.is_contiguous()
            or out.numel() != self.sizes[2]
        ):
            raise ValueError('Invalid output buffer')
        if any(
            out.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
            for t in (a, b)
        ):
            raise ValueError('Output must not alias an operand')
        if beta != 0.0 and (
            addend is None
            or addend.shape != out.shape
            or addend.dtype != out.dtype
            or addend.device != out.device
            or not addend.is_contiguous()
        ):
            raise ValueError('Invalid matrix addend')
        tile = self.tile
        flags = {
            'FUSED': False,
            'DROPOUT': False,
            'RANDOM': False,
            'SCALAR_PROBABILITY': False,
            'COPY': False,
            'GEMM_TASKS': 0,
            'X_ROW_STRIDE': 0,
            'Y_ROW_STRIDE': 0,
        }
        specialization = {}
        kernel: Any = _matrix_group if self.specialized else ragged._grouped_dot
        if self.specialized:
            kernel[(self.tasks.shape[0],)](
                a,
                b,
                out,
                self.descriptors,
                self.tasks,
                len(self.signatures),
                tuple(
                    v
                    for s, t in zip(self.signatures, self.tiles)
                    for v in (*s, t.m, t.n, t.k)
                ),
                self.transpose_a,
                self.transpose_b,
                addend,
                alpha,
                beta,
                num_warps=tile.warps,
                num_stages=tile.stages,
            )
            return out
        kernel[(self.tasks.shape[0],)](
            a,
            b,
            a,
            out,
            self.descriptors,
            self.tasks,
            None,
            None,
            self.descriptors,
            tile.m,
            tile.n,
            tile.k,
            False,
            Addend=addend,
            ALPHA=alpha,
            BETA=beta,
            **flags,
            **specialization,
            num_warps=tile.warps,
            num_stages=tile.stages,
        )
        return out
