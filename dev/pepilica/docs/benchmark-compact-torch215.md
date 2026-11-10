# Compact hidden stacks on the unified Torch 2.15 setup

Completed on 2026-11-03 in `/home/kujbidavs/tabpack-current`. This measures the existing compact hidden stack directly. It does not replace the full TabPack backbone in Churn training.

## Environment and protocol

Python 3.12.9, Torch 2.15.0+cu132, Triton 3.9.0, RTX A4000 (SM86), GPU UUID `GPU-a922981b-b848-d027-b714-170d91808e71`. The GPU was idle with more than 15 GiB free before the run. Other processes were retained; clocks were unlocked. The allocator limit was 45% of device capacity (about 7.2 GiB).

The active project pins Torch 2.15.0 and Triton 3.9.0 in `pyproject.toml`, Python 3.12.9 in `.python-version`, and resolves the full environment in `uv.lock`. Both the measurements and checks use this same `.venv` through `uv run --locked --all-extras --group pepilica`. The old Triton 3.3 tuple compatibility branch was removed.

All models use identical weights, inputs and upstream gradients, BF16 parameters/activations/gradients, batch size 256, ReLU and first-order autograd. Depth-inactive members carry their previous values. Compact activations remain contiguous `[batch, sum(widths)]` between layers. There are no intermediate padded activation conversions.

Both padded controls call the current native `MLPBackbonePack`: `dense` evaluates every member slot, while `indexed` skips depth-inactive members. Shared widths/depths use scalar native settings, avoiding redundant all-true masks for homogeneous packs. All native masks preserve BF16; the obsolete FP32-mask control is no longer used.

CUDA Graph timings use seven randomized paired blocks, ten warm graph replays and twenty timed replays per sample. Forward includes output allocation; forward+backward includes input/weight/bias gradients and clearing/accumulation of existing gradients. Compilation, planning, boundary packing, loss, feature processing, input projection, output head, dropout and optimizer are excluded. No `torch.compile` variant is included in this hidden-stack experiment.

## GPU execution time

Times below are CUDA-event device timings under CUDA Graph replay, not synchronized eager wall time. The fixed width-aware `k_desc` policy is the main compact comparison. The fused variant uses the same policy and fuses GEMM/bias/ReLU; scheduling alternatives are retained in the JSON report rather than selecting a different winner for each row.

| Workload | Native indexed F+B, ms | Compact F+B, ms | Speedup | Fused compact F+B, ms | Fused speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 × 256, depth 3 | 0.593 | 0.396 | 1.50x | 0.376 | 1.58x |
| 64 × 128/256/384, depths 1–4 | 5.350 | 1.502 | 3.56x | 1.395 | 3.83x |
| 9 widths 16–4096, depth 2 | 21.977 | 2.776 | 7.92x | 2.736 | 8.03x |
| 99 × 64 + 1 × 1024, depths 1–4 | 35.125 | 0.776 | 45.27x | 0.723 | 48.61x |

| Workload | Native indexed forward, ms | Compact forward, ms | Fused forward, ms |
| --- | ---: | ---: | ---: |
| 16 × 256, depth 3 | 0.138 | 0.086 | 0.067 |
| 64 × 128/256/384, depths 1–4 | 1.254 | 0.392 | 0.283 |
| 9 widths 16–4096, depth 2 | 3.492 | 0.620 | 0.549 |
| 99 × 64 + 1 × 1024, depths 1–4 | 6.740 | 0.215 | 0.174 |

The heterogeneous cases retain substantial GPU gains on Torch 2.15. The uniform case also improves, but much less. These results do not establish the same speedup for full training: the prefix, head, optimizer and any integration-boundary conversions still need separate measurement.

## Synchronized eager wall time

The short five-step eager samples in the GPU report were variable. A separate run uses seven randomized samples of fifty complete forward/backward calls per variant, synchronized around each block. The same device, environment, weights, seed, dtype and geometry are used. No CUDA Graph capture or profiler is active during these blocks.

| Workload | Native dense, ms | Native indexed, ms | Compact, ms | Fused compact, ms | Compact vs faster native | Fused vs faster native |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform | 4.310 | 4.154 | 5.060 | 5.166 | 0.82x | 0.80x |
| mixed_depth | 8.628 | 11.969 | 8.104 | 6.255 | 1.06x | 1.38x |
| vocabulary | 21.877 | 21.917 | 4.771 | 3.955 | 4.59x | 5.53x |
| skewed | 35.677 | 35.332 | 7.949 | 6.346 | 4.44x | 5.57x |

The small uniform workload regresses in eager mode despite faster GPU execution. The mixed-depth native dense path beats its indexed path in eager wall time, so the eager speedup columns use the faster native implementation. Compact storage has a small median eager advantage there, while fusion improves it further. Wide and skewed workloads retain larger eager gains, but these are much smaller than CUDA Graph speedups. Python/autograd/launch overhead and host variability remain relevant; the data do not establish one universal speedup. All raw samples, including occasional much faster host samples, are retained.

Remote eager artifacts: `/home/kujbidavs/tabpack-current/.cache/benchmarks/compact-torch215-eager50-20261103/`. All 36 output/gradient comparisons pass in this run as well.

```bash
CUDA_VISIBLE_DEVICES=<idle-GPU-UUID> TRITON_DEFAULT_FP_FUSION=1 \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
taskset -c 6,7 uv run --locked --all-extras --group pepilica python \
  python -m dev.pepilica.scripts.benchmark_compact_mlp --case all --repeats 7 \
  --eager-only --eager-steps 50 --no-profile --memory-fraction 0.45 \
  --output .cache/benchmarks/compact-eager-repeat
```

## Storage payloads

These are tensor payload sizes in MiB, not allocator peaks. Gradient payloads equal parameter payloads in this BF16 experiment.

| Workload | Native parameters | Compact parameters | Native boundary input | Compact boundary input |
| --- | ---: | ---: | ---: | ---: |
| uniform | 6.023 | 6.023 | 2.000 | 2.000 |
| mixed_depth | 72.188 | 23.203 | 12.000 | 7.938 |
| vocabulary | 576.141 | 85.364 | 18.000 | 3.992 |
| skewed | 800.781 | 9.960 | 50.000 | 3.594 |

The native stack allocates full-width weights for every member and layer. The standalone compact stack stores only useful matrices for members active at each layer. This storage benefit does not describe the earlier padded-parameter LinearPack adapter.

## Correctness and artifacts

The 62 targeted checks in the locked environment pass, covering connected outputs/gradients, storage handoff without repacking, scalar dropout/RNG and generic/specialized matrix products with all transpose combinations.

All 36 workload/variant combinations pass output and input/weight/bias-gradient comparisons with relative L2 error below 0.02. Maximum observed error by workload:

- uniform: 0.000000.
- mixed_depth: 0.003054.
- vocabulary: 0.000018.
- skewed: 0.011964.

Every workload has an actual three-step CPU/CUDA profiler trace for all nine variants, an operator table and a trace summary. Reports retain all seven raw timing samples, randomized orders, shapes, selected tiles, incoming gradient layouts, source snapshots/hashes and installed packages. Memory fields report parameter/gradient/input/output payloads and incremental transient allocation while variants coexist; they are not isolated per-model peak comparisons.

Remote GPU artifacts: `/home/kujbidavs/tabpack-current/.cache/benchmarks/compact-torch215-unified-20261103-final/`. The earlier directory without `-final` used redundant native tensor settings for homogeneous cases and is exploratory.

## Reproduce

```bash
uv sync --locked --all-extras --group pepilica
CUDA_VISIBLE_DEVICES=<idle-GPU-UUID> TRITON_DEFAULT_FP_FUSION=1 \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
taskset -c 6,7 uv run --locked --all-extras --group pepilica python \
  python -m dev.pepilica.scripts.benchmark_compact_mlp --case all --repeats 7 --replays 20 \
  --memory-fraction 0.45 --output .cache/benchmarks/compact-repeat
```

`--case uniform|mixed_depth|vocabulary|skewed` selects one geometry. Current scripts also support `--eager-only --eager-steps 50` for longer synchronized eager wall measurements.

## Validation of the unified setup

`uv sync --locked --all-extras --group pepilica --check` reports no changes, `uv pip check` passes, and Ruff formatting/lint and ty checks pass for the ported modules/scripts/tests. The same locked environment ran all benchmark variants and the 62 targeted checks. Whole-tree ty still reports 16 diagnostics in unchanged upstream code; these are recorded separately under `.cache/torch215-unified/ty-full.log`.
