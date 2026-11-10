# Width 384 in the fixed vocabulary and saved-config runs

Completed on 2026-11-04 (Europe/Moscow) in `/home/kujbidavs/tabpack-current`.

## Implementation

Width 384 was already supported through the observed aligned-shape path. It is now included in the shared fixed vocabulary: `16, 32, 64, 128, 256, 384, 512, 1024, 2048, 4096`. Linear forward/dX/dW, compact MLP and the Muon matrix planner consume the same list.

A 384-dimensional hidden layer is a full 384x384 linear map. Interpreting the feature axis as 256+128 preserves both off-diagonal cross-connections. The existing 32/64-sized tiles cover 384 exactly, so a 512-wide padded matrix or separate independent models are unnecessary. The relevant tests explicitly zero both diagonal weight blocks and verify the cross-block forward and gradients against independent full-matrix references.

The benchmark now accepts `--config` and records the original JSON snapshot/hash, resolved widths/depths, batch and initialization seed. `--case` remains available for the existing presets. Config-driven hidden-stack runs accept both the benchmark schema and sampled-ensemble ranges. Sampled configs use the same sampler as the full California run and save the resolved per-member hyperparameters. Hidden-stack measurements use geometry and seed only; they do not run the dataset, dropout, head, loss or optimizer specified elsewhere in a full training config.

## Setup

These preliminary runs used the saved `dev/pepilica/configs/churn.json` geometry while awaiting the new config: 64 models, 22/21/21 members of widths 128/256/384, 16 members of each depth 1/2/3/4, batch 256 and seed 0. Inputs and upstream gradients are synthetic. All parameters, activations and gradients are BF16. ReLU and input/weight/bias gradients are included. Packing at the stack boundaries and planning/compilation are outside timing.

Python 3.12.9, Torch 2.15.0+cu132, Triton 3.9.0, CUDA 13.2, RTX A4000 SM86, physical GPU 1 (`GPU-a922981b-b848-d027-b714-170d91808e71`), CPUs 6/7 and two Torch threads. All commands use the locked uv environment. The GPU was idle with over 15 GiB free before running; the allocator cap is 45% of device memory. Clocks were unlocked on this shared host.

## Results

CUDA Graph device timings use seven randomized paired samples, ten warm replays and twenty timed replays per sample. Eager timings are collected separately in seven randomized blocks of fifty synchronized complete forward/backward calls. The short eager samples in the GPU report are diagnostic; the longer eager run is used below.

| Variant | GPU forward, ms | GPU forward+backward, ms | Eager forward+backward, ms |
| --- | ---: | ---: | ---: |
| Native dense | 1.547 | 5.431 | 8.573 |
| Native indexed | 1.251 | 5.334 | 12.134 |
| Compact default | 0.379 | 1.468 | 8.243 |
| Compact width-aware, k_desc | 0.384 | 1.478 | 8.221 |
| Fused compact, same tile policy | 0.278 | 1.387 | 6.639 |

Against native indexed GPU execution, the fixed width-aware compact path is 3.61x faster and fused compact is 3.85x faster. In eager wall time, native dense is the stronger baseline: compact is 1.04x faster and fused compact is 1.29x faster on the recorded medians. Host/launch overhead and variability remain relevant, and these are hidden-stack results rather than whole-training speedups.

The measured width-aware plans have useful FLOPs equal to scheduled FLOPs for forward, dX and dW, including width 384. The profiles contain 117 GPU kernel launches per forward/backward step for native indexed, 36 for compact and 32 for fused compact.

## Validation and artifacts

All 15 targeted checks pass: fixed-vocabulary dispatch across pack compositions, exact tile coverage, 384 cross-connections in BF16/FP16 for forward/dX/dW, fused/unfused connected compact gradients, and shared-vocabulary Muon/zero-update execution. All nine benchmark variants pass output/gradient comparisons in both timing runs, with maximum relative L2 error below 0.02. Ruff, formatting, ty for the changed code, uv lock/environment consistency and `git diff --check` pass.

Raw timings, the config snapshot, source snapshots/hashes, actual three-step CPU/CUDA traces, operator tables and trace summaries are stored remotely in:

- `/home/kujbidavs/tabpack-current/.cache/benchmarks/compact384-config-20261104/`
- `/home/kujbidavs/tabpack-current/.cache/benchmarks/compact384-config-eager50-20261104/`
- `/home/kujbidavs/tabpack-current/.cache/width384-20261104/` (validation and pre-edit backup).

Local mirrors use the corresponding directory names under `results/`. Prior measurements used a different seed and were separate runs, so they are not a controlled before/after timing of vocabulary promotion.

## Reproduce

```bash
uv sync --locked --all-extras --group pepilica
CUDA_VISIBLE_DEVICES=<idle-GPU-UUID> TRITON_DEFAULT_FP_FUSION=1 \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
taskset -c 6,7 uv run --locked --all-extras --group pepilica python \
  python -m dev.pepilica.scripts.benchmark_compact_mlp --config dev/pepilica/configs/churn.json \
  --repeats 7 --replays 20 --eager-steps 5 --memory-fraction 0.45 \
  --output .cache/benchmarks/compact384-repeat

CUDA_VISIBLE_DEVICES=<idle-GPU-UUID> TRITON_DEFAULT_FP_FUSION=1 \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
taskset -c 6,7 uv run --locked --all-extras --group pepilica python \
  python -m dev.pepilica.scripts.benchmark_compact_mlp --config dev/pepilica/configs/churn.json \
  --repeats 7 --eager-only --eager-steps 50 --no-profile \
  --memory-fraction 0.45 --output .cache/benchmarks/compact384-eager-repeat
```

The subsequently supplied `dev/pepilica/configs/diverse_california.json` is run separately; see [the California experiment](benchmark-diverse-california.md). The numbers above are not measurements of that supplied config.
