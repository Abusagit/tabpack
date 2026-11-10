# Homecredit default, epoch-3 performance — 2026-11-05

The TabReD default split is preprocessed using the original TabPack Homecredit recipe from `~/tabpack-archive`: noisy quantile numeric processing, binary extraction, binary-to-categorical conversion and ordinal categorical encoding. It yields 267,645 train rows, 587 numerical and 91 categorical columns, with total one-hot cardinality 407. Training uses binary cross entropy. As in the original recipe, binary-column detection inspects all splits; the numeric transformer and categorical encoder are fitted on train. The preprocessor required the archived code and environment, so the benchmark is not yet self-contained.

All runs use 64 candidates, batch 1024, seed 0, depths 1–4, paper optimizer/dropout ranges, no numeric embeddings, and the sampled models used in Maps-routing. Two full train-only epochs precede a 12-batch profiler capture at the start of epoch 3. There is no validation, early stopping or test evaluation, so all models remain active. Four sequential captures ran on the same RTX A4000 (device 4). Values below are sums of GPU kernel durations per batch attributed to the indicated phases; loss is separate from forward. Synchronized wall-step times include profiler instrumentation and exclude batch gathering. Peaks include train data; no validation/test data is resident.

| Setup | Gather ms | Forward ms | Loss ms | Backward ms | Optimizer ms | Wall step ms | Peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Native fixed width 384 | 10.266 | 15.706 | 0.026 | 9.564 | 20.477 | 58.625 | 2.427 |
| Native diverse | 10.265 | 18.538 | 0.026 | 12.778 | 35.473 | 83.150 | 2.828 |
| Grouped fixed slots 32×32 | 10.265 | 18.845 | 0.026 | 10.478 | 21.411 | 60.600 | 2.801 |
| Grouped fixed slots 64×64 | 10.266 | 18.236 | 0.026 | 9.503 | 19.311 | 56.139 | 2.801 |

The diverse fixed 64 step is 1.48× faster than the diverse native profiled step. Forward attribution includes `aten::cat` about 3.95 ms/batch and `aten::scatter_` about 1.67 ms/batch in all four runs; these are consistent with categorical encoding and concatenation but individual operator costs are not exclusive to that interpretation. 64×64 remains experimental. Single sequential profiler captures on a shared GPU cannot establish a stable whole-training speedup or model quality.

Reports and traces: `results/homecredit-profiler-20261105/`, server `.cache/benchmarks/homecredit-profile-*-20261105/`. Entry points: `dev/pepilica/scripts/profile_homecredit_epoch3.py` and `dev/pepilica/scripts/profile_homecredit_tile64.py`. Config files are the Maps-routing benchmark candidate configs, reused intentionally. The inherited benchmark output metadata was corrected to record 524 warm-up steps (262 per epoch); the actual run already executed two full epochs.


## CompactFixedSlotMLP follow-up

Same Homecredit preprocessing, 64 diverse candidates, two full training epochs and 12 profiler steps at the start of epoch 3 on the same RTX A4000 (device 4), with no evaluation or candidate removal. The benchmark-only bridge shares the existing fixed-slot FP32 master parameters and grouped optimizer, using CompactFixedSlotMLP for the hidden stack; first block and head remain native. Both new runs use 32x32 fixed-slot GEMMs. CUDA profiler phase values are mean kernel durations per step; wall is the median synchronized profiled step, excluding gather. Peaks include resident train data.

| Setup | Gather ms | Forward ms | Loss ms | Backward ms | Optimizer ms | Wall step ms | Peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Native fixed width 384 | 10.266 | 15.706 | 0.026 | 9.564 | 20.477 | 58.625 | 2.427 |
| Native diverse | 10.265 | 18.538 | 0.026 | 12.778 | 35.473 | 83.150 | 2.828 |
| Grouped fixed slots 32x32 | 10.265 | 18.845 | 0.026 | 10.478 | 21.411 | 60.600 | 2.801 |
| Grouped fixed slots 64x64 | 10.266 | 18.236 | 0.026 | 9.503 | 19.311 | 56.139 | 2.800 |
| Compact fixed slots 32x32, unfused | 10.266 | 14.739 | 0.027 | 8.254 | 21.422 | 52.907 | 2.648 |
| Compact fixed slots 32x32, fused | 10.266 | 14.491 | 0.027 | 8.268 | 21.467 | 53.087 | 2.649 |

These are single profiler captures, not full-epoch wall times or quality measurements. Dropout RNG differs between implementations, and fusion changes the epilogue, so this is not a bitwise-identical training trajectory. Separate runs were not simultaneous; resource contention and profiler overhead can affect comparisons. New traces and reports are in `.cache/benchmarks/homecredit-profile-compact-fixed-{unfused,fused}32-20261105/`; run using `dev/pepilica/scripts/profile_homecredit_compact_fixed.py` and set `COMPACT_FIXED_FUSED=0` for the unfused control. The entry point reuses the existing `fixed_grouped` variant flag for setup and adds explicit implementation metadata to these saved reports.


## Unfused compact 64x64 follow-up

The unfused CompactFixedSlotMLP run was repeated with 64x64 tiles for hidden forward, input and weight gradients, and grouped optimizer GEMMs. Bias reductions and native PyTorch operations are unchanged. It used the same GPU, candidates, batch size, two full warm-up epochs, and 12 epoch-3 profiler steps as the prior runs; no evaluation or candidate removal. CUDA columns show mean sums of kernel durations per batch; wall is the median synchronized profiler step, excluding gather. Peaks include resident training data.

| Setup | Gather ms | Forward ms | Loss ms | Backward ms | Optimizer ms | Wall step ms | Peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Native diverse | 10.265 | 18.538 | 0.026 | 12.778 | 35.473 | 83.150 | 2.828 |
| Grouped fixed slots 64x64 | 10.266 | 18.236 | 0.026 | 9.503 | 19.311 | 56.139 | 2.800 |
| Compact fixed slots unfused 32x32 | 10.266 | 14.739 | 0.027 | 8.254 | 21.422 | 52.907 | 2.648 |
| Compact fixed slots unfused 64x64 | 10.266 | 14.021 | 0.026 | 7.211 | 19.212 | 50.049 | 2.647 |

64x64 uses the experimental benchmark wrapper, not a changed production default. It passed all 12 CompactFixedSlotMLP tests under the wrapper. One profiled capture per configuration does not establish full-training speedup or quality. Run artifacts: `.cache/benchmarks/homecredit-profile-compact-fixed-unfused64-20261106/`; entry point `dev/pepilica/scripts/profile_homecredit_compact_fixed_tile64.py` with `COMPACT_FIXED_FUSED=0`.
