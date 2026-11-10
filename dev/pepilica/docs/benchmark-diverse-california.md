# California Housing with the supplied diverse config

Run date: 2026-11-04 (Europe/Moscow). Checkout: `/home/kujbidavs/tabpack-current`, base commit `efc02e2b4d123bc9b4c54268fb758664de844138`, with uncommitted experimental changes retained.

The exact supplied `dev/pepilica/configs/diverse_california.json` has SHA256 `e64772e6f32adb5fd0f1f34a3155c91534f696c7cb1f54695e5dcbf8608729b5`. It is used for full native training and for a separate compact hidden-stack experiment. The compact hidden stack is not installed as the full training backbone.

## Training setup

64 models, seed 42, batch 256, up to 60 epochs, model patience 16 and ensemble patience 32. Width counts are 32:14, 64:12, 128:10, 256:9, 384:11 and 512:8. Depth counts are 1:22, 2:15, 3:15 and 4:12. All per-model settings are saved in `hyperparameters.json`.

Settings absent from the JSON follow the repository example: cosine numerical embeddings, uniform choices from the width/embedding lists, inclusive uniform integer depths, log-uniform continuous ranges, and dropout zero with probability one half or uniform in [0, 0.7] otherwise (33 members sampled nonzero dropout). Sampling is deterministic with Python Random(42). The script explicitly uses native indexed execution, FP32 parameters, BF16 autocast, MuonAdamWPack, no compilation, and no zero-update skipping in Muon (the CUDA example/default behavior).

Data comes from the cached official TabPack archive, SHA256 `89338c628fed24af03084c9348ca0a5c8ca4f12f8b988a576d9d1711cc661558`. The default repository split contains 13,209 train, 3,303 validation and 4,128 test rows, with eight numerical features and no categorical features. A noisy normal quantile transform and target standardization are fitted only on the training split. The test split is evaluated only after validation-based checkpoint/ensemble selection. Full data hashes, preprocessing objects, target scaling and split indices are saved.

Python 3.12.9, Torch 2.15.0+cu132, Triton 3.9.0, CUDA 13.2, RTX A4000 SM86. GPU UUID `GPU-a922981b-b848-d027-b714-170d91808e71`, CPUs 6/7, two Torch threads, allocator cap 45%, TF32 and BF16 reduced-precision reductions disabled. The locked uv environment is used. Clocks are unlocked on this shared host.

## Training result

| Measurement | Result |
| --- | ---: |
| Validation RMSE, original target units | 0.422807 |
| Test RMSE, original target units | 0.417328 |
| Epochs / updates | 60 / 3,120 |
| Training call wall time | 148.56 s |
| Native training-loop timer | 147.20 s |
| Selected ensemble | 16 checkpoints from 10 distinct model IDs |

Online greedy ensembling selects among checkpoint candidates and uses at most 64 candidates. Different checkpoints of the same model can be selected; the saved model IDs therefore repeat. The best selected checkpoint steps and complete histories are in the report. The training timer includes the epoch-3 profiling/export and checkpoint overhead and member early stopping; it is not an unprofiled fixed-pack throughput estimate.

A one-epoch smoke run passed before the full run. All 3,143 validation-history metric entries are finite. Reloading the saved predictor reproduces the first 1,024 saved test predictions exactly (maximum absolute difference 0).

## Actual epoch-3 training trace

All 64 models were still active. The trace contains 52 complete training compute/update steps; the initial batch gather, validation and checkpoint serialization are outside the profiled region. The checkpoint immediately before this epoch includes model/optimizer state, the first batch, RNG state and resolved hyperparameters for later replay.

| Phase | GPU kernel time per step, ms | Kernel launches per step |
| --- | ---: | ---: |
| forward | 4.128 | 123.00 |
| backward | 6.297 | 79.00 |
| optimizer | 29.669 | 208.00 |
| loss | 0.007 | 3.00 |
| unassigned | 0.015 | 2.96 |

Optimizer kernels account for 74.0% of summed kernel durations. These are profiler observations, not independent unprofiled phase wall timings. CUDA copies/memsets and overlapping activities are reported separately by the trace summary. Improving the hidden stack alone does not remove this optimizer cost.

## Compact hidden-stack comparison

The exact same sampled widths/depths and batch size are used. Parameters, activations and gradients are BF16 in every benchmark variant. Inputs and upstream gradients are synthetic. This experiment contains connected square hidden linears, ReLU and input/weight/bias gradients. It excludes the numerical embedding prefix, task head, loss, dropout and optimizer. Boundary packing and planning/compilation are outside timing. Consequently, these speedups are not complete-training speedups or an accuracy comparison.

CUDA Graph timings use seven randomized paired samples, 10 warm replays and 20 timed replays per sample. Eager timings use seven randomized blocks of 50 synchronized forward/backward calls. A second eager-only run was collected because host timing varied markedly.

| Variant | GPU forward, ms | GPU forward+backward, ms | Eager initial, ms | Eager repeat, ms |
| --- | ---: | ---: | ---: | ---: |
| Native dense | 2.175 | 7.406 | 7.603 | 7.793 |
| Native indexed | 1.566 | 7.055 | 7.348 | 11.954 |
| Compact default | 0.312 | 1.222 | 7.022 | 8.261 |
| Compact width-aware, k_desc | 0.319 | 1.250 | 7.203 | 8.324 |
| Fused compact, same tile policy | 0.250 | 1.168 | 3.624 | 6.720 |

The fixed width-aware compact path is 5.64x faster than native indexed GPU forward/backward; fused compact is 6.04x faster. GPU sample ranges are stable (native indexed 7.041–7.064 ms; fused 1.162–1.181 ms). Eager samples show substantial variability, so neither the initial eager median nor the GPU ratio is presented as a stable whole-training gain. Both eager runs are retained.

Analytical forward GEMM arithmetic falls from 19.462 GFLOPs for indexed padded execution to 5.235 GFLOPs of useful work, a 3.72x reduction (73.1% padding). The tiled compact plans schedule exactly the useful FLOPs for forward, dX and dW, including width 384. Three-step traces record 117 kernel launches per native indexed forward/backward, 36 for compact and 32 for fused compact.

All nine variants pass connected output/gradient checks in both runs; maximum relative L2 error in the GPU run is 0.001013 or less. Source/config snapshot hashes were verified, and the sampled hyperparameter lists match the training run exactly. Width-384 dispatch/cross-connection checks are documented in [the width-384 report](benchmark-384.md).

## Artifacts and reproduction

Remote result directories under `/home/kujbidavs/tabpack-current/.cache/benchmarks/`:

- `diverse-california-20261104/`: report, full histories, config, hyperparameters, data hashes, preprocessor, predictions, `predictor.pt`, `profile_start.pt`, `epoch03-trace.json`, operator table, trace summary and source snapshots.
- `compact-diverse-california-20261104/`: device/eager samples, output/gradient checks, memory accounting, tile plans, actual three-step trace, operator table and source snapshots.
- `compact-diverse-california-eager-20261104/`: independent eager repeat and its full snapshots.
- `diverse-california-smoke-20261104/`: one-epoch validation run.

Local copies are under `results/` with the same names for the three main runs. The 516 MiB replay checkpoint and 64 MiB predictor remain on the server; all other main-run artifacts are mirrored locally.

```bash
uv sync --locked --all-extras --group pepilica
# Set CUDA_VISIBLE_DEVICES to a currently idle GPU UUID before each run.
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TRITON_DEFAULT_FP_FUSION=1

taskset -c 6,7 uv run --locked --all-extras --group pepilica python -u \
  python -m dev.pepilica.scripts.run_diverse_california --config dev/pepilica/configs/diverse_california.json \
  --output .cache/benchmarks/diverse-california-repeat

taskset -c 6,7 uv run --locked --all-extras --group pepilica python -u \
  python -m dev.pepilica.scripts.benchmark_compact_mlp --config dev/pepilica/configs/diverse_california.json \
  --repeats 7 --replays 20 --eager-steps 50 --memory-fraction 0.45 \
  --output .cache/benchmarks/compact-diverse-california-repeat

# Eager-only validation: add --eager-only --no-profile and a fresh output path.
```

The training script defaults to the cached archive path on this host; `--data-archive` and `--data-root` override it. Output directories must be new. `--max-epochs 1 --no-profile` runs a smoke check without altering the supplied JSON. The benchmark reads both the earlier benchmark-config schema and this sampled-ensemble schema, but only uses resolved geometry/seed for hidden-stack execution.

## Full training comparison — 2026-11-04

These runs use the same sampled models, preprocessing, seed 42 and independently fixed per-model minibatch orders, with profiling disabled. The fresh native row is the comparable baseline; the earlier 148.56 s run included profiling. All four rows ran 60 epochs / 3,120 updates.

| Variant | Total, s | Warm epoch, s | Peak allocated, GiB | Val RMSE | Test RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native eager | 137.60 | 2.574 | 1.324 | 0.424254 | 0.416307 |
| Native compiled | 214.53 | 2.335 | 1.462 | 0.423372 | 0.416063 |
| Compact | 546.39 | 2.340 | 1.252 | 0.421786 | 0.414582 |
| Fused compact | 1227.04 | 2.327 | 1.253 | 0.421786 | 0.414582 |

Warm time is the median duration of epochs 4–12, with all 64 members active, including validation/checkpoint work. Total is training-call wall time, including compilation and layout changes from early stopping; preprocessing and final export are excluded. Existing disk caches were retained; these are not fresh-cache installation timings.

Compact/fused execution keeps the native first block, output head and padded FP32 master parameters/optimizer state. The remaining hidden blocks share compact inter-layer activations; weight/gradient and boundary transfers are included. Compact and fused each made 3,360 backbone calls with zero fallbacks and 23 geometry rebuilds. Changing layouts triggers Triton recompilation, which dominates its total time. Native compilation uses the framework's model and Muon Newton–Schulz flags; it created 31 graphs and reached the default recompilation limit of 8, causing partial eager fallback.

These are single-seed quality results. Artifacts are under `/home/kujbidavs/tabpack-current/.cache/benchmarks/california-comparison-20261104/`. Reproduce each variant with `dev/pepilica/scripts/run_diverse_california.py --variant NAME --no-profile --independent-shuffle`, the supplied config and a fresh output directory.

Fused compact improves warmed epoch time by 1.11x over native eager, but full-run time is 8.92x longer because layout recompilation dominates. Fusion provides no measured end-to-end speedup in this setup. Peak allocated memory is about 5.4% lower than native eager.

All five integration/regression checks pass. Reloaded predictors reproduce saved test predictions exactly, and compact/fused exported predictions are bitwise identical. Raw reports, histories, samples and source snapshots are mirrored locally under `results/california-comparison-20261104/`; predictor files remain on the server.

## Fixed-slot forward prototype

Synthetic BF16 test on the same RTX A4000, batch 128, 12 candidate widths 32–384, with active sets shrinking from 12 to 8 to 4. Slots, descriptors and activation stride stay fixed; a compact tile list contains only active models. This is a forward-only prototype, outside the training pipeline; it has no backward, dropout, optimizer or early-stopping integration. The current-path baseline uses width-aware tiles and repacks into a changing compact layout.

| Active models | Fixed first call, s | Changing layout first call, s | Fixed warm forward, µs | Changing layout warm forward, µs |
| ---: | ---: | ---: | ---: | ---: |
| 12 | 2.485 | 3.312 | 13.50 | 8.89 |
| 8 | 0.00017 | 3.338 | 10.48 | 7.51 |
| 4 | 0.00008 | 3.305 | 11.67 | 8.28 |

Each configuration's first call is timed with a fresh isolated Triton cache for this experiment; the fixed kernel reuses its first specialization on the two subsequent layouts. Warm times are medians of seven 100-replay CUDA Graph samples. Setup/repacking is excluded from warm timings and recorded separately in the raw report. The fixed kernel uses a uniform 32×32 tile and a runtime reduction length, whereas the baseline uses its existing width-aware specialized tiles.

At four active models, fixed slots retain all 1,920 activation columns versus 480 with a repacked layout, and all 489,472 weight elements versus 87,040 active weight elements. Three tests cover arbitrary non-contiguous active sets, correctness and unchanged inactive outputs. Raw results and the exact benchmark source are under `results/fixed-slots-prototype-20261104/`; reproduce with `dev/pepilica/scripts/prototype_fixed_slots.py --output <new-directory>`, selecting an idle CUDA device and a fresh `TRITON_CACHE_DIR`.

The fixed-slot prototype now also has first-order autograd for input, weight and bias gradients. Inactive output and gradient slots are zero; a prepared active-set plan must remain unchanged until its backward completes. Eleven CUDA checks pass for BF16/FP16, uneven widths and batches, empty/non-contiguous active sets, frozen inputs and overlapping forwards. This remains a standalone linear prototype: optimizer integration, FP32 master weights, depth identity paths, dropout and full-training performance are not yet implemented or measured.

The fixed-slot prototype now has a correctness-first Muon/AdamW wrapper: each model owns native optimizer state and views into stable weight/bias storage; `step(active)` calls only selected optimizers. Biases follow the native no-weight-decay policy. A saved state can be restored, and 13 CUDA tests pass including inactive-state freezing, reactivation and checkpoint continuation. This version uses per-model Python/optimizer calls, has not been performance benchmarked, and is not yet integrated into full TabPack training.

## Fixed-slot MLP training experiment

A synthetic hidden-stack comparison uses 12 candidate widths 32–384, depths 1–3, batch 128, FP32 master weights, BF16 computation and per-member dropout 0 or 0.3. All variants use Muon/AdamW. The fixed path retains all model slots and calls one optimizer per active model per layer; native and changing-layout paths physically shrink the pack. Each case starts from a separate fresh state. These are synthetic steps with supplied output gradients, excluding embeddings, head, supervised loss and validation.

| Active models | Native step, ms | Changing-layout step, ms | Fixed-slot step, ms | Native / fixed peak, MiB |
| ---: | ---: | ---: | ---: | ---: |
| 12 | 12.44 | 15.22 | 26.62 | 92.34 / 28.66 |
| 8 | 11.59 | 15.88 | 20.08 | 63.52 / 27.86 |
| 4 | 12.90 | 15.63 | 12.38 | 21.62 / 24.36 |

Values are medians of five blocks of three synchronized eager steps, in fixed execution order on a shared GPU; differences near measurement noise are inconclusive. The separately synchronized optimizer phase at 12 active members is 5.06 ms native versus 21.28 ms fixed. On a fresh isolated Triton cache, forward/dX/dW/bias-gradient each had one fixed-slot specialization across these three membership sizes, while the changing-layout path produced nine specialized GEMM cache entries. First complete steps include initialization and compilation: native 0.784/0.033/0.086 s, changing 14.474/11.610/12.441 s, fixed 1.182/0.040/0.343 s for 12/8/4 members respectively. These first steps include allocation and plan creation, not compilation alone. The fixed path currently casts the entire fixed weight/bias buffers and performs dropout/masking across all slots, including inactive ones.

The full California run completed with the supplied config. It uses the same data, sampled hyperparameters and deterministic minibatch schedule as the native comparison; dropout streams differ across implementations. The first block, embeddings, head and their optimizer remain native. The bridge synchronizes fixed hidden weights into native checkpoints before validation; training-call timing includes those copies.

| California variant | Training call, s | Warm epoch, s | Peak allocated, GiB | Val RMSE | Test RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native eager | 137.60 | 2.574 | 1.324 | 0.424254 | 0.416307 |
| Fixed slots | 244.91 | 4.718 | 1.140 | 0.422153 | 0.415120 |

Both ran 60 epochs / 3,120 updates. Training-call time excludes preprocessing and final predictor export; warm epochs are the median of epochs 4–12 with 64 models active. Fixed was 1.78× slower end to end, and its peak allocated CUDA tensor memory was 13.9% lower. The earlier changing-layout compact run took 546.39 s, but compilation caches differ: the fixed run followed a smoke test. Quality values are single-seed results and do not establish a quality advantage.

Fixed hidden execution ran 3,360 times with zero native fallbacks. The transfer-plan cache rebuilt 72 times for combinations of membership and batch size; these are metadata rebuilds, not 72 kernel recompilations. The cache contained four forward specializations for the four train/evaluation batch sizes and two each for the three gradient kernels. The first block and native padded hidden checkpoint copies are retained. Full-slot weight casts, dropout and masking still perform work for inactive slots, even though matmuls and fixed hidden optimizer updates select active members only.

Twenty-one CUDA checks pass, including independent multi-step updates, dropout rates/reproducibility, model-removal and optimizer-state freezing, inference-to-training cache reuse, and native checkpoint synchronization. The reloaded predictor reproduced 1,024 saved test predictions exactly. Config/data/hyperparameter/source hashes were checked against the native run, along with the initial minibatch hashes. These results supersede the earlier standalone-prototype limitations above.

Artifacts: `results/fixed-slot-training-20261104/` and `results/fixed-slot-california-20261104/` locally, with corresponding directories under `.cache/benchmarks/` on the server. The saved predictor remains on the server. Reproduce with `dev/pepilica/scripts/run_diverse_california.py --variant fixed --no-profile --independent-shuffle` using the supplied config and a fresh output directory.

## Grouped fixed-slot optimizer

The fixed-slot hidden stack now has a grouped Muon/AdamW optimizer. It retains stable parameter and state slots and schedules only active models in shared Triton launches, including the Newton–Schulz matrix multiplications. Bias AdamW uses each model's own step counter and hyperparameters. Inactive parameters and optimizer state stay unchanged; their storage remains allocated. The first block, embeddings, head and their optimizer remain native.

On the RTX A4000, an isolated optimizer-step benchmark with 12 candidates of widths 32–384 produced the following medians in milliseconds (seven randomized blocks of five synchronized eager steps):

| Active models | Native padded | Per-model fixed slots | Grouped fixed slots |
| ---: | ---: | ---: | ---: |
| 12 | 1.777 | 10.628 | 0.571 |
| 8 | 1.208 | 7.060 | 0.515 |
| 4 | 0.978 | 3.502 | 0.507 |

In a separate synthetic hidden-stack benchmark (batch 128, depths 1–3, dropout 0 or 0.3), median forward/backward/optimizer step times were:

| Active models | Native, ms | Per-model fixed, ms | Grouped fixed, ms | Grouped peak, MiB |
| ---: | ---: | ---: | ---: | ---: |
| 12 | 11.53 | 30.38 | 4.54 | 35.64 |
| 8 | 13.73 | 20.32 | 10.68 | 35.63 |
| 4 | 13.69 | 14.32 | 10.73 | 35.61 |

These are five blocks of three synchronized steps, in fixed variant order on a shared machine. The samples were noisy under CPU contention; the optimizer-only and full-step measurements are not full-model training results. The grouped path retains extra state and BF16 workspaces for all fixed slots.

The same 60-epoch California experiment gives:

| Variant | Training call, s | Warm epoch, s | Peak allocated, GiB | Val RMSE | Test RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native eager | 137.60 | 2.574 | 1.324 | 0.424254 | 0.416307 |
| Per-model fixed slots | 244.91 | 4.718 | 1.140 | 0.422153 | 0.415120 |
| Grouped fixed slots | 79.29 | 1.330 | 1.285 | 0.421561 | 0.415556 |

The grouped training call is 3.09× faster than the earlier per-model fixed run and 1.74× faster than the earlier native run. These are single runs on the same physical GPU at different times, with differing cache and shared-machine conditions; they are not a controlled repeated speedup measurement. The grouped run follows a one-epoch smoke test, as did the earlier fixed run. Config, data, sampled hyperparameters and the first three minibatch hashes match; dropout streams and numerical trajectories differ, so the single-seed RMSEs do not establish a quality difference. The grouped run executed 3,360 compact calls with no native fallbacks and 80 transfer-plan metadata rebuilds. It used 12.7% more peak allocated memory than per-model fixed slots and 3.0% less than native.

Twenty-nine CUDA tests pass, including independent multi-step optimizer comparisons, zero and tiny gradients, active-set changes, checkpoint restoration, autograd mutation checks and full-model integration. In a two-update full-model comparison against native, small BF16 gradient differences were amplified by Muon: the observed first-layer weight relative L2 difference was about 0.9%, and the test bounds it at 2% along with an absolute-error bound. The exported predictor exactly reproduced the first 1,024 saved test predictions. Ruff checks passed.

Raw measurements: `results/grouped-fixed-optimizer-20261104/report.json`, `results/grouped-fixed-training-20261104/report.json`, and `results/grouped-fixed-california-20261104/`; full predictor and matching artifacts are in `.cache/benchmarks/grouped-fixed-california-20261104/` on the server. Reproduce with `dev/pepilica/scripts/benchmark_grouped_fixed_optimizer.py`, `dev/pepilica/scripts/benchmark_fixed_slot_training.py`, and `dev/pepilica/scripts/run_diverse_california.py --variant fixed_grouped --no-profile --independent-shuffle`, selecting an idle GPU and a fresh output directory.

### Native eager rerun

Native eager was rerun on the same device as the grouped run, with the same 0.15 memory fraction and CPU/thread settings. It completed 60 epochs / 3,120 steps in **138.10 s**, with a **2.576 s** median warm epoch, **1.324 GiB** peak allocated, and val/test RMSE **0.424254 / 0.416307**. This is consistent with the earlier native result (137.60 s) and is 1.74× the grouped run's 79.29 s. The rerun and grouped run have identical configs, dataset hashes, sampled hyperparameters, and first three minibatch hashes. They are still separate runs at different times; dropout streams and training trajectories differ. The other process on the GPU was holding memory but had 0% device utilization when the rerun started. Raw results: `results/native-eager-california-rerun-20261104/`; full server artifacts: `.cache/benchmarks/native-eager-california-rerun-20261104/`.

### Grouped fixed-slot epoch 3 trace

A separate three-epoch California run captured all 52 training steps of epoch 3 with the CPU/CUDA profiler enabled, on the same GPU and with the same config, seed, independent shuffle and thread settings as the grouped run above. This run is for analysis; profiler overhead means its timing is not a replacement for the unprofiled benchmark. The full trace is `results/grouped-fixed-california-profile-20261104/epoch03-trace.json`, with `trace-summary.json` and `operators.txt` beside it. Complete artifacts remain under `.cache/benchmarks/grouped-fixed-california-profile-20261104/` on the server.

| Training phase | GPU kernel launches / step | Sum of GPU kernel durations / step |
| --- | ---: | ---: |
| Forward | 62 | 3.417 ms |
| Loss | 3 | 0.006 ms |
| Backward | 75 | 4.777 ms |
| Optimizer | 163 | 13.687 ms |

The optimizer accounts for about 62.5% of summed phase-attributed kernel time. The grouped optimizer's `_mm` Newton–Schulz matmul kernel is the largest named kernel at 45 launches and 5.495 ms per step; native `aten::baddbmm` and `aten::bmm` operations attributed to the optimizer account for 3.537 ms and 1.571 ms per step respectively. Those native operations remain for model parts outside the fixed hidden stack. GPU gaps between recorded activities cover about 27.5% of the profiled GPU activity envelope, but profiler overhead and CPU scheduling can affect these gaps; this is not a hardware-utilization measurement. Kernel-duration sums can overlap and are not wall-clock phase timings. An older native trace exists, but it was captured under different run conditions and is not a paired comparison.

### Ampere CUTLASS grouped Muon prototype

The opt-in `dev/pepilica/scripts/run_cutlass_california.py` replaces only the grouped fixed-slot optimizer's Newton–Schulz `_mm` calls with an Ampere CUTLASS BF16 grouped GEMM. The fixed-slot MLP, optimizer's other operations, first block and head are unchanged. CUTLASS uses a device-only persistent group scheduler and 64×64×32 tiles, with cached per-active-set pointer/shape metadata; the Triton comparison uses its existing explicit tile tasks and 32×32×32 tiles. Inputs and outputs are BF16 with FP32 accumulation. This prototype requires square widths divisible by 8, targets SM86 and errors for unsupported widths; it is not the default backend. The results compare complete implementations, so they do not isolate the effect of scheduler versus tile shape or implementation language.

On the RTX A4000, the nine individual Newton–Schulz GEMM cases for the actual California active widths (42, 27 and 12 models by layer) were 2.3–3.1× faster with CUTLASS in this run. For example, the first layer's `X Xᵀ` took 0.222 ms Triton versus 0.072 ms CUTLASS. Timings are medians of seven randomized blocks of 20 calls using CUDA events; Python launch delays can still contribute to the measured event interval. They exclude one-time plan creation.

| Training measurement | Triton | CUTLASS |
| --- | ---: | ---: |
| One-epoch training call including setup | 7.972 s | 7.899 s |
| Warm epoch, pair 1 | 1.783 s | 1.171 s |
| Warm epoch, pair 2 (reverse order) | 1.392 s | 1.168 s |

The warmed measurements are the second epoch after one warm-up epoch, including validation. Both implementations use the same 52-step epoch and were run on GPU 0 with matching seeds, config, sampled hyperparameters and minibatch hashes. The two pairs ran in opposite orders but were not interleaved step by step; they are too few and noisy to claim a stable end-to-end speedup. One-epoch peak allocated memory was 1.052 GiB for both. The selected one-epoch val/test RMSE was 0.585835/0.592895 for both; by epoch 2 the results differ slightly (Triton 0.505401/0.503898, CUTLASS 0.504258/0.503733). The BF16 kernels differ in rounding and Muon's subsequent nonlinear iterations can amplify small differences; no longer-run quality equivalence is established.

Nine direct grouped-GEMM correctness tests passed across different active sets and all three Newton–Schulz matmul forms; eleven existing optimizer and fixed-slot integration tests passed with this backend (the unrelated unaligned-width test was deselected because this prototype rejects those widths). Ruff passed for the Python prototype. Build provenance, raw measurements and configs are in `results/cutlass-grouped-20261104/`; full artifacts are under the corresponding `.cache/benchmarks/` directories on the server. The tested CUTLASS source is commit `37b77875c76384f175a88381ba04e865eb2715c6` in `.cache/cutlass-source`. Build from the repository root using `sh experiments/cutlass_grouped/build.sh`; then run `dev/pepilica/scripts/benchmark_cutlass_muon.py` for GEMMs or `dev/pepilica/scripts/run_cutlass_california.py --variant fixed_grouped --no-profile --independent-shuffle` for training. The training runner's existing report names its variant `fixed_grouped`; the CUTLASS wrapper and source hashes distinguish this experiment.

### Grouped fixed slots with 1024-wide models

A separate run adds 1024 to the width choices, with the other California settings unchanged. Seed 42 sampled 13 models of width 1024 among 64; changing the width choices also changes the sampled ensemble, so it is not a one-factor comparison of the same models. On RTX A4000 device 1, after a one-epoch smoke test, the full 60-epoch / 3,120-step grouped fixed-slot run took **213.85 s** for the training call, with a **4.344 s** median for epochs 4–12, **4.457 GiB** peak allocated CUDA memory, validation RMSE **0.422742**, and test RMSE **0.414900**. The run had 3,360 compact calls, no native fallbacks, and 80 transfer-plan metadata rebuilds. The GPU memory fraction was 0.40; peak reserved was 5.344 GiB. Timings are a single run on a shared GPU, exclude preprocessing and predictor export, and are not directly attributable to width alone.

Config: `dev/pepilica/configs/diverse_california_1024.json`; raw metrics: `results/grouped-california-1024-20261104/`. Full artifacts including the predictor remain under `.cache/benchmarks/grouped-california-1024-20261104/` on the server.

### Native uniform width 384 versus mixed widths

Vanilla TabPack was run with all 64 model hidden widths set to 384, using `dev/pepilica/configs/diverse_california_uniform384.json`. The other config fields are unchanged from the 1024-width experiment. Following a one-epoch smoke test, it ran 60 epochs / 3,120 steps on RTX A4000 device 2. The grouped fixed-slot runs below were performed earlier on a different RTX A4000 (device 1).

| Setup | Training call, s | Warm epoch, s | Peak allocated, GiB | Val RMSE | Test RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native, all widths 384 | 86.69 | 1.654 | 0.883 | 0.422094 | 0.414216 |
| Grouped fixed slots, widths 32–512 | 79.29 | 1.330 | 1.285 | 0.421561 | 0.415556 |
| Grouped fixed slots, widths 32–1024 | 213.85 | 4.344 | 4.457 | 0.422742 | 0.414900 |

Warm epoch is the median of epochs 4–12. Training call excludes preprocessing and final predictor export. Each is a single run on a shared GPU; different device, run time, sampled widths and potentially other sampled hyperparameters mean this is a comparison of whole configurations, not an isolated backend speedup or evidence of a quality improvement. The 1024 setup sampled 13 width-1024 models. Native uniform report and history are in `results/native-california-uniform384-20261104/`, with the full artifacts on the server under `.cache/benchmarks/native-california-uniform384-20261104/`.

### Uniform 384 with the width-specialized compact kernel

The earlier changing-layout compact backbone (`--variant compact`, the 384 width-specialized ragged kernel) was run on the same uniform-384 config and the same GPU as the native row. Following its own one-epoch smoke run, it completed 60 epochs / 3,120 steps: training call **507.59 s**, warm epoch median of epochs 4–12 **1.514 s**, peak allocated **0.856 GiB**, validation RMSE **0.422403**, test RMSE **0.415825**. Native was 86.69 s, 1.654 s, 0.883 GiB, and 0.422094/0.414216 respectively. Both have matching config, dataset hashes, sampled hyperparameters and first three minibatch hashes; dropout and numerical trajectories differ. This single-run comparison was not interleaved and compilation caches and shared-machine conditions can differ.

The compact bridge recorded 3,360 compact calls, zero fallbacks and 22 geometry rebuilds. Geometry rebuilds are not compilation counts. The log shows roughly 20-second stalls following model removals, consistent with changing-layout specializations; these inflate training-call time but not the selected early warm-epoch metric. The fixed-slot backend is a different implementation. Results: `results/compact-california-uniform384-20261104/`; complete server artifacts under `.cache/benchmarks/compact-california-uniform384-20261104/`.

### Grouped fixed slots, uniform width 384

The grouped fixed-slot backend was run on the uniform-384 config, on the same RTX A4000 as the native and changing-layout compact runs, after a one-epoch smoke run. It completed 60 epochs / 3,120 steps: training call **95.97 s**, warm epoch median of epochs 4–12 **1.763 s**, peak allocated **1.221 GiB**, validation RMSE **0.421808**, and test RMSE **0.416605**. The corresponding native values were 86.69 s, 1.654 s, 0.883 GiB and 0.422094/0.414216; changing-layout compact values were 507.59 s, 1.514 s, 0.856 GiB and 0.422403/0.415825. All three runs match config, dataset hashes, sampled hyperparameters and first three minibatch hashes, but dropout streams and numerical trajectories differ. These are separate single runs on a shared GPU, not a repeated benchmark.

The grouped fixed-slot bridge made 3,360 compact calls with no native fallbacks and 96 transfer-plan metadata rebuilds; these rebuilds are not compilation counts. No long pack-change stalls appeared in the run log. The grouped path retains additional FP32 optimizer state and BF16 workspaces. Full server artifacts are under `.cache/benchmarks/grouped-fixed-uniform384-20261104/` and summarized outputs under `results/grouped-fixed-uniform384-20261104/`.

### Optimizer cleanup — 2026-11-05

The per-model fixed-slot optimizer backend and `--variant fixed` were removed. The grouped implementation remains available as `--variant fixed_grouped`; references to per-model fixed slots and commands using `--variant fixed` above describe historical experiments and are no longer reproducible with the current entry point. Grouped optimizer tests now compare directly against native padded TabPack; no per-model optimizer implementation is retained.

### Maps Routing training-step benchmark — 2026-11-05

The available TabReD default split has 160,019 train rows, 59,975 validation rows, and 59,951 test rows. The original TabPack experiment's `build_dataset` recipe was used (no numeric transform, binary extraction and ordinal categorical encoding); its implementation detects numeric binary features across splits, as in the original code. Labels are standardized using the training split. Paper-style TabPack uses 64 models, width 384, depths 1–4, batch 1024, no numerical embeddings, paper optimizer/dropout ranges, and seed 0. The two diverse setups sample from [64, 128, 256, 384, 512] with the same sampled hyperparameters for both backends. Sampling uses the benchmark's Python random generator, rather than claiming an exact match to the paper's Optuna random draws.

For this performance-only experiment, each model trained on independent shuffled train batches for two complete epochs, with no validation, test evaluation or early stopping; all 64 models remain active. Times below are medians of 12 batches at the start of epoch 3 on an RTX A4000, using CUDA events around forward+loss, backward, and optimizer, and synchronized host wall time for a full step. Batch gathering and zeroing are not in the event phases; wall time includes zeroing but excludes gathering. CUDA events can include CPU submission gaps and are not summed GPU kernel durations. Peak allocated memory includes the training data and initial epoch sampling; it excludes validation and test arrays.

| Setup | Forward + loss ms | Backward ms | Optimizer ms | Wall step ms | Peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native paper width 384 | 11.564 | 11.110 | 25.306 | 48.093 | 2.156 |
| Native diverse widths | 14.844 | 15.040 | 43.507 | 73.588 | 2.562 |
| Grouped fixed-slot diverse | 14.410 | 11.248 | 24.234 | 50.084 | 2.531 |

The diverse fixed-slot step is 1.47× faster than native diverse in this run. Models remain active because there was no early-stopping evaluation; checkpointing, validation, final predictor export and full-training wall time are not measured. This is one sequential pass per variant on a shared GPU, not interleaved repeated independent runs. Raw outputs are in `results/maps-performance-20261105/`, with server runs under `.cache/benchmarks/maps-epoch3-*-20261105/`. Preprocessing used the original experiment code in `~/tabpack-archive/src/lib/data.py` with the archive's Python environment; `dev/pepilica/scripts/preprocess_maps_paper.py` records the invocation. The experiment is not standalone from that archive until its preprocessing dependency is ported.

### Maps Routing epoch-3 profiler traces

Separate CPU/CUDA profiler runs captured 12 epoch-3 train batches after two epochs without validation or early stopping. Phase labels include batch gathering, forward, loss, backward, optimizer and gradient clearing. Sum of GPU kernel durations per batch in ms:

| Phase | Native paper width | Native diverse | Grouped diverse |
| --- | ---: | ---: | ---: |
| Batch gather | 16.723 | 16.737 | 16.715 |
| Forward | 10.654 | 13.658 | 13.880 |
| Backward | 9.648 | 13.016 | 10.747 |
| Optimizer | 20.580 | 35.841 | 21.496 |

The first two traces ran on RTX A4000 device 2. The grouped run on that device suffered concurrent GPU load and is not suitable for timing comparison; it was repeated on an initially idle RTX A4000 device 3, shown above. These runs were not interleaved or repeated and kernel-duration sums are not end-to-end latency; profiler overhead and asynchronous execution can affect them. The approximately 16.7 ms per batch of GPU-side `aten::index` gathering was excluded from the earlier step timings. Offline dataset preprocessing remains outside the traces. In grouped diverse, `_mm` accounts for 7.565 ms/batch over 45 launches; in native diverse, optimizer `baddbmm` and `bmm` account for 16.193 and 7.857 ms/batch. Operator attribution can overlap; do not add these to the phase totals.

Traces and summaries are in `results/maps-traces-20261105/` locally and `.cache/benchmarks/maps-profile-*-20261105/` on the server. The instrumented entry point is `dev/pepilica/scripts/profile_maps_epoch3.py`.

### Nsight Systems: Maps Routing diverse widths

Nsight Systems 2026.1.3 captured 12 batches at the start of epoch 3 after two no-validation training epochs, all 64 models active, for native and grouped fixed slots on the same RTX A4000 (GPU 3). Capture uses CUDA profiler start/stop and NVTX gather/forward/loss/backward/optimizer ranges, CUDA and NVTX tracing, and disables CPU sampling. The benchmark explicitly synchronizes after gather and every step, so CUDA synchronization in the API timeline is a measurement choice and cannot be treated as an inherent pipeline bottleneck. No evaluation or test metrics were measured.

GPU kernel sums: the `aten::index` kernel family occupied 200.67 ms across 12 native batches and 200.38 ms across 12 grouped batches (about 16.7 ms/batch each; 21.2% and 26.5% of summed GPU kernel durations respectively). In the grouped trace, `_mm` occupied 91.86 ms across 540 launches, about 7.65 ms per batch and 45 launches per batch. Nsight's kernel summary aggregates all phases, while NVTX summaries represent host-side annotation durations; neither is a direct breakdown of end-to-end GPU wall latency. Individual summed durations may overlap. Event-based synchronized step medians excluding gathering were 73.66 ms native and 50.37 ms grouped under Nsight; use unprofiled benchmarks for speed comparisons.

Reports are `.cache/benchmarks/nsys-maps-diverse-native-20261105.nsys-rep` and `.cache/benchmarks/nsys-maps-diverse-grouped-20261105.nsys-rep`; copies and textual summaries are in `results/maps-nsys-20261105/`. Reproduce with `dev/pepilica/scripts/nsys_maps_epoch3.py` and `nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop` on an idle device. Offline preprocessing is outside the capture.

### Maps Routing forward component microbenchmarks

Using the sampled Maps Routing diverse model widths/depths and batch 1024, separate BF16 synthetic-tensor microbenchmarks compared native padded `torch.baddbmm` with `FixedSlotPlan` for the three connected hidden layers. Both paths receive equivalent prebuilt inputs, weights and bias; the fixed-slot call allocates and zeros its fixed-layout output. Native selection, fixed boundary transfers, activation, dropout and model execution are excluded. Per-model output relative L2 error was below 0.000051. GPU-event medians over seven blocks of 20 calls on RTX A4000 device 3 were:

| Hidden layer | Active models | Native GEMM ms | Fixed-slot GEMM ms |
| --- | ---: | ---: | ---: |
| 1 | 45 | 0.726 | 0.759 |
| 2 | 30 | 0.475 | 0.517 |
| 3 | 18 | 0.287 | 0.315 |

The native and fixed microbenchmark inputs have different layouts, as required by their APIs; preprocessing both layouts is excluded. Runs were sequential in randomized order per layer, not interleaved block-by-block, and samples vary. GPU events may include host submission delays. These numbers are standalone microbenchmarks and should not be compared directly against Nsight's kernel-only sums.

A BF16 cast of FP32 compact activations `[1024, 16192]` took 0.258 ms; a compact weight buffer of 5,754,880 FP32 values took 0.091 ms. The shared padded input projection (`[64,1024,1000]` times `[64,1000,512]`) measured 1.003 ms for the FP32 input cast, 0.503 ms for the weight cast, 2.003 ms for the BF16 matmul and 3.239 ms for casts plus matmul, under these synthetic conditions. Both current variants use the same first-projection implementation; no ragged first-projection prototype or speedup is measured. First-layer times vary relative to the full-model trace, so they cannot replace it as a whole-forward measurement.

Results: `results/maps-forward-pieces-20261105/report.json`; server copy `.cache/benchmarks/maps-forward-pieces-20261105/`. Source is `dev/pepilica/scripts/benchmark_maps_forward_pieces.py` (formatted and cleaned after measurement with the same benchmark operations).

### Maps Routing hidden-forward tile-size comparison

A paired synthetic-tensor benchmark used the Maps-routing diverse widths and depths, batch 1024, the same fixed-slot forward kernel, reduction tile 32, four warps, width-descending task ordering, and BF16 inputs. It compared output tiles of 32×32 and 64×64 against preselected padded native `baddbmm`. Fixed-slot timings include allocating and zeroing the fixed-layout output; native selection and all boundary packing are excluded. Nine randomized-order blocks of 20 calls per variant used CUDA events on RTX A4000 device 3.

| Hidden layer | Active models | Native ms | Fixed 32×32 ms | Fixed 64×64 ms |
| --- | ---: | ---: | ---: | ---: |
| 1 | 45 | 0.712 | 0.745 | 0.475 |
| 2 | 30 | 0.477 | 0.520 | 0.346 |
| 3 | 18 | 0.284 | 0.303 | 0.217 |

The 64×64 version was 1.40–1.57× faster than 32×32 in this microbenchmark. Both tile variants matched valid native outputs with maximum per-member relative L2 error below 0.000051. This experiment only changes the benchmark invocation and task construction; the production fixed-slot forward/backward kernels remain unchanged. It does not measure backward, full forward, training, or the shared first projection. GPU-event timing may include host launch gaps; samples are in `results/maps-forward-tiles-20261105/report.json` and `.cache/benchmarks/maps-forward-tiles-20261105/report.json` on the server. Reproduce with `dev/pepilica/scripts/benchmark_maps_forward_tiles.py` using the saved diverse epoch-3 report.

### Maps Routing: four epoch-3 profiler captures

Sequential CPU/CUDA profiler captures on the same RTX A4000 (device 6) covered 12 epoch-3 batches after two train-only epochs, with all 64 models active and no evaluation. Values below are summed attributed GPU kernel durations per batch, not phase wall time; loss is separate from forward. Each capture has 12 batches.

| Setup | Gather ms | Forward ms | Loss ms | Backward ms | Optimizer ms | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native width 384 | 16.715 | 10.619 | 0.016 | 9.643 | 20.562 | 2.156 |
| Native diverse | 16.718 | 13.521 | 0.016 | 12.946 | 35.347 | 2.562 |
| Grouped fixed slots, 32×32 | 16.712 | 13.862 | 0.016 | 10.739 | 21.545 | 2.531 |
| Grouped fixed slots, 64×64 | 16.713 | 13.215 | 0.016 | 9.711 | 19.338 | 2.531 |

The 64×64 experiment changes forward, input/weight gradient and optimizer GEMM tiles; bias reductions remain unchanged. The 32 and 64 forward launch counts are both 55/batch and optimizer counts both 122/batch. The optimizer `_mm` accounts for 7.577 ms/batch at 32 and 5.507 ms at 64. Captures are sequential single runs, not repeated/interleaved, and profiling influences timings. Full traces and per-run summaries are at `results/maps-profiler-paired-20261105/` and on the server under `.cache/benchmarks/maps-profile-paired-*-20261105/`. The 64×64 kernel variant is experimental and is installed by `dev/pepilica/scripts/profile_maps_epoch3_tile64_all.py`.
