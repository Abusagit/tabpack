> Layout update (2026-11-10): personal kernels now live in `dev/pepilica/src/tabpack_experimental/` and use the `tabpack_experimental` import namespace. Install with `uv sync --locked --all-extras --group pepilica`. The active standalone checkout is `/home/kujbidavs/tabpack`; checkout names below are historical.

# CUDA backends for the current TabPackage framework

Integration base: upstream `efc02e2`, fetched on 2026-11-03. Both
`yandex-research/tabpack` and `Abusagit/tabpack` pointed to this commit. Upstream
replaced the paper repository with the `tabpackage` library and rewrote history;
this is a port onto the new source tree, not a merge of the old histories.

The server checkout is `/home/kujbidavs/tabpack-current`, branch
`integrate/ragged-20261103`. The paper checkout, environments, experimental files,
datasets and historical measurements remain at `/home/kujbidavs/tabpack`.
Its tracked patch and untracked experimental files were also archived under
`.cache/integration-20261103` before creating this worktree.

## Included work

The opt-in modules live in `tabpack_experimental`:

- `ragged`: grouped GEMM for forward, input gradients and weight gradients;
  the fixed-width bank (16 through 4096, including 384), generic shapes and
  width-aware scheduling.
- `matrix_gemm`: independent rectangular/transposed products, now also supporting
  `alpha * A @ B + beta * addend` in the FP32 GEMM epilogue.
- `ragged_linear`: adapters for the current `tabpackage.nn.LinearPack`, including
  compact-only backward snapshots. Native parameters and state-dictionary keys stay
  unchanged. Output masks now preserve the native result dtype.
- `compact_mlp`: the batch-major `[batch, sum(widths)]` hidden-stack prototype,
  packed dropout, recomputed masks, and optional GEMM/ReLU/dropout fusion.
- `ragged_muon`: compact Newton-Schulz execution inside the current optimizer.
- `benchmark`, `real_data` and `dev/pepilica/scripts/`: state-restored timing, phase events,
  actual profiler traces, padding arithmetic and separate memory measurements.

Triton is optional. Importing the standard `tabpackage` library does not import
these kernels. The old width-bucketed control and the before/after saved-tensor
experiment remain in the paper checkout; they are historical experiments.

## Use with the new framework

```python
from tabpackage.nn import TabPack
from tabpackage.optim import MuonAdamWPack
from tabpackage._training import make_optimizer_pack
from tabpack_experimental.ragged_linear import (
    install_ragged_linears, remove_ragged_linears,
)
from tabpack_experimental.ragged_muon import install_ragged_muon

model = TabPack(
    n_numerical_features=16, categorical_cardinalities=[], d_out=1,
    n_blocks=[1, 2, 4], d_block=[64, 256, 1024], dropout=[0, 0.1, 0.2],
    forward_impl='indexed', pack_size=3,
).cuda()
optimizer = make_optimizer_pack(
    model, type=MuonAdamWPack, adamw_lr=0.001, muon_lr=0.01,
    weight_decay=0.01, muon_skip_zero_updates=False,
)
linears = install_ragged_linears(model, scope='all', schedule='k_desc', tiled=True)
muon = install_ragged_muon(optimizer, model)

# Use the usual BF16-autocast forward/loss/backward/optimizer loop.

muon.disable()
remove_ragged_linears(linears)
```

Install after moving the model to its final device and before warmup. A hidden-only
linear scope is also available. Parameters, optimizer references, native dropout,
feature embeddings, depth handling and loss are preserved. Both `dense` and
`indexed` backbone implementations and external member selection are supported.
The adapter is eager CUDA execution, not an export or full-graph compilation API.
The high-level `tabpackage.train` function does not automatically install adapters.

The production adapter still crosses padded/compact boundaries at each linear;
it does not substitute `CompactMLP` for the full framework model. Parameters,
momentum and surrounding activations remain padded. Its custom backward retains
only compact forward-time snapshots, so it no longer checks mutation versions of
the original padded operands. Only first-order gradients are implemented.

## Changes required by upstream

Upstream Muon now evaluates `B = baddbmm(A, A, A, beta=b, alpha=c)` and
`X = baddbmm(X, B, X, beta=a)`. The port performs the scale/add in FP32 before
each BF16 store. It does not reuse the old `bf16(c*A) @ A` rounding sequence.
This also removes the old separate polynomial-combination kernels.

The adapter delegates momentum, Nesterov, weight decay, gradient-mutation policy,
update scaling and parameter updates to the current native `_step_muon` method.
It substitutes only the orthogonalization callable, restoring it even on errors.
Groups with `muon_skip_zero_updates=True` use native execution, preserving its
dynamic member-selection policy; their fallback counter is exposed. The benchmark
uses `False` in both native and ragged variants.

Logical widths may be replaced in a linear adapter; its plans invalidate when
dimensions change. Same-shape checkpoint restoration retains prepared plans.
Muon plans have fixed matrix geometry: disable and rebuild the adapter after
changing widths, removing members, changing device or changing iteration count.
Same-shape checkpoint restoration is supported. Plans are not reentrant across
concurrent streams. Warm all batch sizes/selections before CUDA Graph capture.

The active research checkout targets Torch 2.15.0 and Triton 3.9.0 throughout.
The kernels use Triton 3.9 tuple indexing directly. Earlier Torch 2.7 validation
records below describe the previous port snapshot, not the current environment.

## Reproduce

From `/home/kujbidavs/tabpack-current`, `.python-version` pins Python 3.12.9,
`pyproject.toml` pins Torch 2.15.0 and Triton 3.9.0, and `uv.lock` records the
complete environment. Install and run the same setup with:

```bash
uv sync --locked --all-extras --group pepilica
CUDA_VISIBLE_DEVICES=<idle-GPU-UUID> TRITON_DEFAULT_FP_FUSION=1 \
  uv run --locked --all-extras --group pepilica python -m pytest tests -q
```

All active benchmarks use this `.venv`; no separate foreach environment is needed.

Churn preprocessing is frozen from our earlier train-fitted recipe, with array,
raw-file, split and recipe hashes in `.cache/data/churn.json`. New benchmarks do
not import the old `project`, `lib` or `vendor` packages. To reproduce this
one-time migration of the prepared arrays:

```bash
cd /home/kujbidavs/tabpack
PYTHONPATH=src .venv/bin/python \
  /home/kujbidavs/tabpack/dev/pepilica/scripts/export_paper_data.py \
  --config experiments/tabpack/churn/main/config.json \
  --output /home/kujbidavs/tabpack-current/.cache/data
```

Use a new output path if the frozen arrays already exist. The replay benchmark
uses the new `TabPack`, native optimizer factory and native shuffled-batch helper.
It retains the earlier benchmark's gradient-clearing order (before backward),
which differs from `tabpackage.train` clearing at the end of a step. This affects
peak memory; these are manual-loop execution benchmarks, not high-level adaptive
training or new prediction-quality experiments.

```bash
cd /home/kujbidavs/tabpack-current
CUDA_VISIBLE_DEVICES=<idle-GPU-UUID> TRITON_DEFAULT_FP_FUSION=1 \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
taskset -c 6,7 .venv/bin/python -m dev.pepilica.scripts.benchmark_current \
  --config dev/pepilica/configs/churn.json --steps 10 --repeats 3 \
  --output .cache/benchmarks/current-churn
```

Use `dev/pepilica/configs/skew-99x64-1x1024.json` for the 100-member skewed ensemble. This
compares native, compiled native, ragged linear, ragged Muon and both adapters.
Compilation/plans are warmed; deterministic three-update comparisons and exact
training-mode replay checks precede randomized paired timings. Phase events and
profiler traces are separate from wall timing. Memory collected in the shared
timing process includes inactive adapter workspaces; use separate processes:

```bash
.venv/bin/python -m dev.pepilica.scripts.measure_memory --help
.venv/bin/python -m dev.pepilica.scripts.benchmark_foreach_mm --help
.venv/bin/python -m dev.pepilica.scripts.benchmark_compact_dropout --help
```

Historical speedups and memory peaks from the paper framework are not measurements
of this new framework. Current validation/results are recorded below separately.

## Current Churn replay (2026-11-03)

RTX A4000 (SM86), GPU UUID `GPU-ec74220d-78d7-cc47-69fc-0de1fb580944`,
Python 3.12.9, Torch 2.15.0+cu132, Triton 3.9.0. The frozen Churn split has
6,400 train rows, 7 numerical features and categorical cardinalities 3/2/2/2.
The pack has 64 members, widths 128/256/384, depths 1--4, batch size 256 per
member, BF16 autocast, FP32 parameters/state and sampled per-member dropout/rates.

Three randomized paired samples each contain ten full steps. The benchmark uses
identical restored parameters, optimizer state, batch cursor and RNG outside each
timed block. It restores parameter values without rewriting unchanged static
buffers, avoiding artificial upstream depth-cache invalidation and recompilation.
It also prepares linear selection metadata outside timing. Changing static buffer
values during these timing experiments is rejected; normal checkpoint restoration
is separately covered by the adapter tests.

| Variant | Median step, ms | Paired throughput vs native |
| --- | ---: | ---: |
| Native current framework | 21.852 | 1.000x |
| Compiled native forward + optimizer | 19.858 | 1.098x |
| Ragged linears | 29.187 | 0.749x |
| Ragged Muon | 19.496 | 1.121x |
| Both adapters | 28.242 | 0.775x |

Muon alone reduces median step time by 10.8% (12.1% higher throughput). The linear
adapter regresses on this mixed-width workload in the newer stack. Both remain
independently selectable; the port does not establish a universal speedup or
transfer the old framework's skewed-ensemble results to the new framework.

All four alternatives pass deterministic three-update comparisons against native
execution; all five variants pass exact training-mode state-restored replay.
The final timed compiler-counter delta is empty. The compiled baseline retains
warmed graph breaks; it is not a full-graph compilation claim. Three-step actual
GPU traces and operator tables exist for all five variants. No accuracy or
convergence run is implied.

This is a shared server with unlocked clocks. Another user's resident allocation
remained in place; the selected GPU was idle before the run. The benchmark's
allocator was limited to 10% of device memory (about 1.6 GiB). The largest measured
allocated peak was about 599 MiB. Shared-process peaks include inactive adapter
workspaces and are not isolated memory comparisons.

Accepted artifacts:
`/home/kujbidavs/tabpack-current/.cache/benchmarks/current-churn-20261103-final/`.
The earlier `current-churn-20261103` directory is exploratory: restoration caused
compiler activity during measurement, so its performance numbers are not accepted.

The large 99x64 + 1x1024 replay was deferred: automatic approval review rejected
running it alongside another user's 8.6 GiB GPU allocation because of memory and
contention risk. Its configuration and replay script are ready for a free GPU.

## Initial port validation records

`/home/kujbidavs/tabpack-current/.cache/validation/` contains:

- `torch215-final.log`: 271 tests passed with Torch 2.15.0+cu132,
  Triton 3.9.0 and pytest 9.2.0.
- `torch27-final.log`: 61 linear, Muon and current-framework integration checks
  passed with Torch 2.7.1+cu126 and Triton 3.3.1.
- `ruff-final.log`, `ty-port.log`: the ported modules, scripts and tests pass
  Ruff 0.17.3 and ty 0.0.96. Formatting and `uv lock --check` also pass.
- `requirements-torch215-final.txt`: exact packages in the new environment.

Coverage includes all nine widths, generic rectangles, forward/dX/dW, packed and
fused dropout, CUDA Graph replay, compact saved-storage lifetimes, native AdamW
and Muon policies, unchanged parameter identities, both backbone implementations,
member selection, width replacement and same-shape checkpoint replay. The Churn
benchmark additionally verifies the actual current model/optimizer integration.

At the initial port snapshot, whole-tree `ty` reported 22 diagnostics in
unchanged upstream files, mostly PyTorch `nn.Module.forward` typing and optional
development imports (`ty-full.log`). The port does not edit those native files;
its own checked scope passes.

The active Torch 2.15-only hidden-stack experiment and unified uv setup are described
in [the compact MLP report](benchmark-compact-torch215.md).
