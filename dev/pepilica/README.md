# Active fixed-slot TabPack

This package contains one experimental model: **`ActiveFixedSlotMLP`**.
Candidate parameters and optimizer states have compact, permanent slots.
Activations use `[batch, sum(active widths)]`; tile descriptors execute only
selected candidates. Widths, depths and dropout can differ per candidate.

The native TabPack adapter supports two variants:

| Variant | Grouped layers | Native layers |
|---|---|---|
| `active_grouped_first` (default) | Rectangular first layer and square hidden layers | Numerical embeddings and output head |
| `active_square_matrices_only` | Square hidden layers | First block, numerical embeddings and output head |

Grouped blocks use linear+bias followed by a packed activation/dropout/carry
kernel. Backward replays dropout from its seed and computes grouped dX/dW/db.
Grouped Muon updates matrices; AdamW updates biases with independent candidate
hyperparameters and state. There is no full-slot activation variant or optional
GEMM/ReLU/dropout fusion. Padded native weight copies remain for framework
evaluation/checkpoint compatibility; gradients and optimizer workspaces still
reserve all candidate parameter slots.

## Code

- `src/tabpack_experimental/active_fixed_slot.py`: model, active layouts,
  square kernels and repacking.
- `grouped_mlp_block.py`, `fixed_slot_matrix.py`, `grouped_gemm.py`: shared
  blocks, rectangular descriptors and runtime GEMM.
- `grouped_fixed_optimizer.py`, `rectangular_fixed_optimizer.py`,
  `optimizer_kernels.py`: square/rectangular updates and normalization.
- `transfers.py`, `tabpack_adapter.py`: native/compact transfers and the
  two-mode TabPack integration.
- `benchmark.py`: trace-summary hook for `common.profiling`; no training loop.

Custom kernels target CUDA/Triton. Model implementations do not import from
`scripts/`. Tile settings are explicit; no kernel monkeypatches are used.

## Run

Use the repository's CUDA environment and run from the repository root:

```bash
export PYTHONPATH="$PWD:$PWD/src:$PWD/dev/pepilica/src${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m dev.pepilica.scripts.active_variants \
  common/configs/california.json --variant active_grouped_first \
  --profile none --out .cache/runs/california-active
```

Use `active_square_matrices_only` for the native first block. A `native` control
is also available. Training, data loading and profiling use `common.run` and
`common.profiling`; this package does not implement another training loop.

`scripts/benchmark_active_comparison.py` runs sequential native/active comparisons
on one GPU, with warmups, timing, memory and trace runs. Use `--help` or
`--dry-run`; the default measurement epochs are 3 and 21, configurable with
`--epochs`. Example:

```bash
.venv/bin/python -m dev.pepilica.scripts.benchmark_active_comparison \
  --gpu 0 --config common/configs/maps_routing.json \
  --variants native active_square_matrices_only active_grouped_first \
  --out .cache/benchmarks/maps-active --dry-run
```

Local tests remain ignored in `tests/`; run them with
`python -m pytest -c dev/pepilica/pytest.ini dev/pepilica/tests` in that environment.
The reports in `docs/` are historical evidence; their old commands and removed
implementation names are not current entry points.
