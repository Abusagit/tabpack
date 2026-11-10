# veleboks development workspace

Personal profiling scripts, configurations, environment records, and small
experiment results live here. Library code remains in `src/tabpackage`.

## Contents

- `run_profiling_experiment.py`: California Housing training and inference profiling.
- `configs/`: paper-style and diverse model packs, with eager or compiled training.
- `scripts/gpu_run.py`: snapshot the current worktree and run it on a GPU host.
- `environments/torch212-requirements.txt`: recorded packages from the PyTorch
  2.12.1 CUDA 12.6 experiment environment; this is not a portable dependency lock.
- `results/`: preserved JSON reports, including the sampled hyperparameters and
  hardware/software versions recorded by each experiment.
- `profiles/` and `artifacts/`: ignored local traces and run bundles.

## Run from the repository root

The remote environment needs TabPack's dependencies plus NumPy and scikit-learn.
The runner sets `PYTHONPATH` to the snapshot's `src`, so uncommitted library
changes are included. It copies tracked files and new non-ignored files; check
new files before sending a snapshot.

```sh
ssh ysda_imladris nvidia-smi
python3 dev/veleboks/scripts/gpu_run.py --gpu 4 -- \
  dev/veleboks/run_profiling_experiment.py \
  dev/veleboks/configs/paper_california.json --output ../output
```

Choose an available GPU. Use `--remote-python /absolute/path/to/python` (or a
quoted `~/...` path) before `--` to select a different remote environment. Replace
the config with `paper_california_compile.json`, `diverse_california.json`, or
`diverse_california_compile.json` for the other variants. All variants compile
the inference predictor; `compile_model` controls training compilation only.

Each run uses a separate `~/tabpack-runs/<run-id>/code` snapshot and writes to
the sibling `output/`. The runner retrieves output even when Python exits with
an error. Locally, `dev/veleboks/artifacts/runs/<run-id>/` contains the manifest
with file hashes, HEAD, branch and arguments, Git diff/status, environment,
log, exit code, and `output/`. Keep snapshots and old run bundles unchanged.
Legacy run bundles remain in the repository-root `artifacts/`.

For a direct CUDA run with the dependencies installed:

```sh
PYTHONPATH=src python3 dev/veleboks/run_profiling_experiment.py \
  dev/veleboks/configs/paper_california.json
```

Without `--output`, reports and traces go into this development directory.
For repeated direct runs, choose a separate output directory to preserve earlier
results; reports otherwise reuse the config name.

## Existing results and measurement limits

The files were moved without changing their contents:

- `paper.json` and `diverse.json` are legacy two-epoch runs. Their reports contain
  sample counts but no dataset identity/hash or test RMSE; do not interpret them
  as the later California Housing benchmark. Their original driver is not part
  of the current tracked scripts.
- `*_california.json` record eager training with PyTorch 2.7.1.
- `*_california_compile.json` record compiled training with PyTorch 2.12.1.
- `*_california_torch212_eager.json` record eager training with PyTorch 2.12.1.

The preserved reports identify a Quadro RTX 4000. Run IDs and exact source
snapshots are not embedded in these historical reports; use a corresponding
original run bundle when available rather than inferring provenance.

California Housing uses a seeded 64%/16%/20% train/validation/test split,
train-fitted quantile normalization, and target standardization. The split is
not claimed to match the paper exactly. The profiler captures a training epoch
including validation and online ensemble updates. Training wall time includes
profiling and any training compilation. Inference wall time covers ten warmed
compiled forwards inside the profiler, including profiler overhead; compilation
is excluded. These numbers are not clean latency measurements. Compare software
versions and configs before drawing performance conclusions.

For PyTorch 2.7 inference compilation, the driver replaces `ParameterPack`
subclasses on the selected predictor with ordinary frozen parameters and checks
eager/compiled prediction equivalence. This adjustment is local to the experiment.
