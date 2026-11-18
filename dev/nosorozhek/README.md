# Some profiling is currently going on here

## Directory Structure

- `custom/`: contains custom tuned modules, packs, optimizers, etc.
- `custom/kernels`: contains custom triton kernels
- `scripts`: contains utilitary scripts for benchmarking, collecting traces, etc.

## Datasets
All prepared datasets should be stored under `data/` directory at the repository root

## Benchmarking

The tabpack model setups to be benchmarked are defined at the top of the `scripts/run_benchmark.py` file. 

Benchmarking process can be run with:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m dev.nosorozhek.scripts.run_benchmark
```

Script has several useful flags:
- `--trace` captures the traces for each case
- `--aggregate-only` skips the benchmarking part and only composes summary csv table using existing cached runs
- `--case` allows to specify which cases to run

## Profiling Datasets

From the root of the repo run the following command:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m dev.nosorozhek.scripts.run_research
```

The datasets, batch sizes and whether to use `torch.compile` or not can be configured at the top of the `scripts/run_research.py` script.

## Testing

With pytest installed, run the tests from the repository root using
```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest dev/nosorozhek/tests/
```
