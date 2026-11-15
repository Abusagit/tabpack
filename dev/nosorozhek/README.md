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

Recent benchmarked results can be found in [`benchmarking/benchmark_summary.csv`](benchmarking/benchmark_summary.csv)

## Profiling Datasets

From the root of the repo run the following command:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m dev.nosorozhek.scripts.run_research
```

The datasets, batch sizes and whether to use `torch.compile` or not can be configured at the top of the `scripts/run_research.py` script.
