# Some profiling is currently going on here


## Datasets
All prepared datasets should be stored under `data/` directory at the repository root

## Profiling datasets

From the root of the repo run the following command:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m dev.nosorozhek.scripts.run_research
```

The datasets, batch sizes and whether to use `torch.compile` or not can be configured at the top of the `scripts/run_research.py` script.
