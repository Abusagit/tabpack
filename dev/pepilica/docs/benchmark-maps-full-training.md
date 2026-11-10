# Maps-routing: full training comparison

All runs completed with seed 0 on the same RTX A4000 (device 0), original TabReD default split, 64 candidates, batch size 1024, BF16 autocast with FP32 master parameters, candidate patience 16, greedy online ensemble maximum size 32 and ensemble patience 32. The same sampled hyperparameter list was used by both diverse runs; the first-three-epoch shuffle hashes also match. The fixed-width config uses 384 everywhere and samples its own depths and other parameters. Widths in the diverse configuration are sampled uniformly from 64, 128, 256, 384 and 512. Initial shuffles and model seeds match, but kernel rounding, random-number streams and model stopping histories may differ.

| Run | Training wall s | Loop mean seconds/epoch | Epochs | Peak allocated GiB | Val RMSE | Test RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Vanilla fixed 384 | 308.49 | 5.47 | 56 | 2.579 | 0.1612220 | 0.1611841 |
| Vanilla diverse | 430.94 | 7.14 | 60 | 3.096 | 0.1610591 | 0.1611600 |
| Compact fixed-slot diverse 64x64, unfused | 313.96 | 5.10 | 61 | 2.877 | 0.1610891 | 0.1610991 |

Training wall covers the call to `tabpackage.train`, including validation, online ensemble selection and initialization, with synchronization at the end. It excludes preprocessed data loading and final post-training validation/test predictions. Mean epoch time is the framework training loop time divided by epochs; it includes validation and varies as models stop. Peak is CUDA allocated memory during training, including resident training and validation datasets, and excludes final test inference. These are single-seed runs. RMSE is computed on inverse-standardized targets and is the primary regression metric specified by dataset metadata and TabReD.

The compact run uses a benchmark-only bridge sharing fixed-slot FP32 master parameters and the grouped optimizer; first block and head remain native. Its 64x64 tile wrapper is experimental. The compact path reported 13,176 calls and no fallbacks. Its RNG and numerical precision do not yield bitwise-equivalent training trajectories. Before the full run, the benchmark path passed a two-epoch smoke run and a native comparison with model removal/checkpoint synchronization.

Artifacts: `.cache/benchmarks/maps-full-comparison-20261106/{paper,native-diverse,compact64-diverse}/`, including configs, sampled hyperparameters, histories, reports, checkpoints and predictions. Logs alongside output directories. Reproduce with `dev/pepilica/scripts/run_maps_full_comparison_20261106.sh` after changing its output directory; it uses `dev/pepilica/scripts/run_maps_routing.py` and `dev/pepilica/scripts/run_maps_compact_fixed_tile64.py`.
