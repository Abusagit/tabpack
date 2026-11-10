# Maps-routing: compact fixed-slot comparison

Four sequential runs on RTX A4000 device 4, `CUDA_VISIBLE_DEVICES=4`, same train-only Maps-routing data, seed 0, batch size 1024, 64 candidates, two full 157-batch warm-up epochs and 12 profiled batches at the start of epoch 3. Validation, test and model removal are disabled; all models stay active. The three diverse setups use identical widths and depths, sampled from the shared config. Fixed width is 384 and its sampled depths differ from the diverse candidates. The fixed-slot and compact runs use 64x64 tiles for forward, backward and grouped optimizer GEMMs. Compact uses the unfused blocks in a benchmark-only bridge with the native first block and output head.

| Setup | Gather ms | Forward ms | Loss ms | Backward ms | Optimizer ms | Wall step ms | Peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Vanilla fixed width 384 | 16.721 | 10.685 | 0.016 | 9.691 | 20.750 | 49.157 | 2.156 |
| Vanilla diverse | 16.721 | 13.647 | 0.016 | 13.016 | 35.772 | 74.461 | 2.562 |
| Grouped fixed slots 64x64 | 16.724 | 13.300 | 0.016 | 9.771 | 19.501 | 46.697 | 2.531 |
| Compact fixed slots unfused 64x64 | 16.720 | 9.156 | 0.016 | 7.520 | 19.687 | 40.485 | 2.345 |

GPU columns are mean sums of per-phase kernel durations per batch. Wall is median synchronized step with profiler enabled and excludes gather. Peaks include resident training data. Dropout RNG and floating-point arithmetic may differ across variants; no test quality was measured. Each is one capture, and profiler overhead and shared-device activity can affect comparisons.

Full traces, reports and operator tables are in `.cache/benchmarks/maps-profile-compare-v2-{paper,native,fixed64,compact64}-20261106/`. Compact entry point: `COMPACT_FIXED_FUSED=0 .venv/bin/python -m dev.pepilica.scripts.profile_maps_compact_fixed_tile64` with the same arguments as `dev.pepilica.scripts.profile_maps_epoch3`.
