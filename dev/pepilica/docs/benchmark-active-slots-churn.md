# Active-only fixed-slot activations: churn-shaped benchmark

`ActiveFixedSlotMLP` keeps weight and bias slots for all candidates, while activations use `[batch, sum(active widths)]`. It supports runtime selected and reordered IDs, depth-dependent identity paths, dropout and first-order backward. The active and fixed bias offsets are independent; metadata and repack indices use bounded LRU caches. The GEMM input row stride is runtime rather than `tl.constexpr`. Gradients remain full-size and zero-initialized for inactive parameters; the optimizer is unchanged. Only the unfused path is implemented. `repack` subsets or reorders an existing layout via an index-select operation; it cannot manufacture activations for newly added models, which need fresh inputs. This is an isolated hidden-stack prototype, not yet a TabPack adapter that writes the first layer directly into the new layout.

The workload uses `dev/pepilica/configs/churn.json` for 64 widths sampled from 128, 256 and 384, batch 256, dropout 0.1 and depths 1–4, using random seed 0; actual churn examples are not used. Same randomly selected IDs, weights layout, BF16 activations, FP32 parameters, and 64x64 GEMM tiles for each pair. GPU RTX A4000 device 6. No optimizer or real-data end-to-end model. CUDA graph results are isolated repeated forward-graph and backward-graph replay timings, not end-to-end training wall time. They reduce host scheduling noise; each captured backward reuses its captured forward tensors. Eager peak allocated memory includes model, input, activations and full parameter gradients, measured during forward/backward; excludes data and optimizer.

| Active | Active / total width | Layout | Forward ms | Backward ms | Eager peak MiB |
| ---: | ---: | --- | ---: | ---: | ---: |
| 64 | 17024 / 17024 | fixed columns | 0.942 | 1.434 | 218.9 |
| 64 | 17024 / 17024 | active-only | 0.866 | 1.367 | 219.0 |
| 32 | 8576 / 17024 | fixed columns | 0.800 | 1.196 | 218.9 |
| 32 | 8576 / 17024 | active-only | 0.602 | 0.912 | 190.0 |
| 8 | 2048 / 17024 | fixed columns | 0.526 | 0.783 | 198.7 |
| 8 | 2048 / 17024 | active-only | 0.321 | 0.449 | 154.8 |

Layout change measured separately from full layout to each subset, using `repack` with real tensor copying. Over five trials, synchronized cold metadata preparation median was 11.06 ms for 32 active and 3.38 ms for 8 active; cold repack including first index construction median was 1.02 ms and 0.47 ms, respectively. The very first cold repack incurred 31.8 ms, so startup can be substantially more expensive. Cached repack GPU event medians across trials were 0.083 ms and 0.094 ms, respectively, but varied with heavy host contention. Cold metadata is per new active set, not per layer invocation; cache holds 32 recent layouts. Returning to cached layouts avoids rebuilding them. These numbers do not include packing directly from native TabPack; no such integration exists yet. Frequent arbitrary active-set changes could make the metadata and compaction costs significant.

Ten correctness tests compare outputs and gradients against existing fixed slots including random ordering, depth variation, dropout, overlapping forward lifetimes, FP32 master weights, repack autograd and tile sizes 32/64. Scripts: `dev/pepilica/scripts/benchmark_active_fixed_slot.py` and `dev/pepilica/scripts/benchmark_active_fixed_graph.py`. Artifacts in `.cache/benchmarks/active-slots-churn-20261106/`.


## Full-pack buffer-fill control

To investigate why active-only was faster with all 64 models active, a scoped control changes only the original `CompactFixedSlotMLP` path: the unfused linear intermediate and `dx` are allocated with `empty_like` instead of `zeros_like`. The intermediate is read only at selected columns by the masked pointwise operation; every `dx` position is written either by the GEMM or the identity carry. Full parameter/bias gradients are still zeroed. This is an experiment monkeypatch, not a safe general replacement for `FixedSlotPlan`, whose standalone callers can observe inactive outputs.

Three repeats in alternating orders, same 64 widths/depths and batch 256 from churn configuration, BF16 activations, FP32 weights, dropout 0.1, 64x64 tiles; RTX A4000 device 0, CUDA graph forward and backward timed separately. Values are medians of each repeat's medians (15 samples of 20 graph replays), milliseconds:

| Full-pack variant | Forward | Backward | Forward + backward (sum) |
| --- | ---: | ---: | ---: |
| Original fixed slots, zero buffers | 0.969 | 1.470 | 2.439 |
| Fixed slots, no redundant fills | 0.891 | 1.393 | 2.284 |
| Active-only slots | 0.891 | 1.404 | 2.295 |

The no-fill fixed-slot control removes approximately 0.155 ms from the sum of forward/backward medians and accounts for essentially all the full-pack difference against active-only in this controlled GPU-graph experiment. Routing metadata preparation and repacking are outside capture, so this does not assess eager wall-clock or changing-ensemble overhead. Eight targeted tests covered identical outputs and gradients with depth variation, dropout, reordered/partial active sets and 64x64 tiles. The experimental control is in `dev/pepilica/scripts/experiment_fixed_nozero.py`, benchmark entry `dev/pepilica/scripts/benchmark_fixed_nozero_control.py`, raw output `.cache/benchmarks/active-slots-churn-20261106/nozero-control.json`.
