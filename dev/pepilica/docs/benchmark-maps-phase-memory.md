# Maps-routing phase-by-phase CUDA memory

Two sequential training runs on the same NVIDIA RTX A4000 GPU 1: full fixed-column CompactFixedSlotMLP and active-only ActiveFixedSlotMLP, both 64x64, unfused hidden stack, grouped fixed-slot optimizer and benchmark-only native TabPack bridge. Same seed 0, sampled hyperparameters, BF16 autocast/FP32 master weights, batch size 1024, train and validation data resident, independent per-model shuffles. Both naturally reached identical candidate IDs at epoch 23: 32 active models and summed widths 6336 out of 16192. Measured five consecutive batches, zero-based batch indices 2–6, at epoch 3 and epoch 23, after two unmeasured batches to warm any changed-layout caches. Runs were deliberately stopped after both captures; they are not full-quality training runs.

For every measured phase, synchronize the GPU, reset peak memory stats, execute the phase and synchronize, then record CUDA allocated and reserved memory at start, end and peak. Gather occurs before the forward measurement; starting allocations include its batch and all live persistent allocations. The forward phase is `training.forward`, loss is loss computation, backward is `loss.backward()`, optimizer is the full native plus grouped step, and zero-grad is optimizer.zero_grad(). Values are medians over five batches and are absolute GiB allocated (not extra allocation during the phase). These measurements have synchronization overhead and are not timing benchmarks.

| Active | Layout | Forward peak | Loss peak | Backward peak | Optimizer peak | Allocated after backward |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 64 | Fixed columns | 2.877208 | 2.629711 | 2.699841 | 2.542789 | 2.350284 |
| 64 | Active-only | 2.877394 | 2.629897 | 2.700027 | 2.542975 | 2.350470 |
| 32 | Fixed columns | 2.070116 | 1.946321 | 2.018260 | 1.861047 | 1.760095 |
| 32 | Active-only | 1.994280 | 1.870486 | 1.909519 | 1.861506 | 1.760553 |

At epoch 3 the observed phase peak is forward, not optimizer, and both layouts have the same width. After shrinking, active-only lowers forward peak by 0.075836 GiB (77.7 MiB) and backward peak by 0.108741 GiB (111.4 MiB). Allocated after backward and optimizer peak are effectively identical; saved activations have been released and full-shaped gradients, weights, optimizer states and compatibility storage persist. Phase accounting does not isolate which individual forward operation is responsible for the peak. The per-step peak after shrinking is lower, but the whole-run peak remains dominated by an earlier full-width step. The median reserved peak is 2.992188 GiB during epoch 3 and 3.199219 GiB during the shrunken measurement in both variants, illustrating caching allocator retained memory; allocated and reserved have different meanings.

Each output `phase-memory.json` contains all individual sample start/end/peak allocated/reserved bytes and active IDs. Files live under `.cache/benchmarks/maps-phase-memory-20261106/{fixed,active}/`. Reproduce via `dev/pepilica/scripts/measure_maps_phase_memory.py` or `dev/pepilica/scripts/run_maps_phase_memory_comparison_20261106.sh` after selecting a fresh output path.


## Fixed-width vanilla follow-up

Vanilla TabPack with all widths fixed at 384 was measured on the same GPU using the same epoch-3 and first-shrunken-pack protocol. It has the same random seed and batch size but samples its own depths and hyperparameters. At the later capture it had 27 active models at epoch 21, whereas both diverse layouts had 32 at epoch 23; these are not matched active packs.

| Active | Layout | Forward peak GiB | Backward peak GiB | Optimizer peak GiB | Allocated after backward GiB |
| ---: | --- | ---: | ---: | ---: | ---: |
| 64 | Vanilla fixed width 384 | 2.5787 | 2.3443 | 2.0547 | 1.9200 |
| 64 | Fixed-slot diverse | 2.8772 | 2.6998 | 2.5428 | 2.3503 |
| 64 | Active-only diverse | 2.8774 | 2.7000 | 2.5430 | 2.3505 |
| 27 | Vanilla fixed width 384 | 1.5316 | 1.4425 | 1.3426 | 1.2792 |
| 32 | Fixed-slot diverse | 2.0701 | 2.0183 | 1.8610 | 1.7601 |
| 32 | Active-only diverse | 1.9943 | 1.9095 | 1.8615 | 1.7606 |

At epoch 3 active-only diverse forward peak exceeds fixed-width vanilla by 0.2987 GiB (305.8 MiB), about 11.6%. The active-only integration retains compact fixed weights and optimizer state for all sampled members and native padded hidden weights for compatibility, whereas native fixed-width training removes finished models; the detailed allocation contributions were not separately measured. This explains why lower summed active widths need not imply lower total allocation. Later comparisons also differ in active count and model hyperparameters, so cannot isolate layout impact. Sample file: `.cache/benchmarks/maps-phase-memory-20261106/vanilla-fixed384/phase-memory.json`.
