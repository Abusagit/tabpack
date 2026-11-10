# Homecredit-default: full training comparison

All runs used seed 0 on the same RTX A4000 (device 0), original TabReD default split (267,645 train, 58,018 validation, 56,001 test), 64 candidates, batch size 1024, BF16 autocast with FP32 master parameters, model patience 16, online greedy ensemble maximum 32 and ensemble patience 32. Original Homecredit recipe uses noisy-quantile numeric processing, binary extraction and ordinal categories; its binary-feature detection inspects all splits, while numeric and category encoders are train-fitted. Classification uses binary cross-entropy with logits; model probabilities are averaged for online ensembles. Validation and final evaluation use ROC-AUC (the dataset metadata primary score).

| Run | Training wall s | Loop mean seconds/epoch | Epochs | Peak allocated / reserved GiB | Val ROC-AUC | Test ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Vanilla fixed 384 | 622.25 | 8.61 | 72 | 2.791 / 3.156 | 0.836660 | 0.854703 |
| Vanilla diverse | 756.26 | 12.15 | 62 | 3.500 / 3.648 | 0.838483 | 0.854831 |
| Compact fixed-slot diverse 64x64, unfused | 561.16 | 8.46 | 66 | 3.107 / 3.568 | 0.838133 | 0.855206 |

The two diverse runs use the same sampled hyperparameter list and have identical first-three-epoch shuffle hashes. Fixed width 384 uses its own sampled depths and hyperparameters. Kernel rounding and dropout RNG differ, and the compact run stopped when its ensemble finalized with two models still active. The fixed and native diverse runs finished all candidates. Scores are from single-seed runs and do not establish statistical significance.

Training wall covers `tabpackage.train`, including validation, online ensemble selection and initialization, with CUDA synchronization at the end; it excludes data loading and final post-training validation/test prediction. Mean epoch time is the framework training loop time divided by epoch count; it includes validation and changes as models stop. Peak memory is measured during training with train and validation datasets resident; final test inference is excluded. ROC-AUC was independently recomputed from saved predictions using sklearn and matches the framework validation score within floating-point tolerance. Compact used 21,054 calls and no fallbacks.

Compact uses a benchmark-only bridge with native first block and output head, shared FP32 fixed-slot master weights and grouped optimizer. 64x64 tiles are experimental. Full data is preprocessed ahead of timing. Two-epoch smoke test verified training, online ensemble and final evaluation; the ROC-AUC wrapper was validated against sklearn. The dynamic removal/checkpoint integration had been tested on the same backend for Maps-routing.

Artifacts are in `.cache/benchmarks/homecredit-full-comparison-20261106/{paper,native-diverse,compact64-diverse}/` with configs, sampled hyperparameters, histories, reports, saved predictors and predictions; logs are alongside. Entry points are `dev/pepilica/scripts/run_homecredit_full.py` and `dev/pepilica/scripts/run_homecredit_compact_fixed_tile64.py`; serial run script is `dev/pepilica/scripts/run_homecredit_full_comparison_20261106.sh` (change output directory to rerun).
