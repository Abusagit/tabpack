#!/usr/bin/env bash
set -euo pipefail
cd /home/kujbidavs/tabpack
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TRITON_DEFAULT_FP_FUSION=1 PYTHONUNBUFFERED=1 COMPACT_FIXED_FUSED=0
base=.cache/benchmarks/homecredit-full-comparison-20261106
mkdir -p "$base"
trap 'code=$?; printf "%s\n" "$code" > "$base/exit-code"' EXIT
printf 'paper\n' > "$base/status"
taskset -c 6,7 .venv/bin/python -m dev.pepilica.scripts.run_homecredit_full --config dev/pepilica/configs/maps_routing_paper.json --output "$base/paper" --variant native --no-profile --independent-shuffle --memory-fraction 0.5 > "$base/paper.log" 2>&1
printf 'native-diverse\n' > "$base/status"
taskset -c 6,7 .venv/bin/python -m dev.pepilica.scripts.run_homecredit_full --config dev/pepilica/configs/maps_routing_diverse.json --output "$base/native-diverse" --variant native --no-profile --independent-shuffle --memory-fraction 0.5 > "$base/native-diverse.log" 2>&1
printf 'compact64-diverse\n' > "$base/status"
taskset -c 6,7 .venv/bin/python -m dev.pepilica.scripts.run_homecredit_compact_fixed_tile64 --config dev/pepilica/configs/maps_routing_diverse.json --output "$base/compact64-diverse" --variant fixed_grouped --no-profile --independent-shuffle --memory-fraction 0.5 > "$base/compact64-diverse.log" 2>&1
printf 'complete\n' > "$base/status"
