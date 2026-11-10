#!/usr/bin/env bash
set -euo pipefail
cd /home/kujbidavs/tabpack
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TRITON_DEFAULT_FP_FUSION=1 PYTHONUNBUFFERED=1
base=.cache/benchmarks/maps-phase-memory-20261106
mkdir -p "$base"
trap 'code=$?; printf "%s\n" "$code" > "$base/exit-code"' EXIT
printf 'fixed\n' > "$base/status"
taskset -c 6,7 .venv/bin/python -m dev.pepilica.scripts.measure_maps_phase_memory --layout fixed --config dev/pepilica/configs/maps_routing_diverse.json --output "$base/fixed" --variant fixed_grouped --independent-shuffle --profile-steps 5 --memory-fraction .45 > "$base/fixed.log" 2>&1
printf 'active\n' > "$base/status"
taskset -c 6,7 .venv/bin/python -m dev.pepilica.scripts.measure_maps_phase_memory --layout active --config dev/pepilica/configs/maps_routing_diverse.json --output "$base/active" --variant fixed_grouped --independent-shuffle --profile-steps 5 --memory-fraction .45 > "$base/active.log" 2>&1
printf 'complete\n' > "$base/status"
