#!/usr/bin/env bash
set -euo pipefail
cd /home/kujbidavs/tabpack
export CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TRITON_DEFAULT_FP_FUSION=1 PYTHONUNBUFFERED=1
base=.cache/benchmarks/maps-active-1024-full-epoch3-20261110
trap 'code=$?; printf "%s\n" "$code" > "${base}.exit-code"' EXIT
taskset -c 6,7 .venv/bin/python -m dev.pepilica.scripts.profile_maps_active_fixed_tile64 \
  --config dev/pepilica/configs/maps_routing_diverse_1024.json \
  --output "$base" --variant fixed_grouped --independent-shuffle \
  --profile-steps 12 --memory-fraction .55 > "${base}.log" 2>&1
