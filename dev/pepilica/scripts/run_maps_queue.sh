#!/bin/bash
set -euo pipefail
cd /home/kujbidavs/tabpack
export CUDA_VISIBLE_DEVICES=GPU-ec74220d-78d7-cc47-69fc-0de1fb580944
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TRITON_DEFAULT_FP_FUSION=1
while kill -0 1427684 2>/dev/null; do sleep 10; done
test -s .cache/benchmarks/maps-paper-native-20261105/report.json
run() {
  local name=$1 config=$2 variant=$3
  shift 3
  taskset -c 6,7 .venv/bin/python -u -m dev.pepilica.scripts.run_maps_routing \
    --config "dev/pepilica/configs/maps_routing_${config}.json" \
    --output ".cache/benchmarks/maps-${name}-20261105" \
    --variant "$variant" --independent-shuffle --memory-fraction .75 "$@" \
    > ".cache/benchmarks/maps-${name}.log" 2>&1
}
run diverse-native diverse native --no-profile
run diverse-grouped diverse fixed_grouped --no-profile
run paper-native-profile paper native --max-epochs 3 --profile-steps 20
run diverse-native-profile diverse native --max-epochs 3 --profile-steps 20
run diverse-grouped-profile diverse fixed_grouped --max-epochs 3 --profile-steps 20
