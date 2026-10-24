#!/bin/bash
# Ablation, threshold sensitivity and targeted placement under the faithful observation model (obs="ru").
cd "$(dirname "$0")/.."
export PYTHONPATH=.
for spec in "abl_ru_sparse 20 3" "abl_ru_dense 20 2" "sens_ru_sparse 30 3" "sens_ru_dense 30 2" "net_ru_sparse 20 3" "net_ru_dense 20 2"; do
  set -- $spec
  for attempt in 1 2 3; do
    echo "$(date +%H:%M:%S) START $1 (attempt $attempt)"
    python scripts/run_grid.py --regime $1 --seeds $2 --workers $3 --out eval/results/grid_$1.parquet > eval/results/$1.out 2>&1
    tail -1 eval/results/$1.out | grep -q "^done" && break
  done
  echo "$(date +%H:%M:%S) END $1 $(tail -1 eval/results/$1.out)"
done
