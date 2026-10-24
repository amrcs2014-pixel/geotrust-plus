#!/bin/bash
# Campaigns under the faithful observation model (obs="ru"), run after the PLOS ONE editorial assessment.
cd "$(dirname "$0")/.."
export PYTHONPATH=.
for spec in "ru_sparse 30 3" "ru_dense 30 2" "loss_sparse 30 3" "loss_dense 30 2" "frame_sparse 30 3" "frame_dense 30 2"; do
  set -- $spec
  for attempt in 1 2 3; do          # resume after a worker crash (memory); finished part files are skipped
    echo "$(date +%H:%M:%S) START $1 (attempt $attempt)"
    python scripts/run_grid.py --regime $1 --seeds $2 --workers $3 --out eval/results/grid_$1.parquet > eval/results/$1.out 2>&1
    tail -1 eval/results/$1.out | grep -q "^done" && break
  done
  echo "$(date +%H:%M:%S) END $1 $(tail -1 eval/results/$1.out)"
done
