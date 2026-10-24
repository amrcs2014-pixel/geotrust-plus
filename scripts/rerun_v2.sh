#!/bin/bash
# Re-run every campaign that involves GeoTrust+ with the threshold blame rule (collusion bound f=2).
cd "/e/network research/p7/geotrust-mpr"
export PYTHONPATH=.
for spec in "rfc 30" "rfcdense 30" "abl_sparse 20" "abl_dense 20" "sens_sparse 30" "sens_dense 30" "net_sparse 20" "net_dense 20"; do
  set -- $spec
  echo "$(date +%H:%M:%S) START $1"
  python scripts/run_grid.py --regime $1 --seeds $2 --workers 3 --out eval/results/grid_$1.parquet > eval/results/v2_$1.out 2>&1
  echo "$(date +%H:%M:%S) END $1 exit=$? $(tail -1 eval/results/v2_$1.out)"
done
