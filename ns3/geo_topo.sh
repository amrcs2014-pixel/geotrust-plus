#!/bin/bash
# Dump topology snapshots for OLSR, OLSR+watchdog and OLSR+GeoTrust+ (same seed, same mobility).
set -e
cd ~/ns-3-geotrust
export LD_LIBRARY_PATH=$PWD/build/lib
B=$PWD/build/scratch/ns3.48-geotrust_attack-optimized
OUT="${OUT:-$HOME/geotrust-results}"   # where the snapshot CSVs are copied
mkdir -p "$OUT"
ATT=${1:-shrink}
SEED=${2:-3}
for m in rfc rfcw rfcg; do
  F=/tmp/ns3_topo_${ATT}_${m}_s${SEED}.csv
  $B --mpr=$m --fpolicy=zero --attack=$ATT --n=30 --seed=$SEED --run=$SEED --qmin=0 --T=110 \
     --topo=$F --topoK=100 | tail -1
  cp "$F" "$OUT/"
  echo "  $m: $(grep -c '^mpr' $F) MPR links, $(grep -c '^excl' $F) exclusions, $(grep -c '^fake' $F) fake links"
done
