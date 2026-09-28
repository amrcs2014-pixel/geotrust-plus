#!/bin/bash
# Faithful ns-3 campaign: Q_min over ALL one-hop relays (Zhou et al. Eq. 3) -> --qmin=0; GeoTrust+ with f = 0
cd ~/ns-3-geotrust
export LD_LIBRARY_PATH=$PWD/build/lib
B=$PWD/build/scratch/ns3.48-geotrust_attack-optimized
OUT=~/geotrust_ns3_qall_full.csv
rm -f $OUT
jobs=()
for s in $(seq 1 20); do
  for a in none forge forge_adapt claim; do
    for m in rfc grr grrw; do
      jobs+=("$B --mpr=$m --attack=$a --n=30 --seed=$s --run=$s --qmin=0 --summary=$OUT")
    done
    jobs+=("$B --mpr=geotrust --fpolicy=zero --attack=$a --n=30 --seed=$s --run=$s --qmin=0 --summary=$OUT")
  done
done
printf '%s\n' "${jobs[@]}" | xargs -P 5 -I{} bash -c '{} > /dev/null 2>&1'
echo CAMPAIGN3_DONE $(wc -l < $OUT)
