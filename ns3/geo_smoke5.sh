#!/bin/bash
cd ~/ns-3-geotrust
./ns3 build geotrust_attack > ~/geo_build5.log 2>&1 || { echo BUILD_FAILED; grep -m5 -A5 "error" ~/geo_build5.log; exit 1; }
export LD_LIBRARY_PATH=$PWD/build/lib
B=$PWD/build/scratch/ns3.48-geotrust_attack-optimized
rm -f /tmp/smoke5.csv
for a in shrink fakelinks; do
  for m in rfc rfct rfcw rfcg; do
    $B --mpr=$m --attack=$a --n=30 --T=120 --tOn=40 --seed=1 --qmin=0 --summary=/tmp/smoke5.csv > /dev/null 2>&1 &
  done
done
wait
cut -d, -f6,9,19,20,21,22,32,33 /tmp/smoke5.csv | column -t -s,
