#!/bin/bash
# GeoTrust+ on RFC 3626 OLSR: rfc | rfcw | rfcg (+ GRR reference) x attacks x 20 seeds, Q_min per Eq. 3 for GRR
cd ~/ns-3-geotrust
./ns3 build geotrust_attack > ~/geo_build4.log 2>&1 || { echo BUILD_FAILED; tail -30 ~/geo_build4.log; exit 1; }
export LD_LIBRARY_PATH=$PWD/build/lib
B=$PWD/build/scratch/ns3.48-geotrust_attack-optimized
OUT=~/geotrust_ns3_rfc.csv
rm -f $OUT
jobs=()
for s in $(seq 1 20); do
  for a in none forge claim; do
    for m in rfc rfcw rfcg; do
      jobs+=("$B --mpr=$m --fpolicy=zero --attack=$a --n=30 --seed=$s --run=$s --qmin=0 --summary=$OUT")
    done
  done
  # denser swarm and higher attacker fraction (stress cases)
  for m in rfc rfcw rfcg; do
    jobs+=("$B --mpr=$m --fpolicy=zero --attack=claim --n=50 --seed=$s --run=$s --qmin=0 --summary=$OUT.n50")
    jobs+=("$B --mpr=$m --fpolicy=zero --attack=claim --frac=0.2 --n=30 --seed=$s --run=$s --qmin=0 --summary=$OUT.f20")
    jobs+=("$B --mpr=$m --fpolicy=zero --attack=forge --frac=0.2 --n=30 --seed=$s --run=$s --qmin=0 --summary=$OUT.f20")
  done
done
printf '%s\n' "${jobs[@]}" | xargs -P 6 -I{} bash -c '{} > /dev/null 2>&1'
echo CAMPAIGN4_DONE $(wc -l < $OUT)
