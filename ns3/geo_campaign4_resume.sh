#!/bin/bash
# Resume the RFC 3626 campaign: run only the (mode, attack, frac, n, seed) combinations missing from the CSVs
cd ~/ns-3-geotrust
export LD_LIBRARY_PATH=$PWD/build/lib
B=$PWD/build/scratch/ns3.48-geotrust_attack-optimized
OUT=~/geotrust_ns3_rfc.csv
have() {  # file mpr attack frac n seed -> 0 if a row exists
  [ -f "$1" ] && awk -F, -v m="$2" -v a="$3" -v f="$4" -v n="$5" -v s="$6" \
    'NR>1 && $6==m && $9==a && ($10+0)==(f+0) && $3==n && $2==s {found=1} END {exit !found}' "$1"
}
jobs=()
for s in $(seq 1 20); do
  for a in none forge claim; do
    for m in rfc rfcw rfcg; do
      have $OUT $m $a 0.1 30 $s || jobs+=("$B --mpr=$m --fpolicy=zero --attack=$a --n=30 --seed=$s --run=$s --qmin=0 --summary=$OUT")
    done
  done
  for m in rfc rfcw rfcg; do
    have $OUT.n50 $m claim 0.1 50 $s || jobs+=("$B --mpr=$m --fpolicy=zero --attack=claim --n=50 --seed=$s --run=$s --qmin=0 --summary=$OUT.n50")
    have $OUT.f20 $m claim 0.2 30 $s || jobs+=("$B --mpr=$m --fpolicy=zero --attack=claim --frac=0.2 --n=30 --seed=$s --run=$s --qmin=0 --summary=$OUT.f20")
    have $OUT.f20 $m forge 0.2 30 $s || jobs+=("$B --mpr=$m --fpolicy=zero --attack=forge --frac=0.2 --n=30 --seed=$s --run=$s --qmin=0 --summary=$OUT.f20")
  done
done
echo "missing runs: ${#jobs[@]}"
printf '%s\n' "${jobs[@]}" | xargs -P 6 -I{} bash -c '{} > /dev/null 2>&1'
echo CAMPAIGN4_DONE
