#!/bin/bash
# ns-3 lie attacks on RFC 3626 OLSR: rfc | rfct | rfcw | rfcg  x  none|forge|claim|fakelinks|stealth|wormhole|shrink
# resume-safe: skips (mode, attack, frac, seed) rows already present
cd ~/ns-3-geotrust
export LD_LIBRARY_PATH=$PWD/build/lib
B=$PWD/build/scratch/ns3.48-geotrust_attack-optimized
OUT=~/geotrust_ns3_lies.csv
have() {
  [ -f "$1" ] && awk -F, -v m="$2" -v a="$3" -v f="$4" -v s="$5" \
    'NR>1 && $6==m && $9==a && ($10+0)==(f+0) && $2==s {found=1} END {exit !found}' "$1"
}
jobs=()
for s in $(seq 1 20); do
  for a in none forge claim fakelinks stealth wormhole shrink; do
    for m in rfc rfct rfcw rfcg; do
      have $OUT $m $a 0.1 $s || jobs+=("$B --mpr=$m --fpolicy=zero --attack=$a --n=30 --seed=$s --run=$s --qmin=0 --summary=$OUT")
    done
  done
  for a in fakelinks shrink; do
    for m in rfc rfct rfcw rfcg; do
      have $OUT.f20 $m $a 0.2 $s || jobs+=("$B --mpr=$m --fpolicy=zero --attack=$a --frac=0.2 --n=30 --seed=$s --run=$s --qmin=0 --summary=$OUT.f20")
    done
  done
done
echo "runs to do: ${#jobs[@]}"
printf '%s\n' "${jobs[@]}" | xargs -P 6 -I{} bash -c '{} > /dev/null 2>&1'
echo CAMPAIGN5_DONE
