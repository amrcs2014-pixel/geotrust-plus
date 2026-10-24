#!/bin/bash
# ns-3 lie attacks with the faithful ranging-report model (--ruModel=1): RUs ride on HELLO frames and are lost with
# them; two-way exchange freshness, verifier views, exchange labels, persistence (m=6), threshold rule (f=2).
# OLSR (rfc) and OLSR + watchdog (rfcw) do not use RUs: their rows come from geo_campaign5.sh (same seeds, same
# binary behaviour).  Resume-safe: skips (mode, attack, frac, seed, variant) rows already present.
cd ~/ns-3-geotrust
export LD_LIBRARY_PATH=$PWD/build/lib
B=$PWD/build/scratch/ns3.48-geotrust_attack-optimized
OUT=~/geotrust_ns3_ru.csv
RU="--ruModel=1 --collF=2 --persist=6 --ttl=2 --fw=2"
have() {   # file mode attack frac seed pendFirst noPend ruLoss
  [ -f "$1" ] && grep -q "^$5,$5," "$1" && awk -F, -v m="$2" -v a="$3" -v f="$4" -v s="$5" -v l="$8" \
    'NR>1 && $6==m && $9==a && ($10+0)==(f+0) && $2==s && ($35+0)==(l+0) {found=1} END {exit !found}' "$1"
}
jobs=()
for s in $(seq 1 20); do
  for a in none forge claim fakelinks stealth wormhole shrink; do
    have $OUT rfcg $a 0.1 $s || jobs+=("$B --mpr=rfcg $RU --pendFirst=1 --fpolicy=zero --attack=$a --n=30 --seed=$s --run=$s --qmin=0 --summary=$OUT")
    have $OUT rfct $a 0.1 $s || jobs+=("$B --mpr=rfct $RU --fpolicy=zero --attack=$a --n=30 --seed=$s --run=$s --qmin=0 --summary=$OUT")
    have $OUT.nopend rfcg $a 0.1 $s || jobs+=("$B --mpr=rfcg $RU --noPend=1 --fpolicy=zero --attack=$a --n=30 --seed=$s --run=$s --qmin=0 --summary=$OUT.nopend")
  done
  for a in fakelinks shrink; do
    have $OUT.f20 rfcg $a 0.2 $s || jobs+=("$B --mpr=rfcg $RU --pendFirst=1 --fpolicy=zero --attack=$a --frac=0.2 --n=30 --seed=$s --run=$s --qmin=0 --summary=$OUT.f20")
    have $OUT.f20 rfct $a 0.2 $s || jobs+=("$B --mpr=rfct $RU --fpolicy=zero --attack=$a --frac=0.2 --n=30 --seed=$s --run=$s --qmin=0 --summary=$OUT.f20")
    have $OUT.f20.nopend rfcg $a 0.2 $s || jobs+=("$B --mpr=rfcg $RU --noPend=1 --fpolicy=zero --attack=$a --frac=0.2 --n=30 --seed=$s --run=$s --qmin=0 --summary=$OUT.f20.nopend")
  done
  for a in none fakelinks shrink; do
    have $OUT.loss20 rfcg $a 0.1 $s 1 0 0.2 || jobs+=("$B --mpr=rfcg $RU --pendFirst=1 --ruLoss=0.2 --fpolicy=zero --attack=$a --n=30 --seed=$s --run=$s --qmin=0 --summary=$OUT.loss20")
  done
done
echo "runs to do: ${#jobs[@]}"
printf '%s\n' "${jobs[@]}" | xargs -P 10 -I{} bash -c '{} > /dev/null 2>&1'
echo CAMPAIGN6_DONE
