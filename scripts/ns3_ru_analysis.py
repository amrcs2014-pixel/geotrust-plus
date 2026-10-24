"""ns-3 lie attacks with the faithful ranging-report model (geo_campaign6_ru.sh), 20 seeds per cell.

GeoTrust+ (rfcg) and the topology check (rfct) run with --ruModel=1: RUs ride on HELLO frames and are lost with them.
OLSR (rfc) and OLSR + watchdog (rfcw) do not use RUs; their rows come from geo_campaign5.sh (same seeds).
Also reported: GeoTrust+ without the TESLA wait (ablation) and with 20% extra RU loss.
Writes eval/results/ns3_ru.csv (+ _f20 / _nopend / _loss20) and eval/results/ns3_ru_wilcoxon.csv.
Usage: python scripts/ns3_ru_analysis.py"""
import os, shutil
import numpy as np, pandas as pd
from scipy import stats

H = os.environ.get("NS3_HOME", "//wsl.localhost/Ubuntu-22.04/home/amr")
R = "eval/results"
COPY = {"geotrust_ns3_ru.csv": "ns3_ru.csv", "geotrust_ns3_ru.csv.f20": "ns3_ru_f20.csv",
        "geotrust_ns3_ru.csv.nopend": "ns3_ru_nopend.csv", "geotrust_ns3_ru.csv.f20.nopend": "ns3_ru_f20_nopend.csv",
        "geotrust_ns3_ru.csv.loss20": "ns3_ru_loss20.csv",
        "geotrust_ns3_lies.csv": "ns3_lies.csv", "geotrust_ns3_lies.csv.f20": "ns3_lies_f20.csv"}
for s, d in COPY.items():
    if os.path.exists(f"{H}/{s}"):
        shutil.copy(f"{H}/{s}", f"{R}/{d}")


def load(name):
    p = f"{R}/{name}"
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()


NAME = {"rfc": "OLSR", "rfct": "OLSR+topo", "rfcw": "OLSR+WD", "rfcg": "OLSR+GeoTrust+"}
ci = lambda x: f"{x.mean():.3f}±{stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)):.3f}"
rows = []
for tag, ru_f, legacy_f, nop_f in (("10% insiders", "ns3_ru.csv", "ns3_lies.csv", "ns3_ru_nopend.csv"),
                                   ("20% insiders", "ns3_ru_f20.csv", "ns3_lies_f20.csv", "ns3_ru_f20_nopend.csv")):
    ru, leg, nop = load(ru_f), load(legacy_f), load(nop_f)
    if ru.empty:
        continue
    d = pd.concat([ru, leg[leg.mpr.isin(["rfc", "rfcw"])]], ignore_index=True)
    print(f"\n######## {tag}  ({len(ru)} faithful-model runs)")
    for a, g in d.groupby("attack"):
        if a not in set(ru.attack):
            continue
        t = g.groupby("mpr").agg(pdr=("pdr_post", ci), share=("att_slot_share", "mean"),
                                 conv_att=("excl_attacker", "mean"), excl_hon=("excl_honest", "mean"),
                                 n=("seed", "count")).rename(index=NAME)
        gg = g[g.mpr == "rfcg"]
        print(f"\n== {a}")
        print(t.round(4).to_string())
        if "excl_honest_conv" in gg:
            print(f"   GeoTrust+ honest convictions {100 * gg.excl_honest_conv.mean():.2f}%  "
                  f"(other honest exclusion {100 * gg.excl_honest_pend.mean():.2f}%)")
        ref = gg.set_index("seed").pdr_post
        comps = [(b, g[g.mpr == b].set_index("seed").pdr_post) for b in ("rfc", "rfct", "rfcw")]
        if not nop.empty:
            comps.append(("no TESLA wait", nop[(nop.attack == a) & (nop.mpr == "rfcg")].set_index("seed").pdr_post))
        if not leg.empty:
            comps.append(("idealised RUs", leg[(leg.attack == a) & (leg.mpr == "rfcg")].set_index("seed").pdr_post))
        for b, other in comps:
            x, y = ref.align(other, join="inner")
            p = stats.wilcoxon(x, y).pvalue if len(x) >= 5 and (x - y).abs().sum() > 0 else np.nan
            lab = NAME.get(b, b)
            print(f"   GeoTrust+ vs {lab:14s}: {(x - y).mean():+.4f}  p={p:.2e}  n={len(x)}")
            rows.append(dict(setting=tag, attack=a, vs=lab, d_pdr=(x - y).mean(), p=p, n=len(x),
                             pdr_g=x.mean(), pdr_other=y.mean()))
loss = load("ns3_ru_loss20.csv")
if not loss.empty:
    base = load("ns3_ru.csv")
    print("\n######## 20% extra RU loss (GeoTrust+)")
    for a, g in loss.groupby("attack"):
        x = g.set_index("seed").pdr_post
        y = base[(base.attack == a) & (base.mpr == "rfcg")].set_index("seed").pdr_post
        x, y = x.align(y, join="inner")
        p = stats.wilcoxon(x, y).pvalue if len(x) >= 5 and (x - y).abs().sum() > 0 else np.nan
        print(f"   {a}: {x.mean():.3f} vs {y.mean():.3f} without extra loss ({(x - y).mean():+.4f}, p={p:.2e}); "
              f"honest convictions {100 * g.excl_honest_conv.mean():.2f}%, liars convicted {g.excl_attacker.mean():.3f}")
        rows.append(dict(setting="10% insiders, 20% extra RU loss", attack=a, vs="no extra loss",
                         d_pdr=(x - y).mean(), p=p, n=len(x), pdr_g=x.mean(), pdr_other=y.mean()))
pd.DataFrame(rows).to_csv(f"{R}/ns3_ru_wilcoxon.csv", index=False)
