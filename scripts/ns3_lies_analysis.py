"""ns-3 lie-attack campaign on RFC 3626 OLSR (rfc | rfct | rfcw | rfcg), 20 seeds per cell."""
import os, shutil
import numpy as np, pandas as pd
from scipy import stats

SRC = os.path.expanduser(os.environ.get("NS3_LIES_CSV", "~/geotrust_ns3_lies.csv"))
for s, d in ((SRC, "eval/results/ns3_lies.csv"), (SRC + ".f20", "eval/results/ns3_lies_f20.csv")):
    try:
        shutil.copy(s, d)
    except FileNotFoundError:
        pass

NAME = {"rfc": "OLSR", "rfct": "OLSR+topo", "rfcw": "OLSR+WD", "rfcg": "OLSR+GeoTrust+"}
ci = lambda x: f"{x.mean():.3f}±{stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)):.3f}"
rows = []
for tag, path in (("10% insiders", "eval/results/ns3_lies.csv"), ("20% insiders", "eval/results/ns3_lies_f20.csv")):
    try:
        d = pd.read_csv(path)
    except FileNotFoundError:
        continue
    print(f"\n######## {tag}  ({len(d)} runs)")
    for a, g in d.groupby("attack"):
        t = g.groupby("mpr").agg(pdr=("pdr_post", ci), share=("att_slot_share", "mean"),
                                 conv_att=("excl_attacker", "mean"), conv_hon=("excl_honest", "mean"),
                                 mpr=("mpr_size", "mean"), n=("seed", "count")).rename(index=NAME)
        print(f"\n== {a}"); print(t.round(3).to_string())
        ref = g[g.mpr == "rfcg"].set_index("seed").pdr_post
        for b in ("rfc", "rfct", "rfcw"):
            x, y = ref.align(g[g.mpr == b].set_index("seed").pdr_post, join="inner")
            p = stats.wilcoxon(x, y).pvalue if len(x) >= 5 and (x - y).abs().sum() > 0 else np.nan
            print(f"   GeoTrust+ vs {NAME[b]:10s}: {((x - y).mean()):+.4f}  p={p:.2e}  n={len(x)}")
            rows.append(dict(setting=tag, attack=a, vs=NAME[b], d_pdr=(x - y).mean(), p=p, n=len(x)))
pd.DataFrame(rows).to_csv("eval/results/ns3_lies_wilcoxon.csv", index=False)
