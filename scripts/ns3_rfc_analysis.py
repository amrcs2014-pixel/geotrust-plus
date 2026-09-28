import pandas as pd, numpy as np
from scipy import stats
ci = lambda x: f"{x.mean():.3f}±{stats.t.ppf(.975, len(x)-1) * x.std(ddof=1) / np.sqrt(len(x)):.3f}"
parts = {"n=30, 10% insiders": "eval/results/ns3_rfc.csv", "n=50, 10%": "eval/results/ns3_rfc.n50.csv",
         "n=30, 20% insiders": "eval/results/ns3_rfc.f20.csv"}
out = []
for tag, p in parts.items():
    d = pd.read_csv(p)
    for a, g in d.groupby("attack"):
        t = g.groupby("mpr").agg(pdr=("pdr_post", ci), mpr=("mpr_size", "mean"), att=("att_slot_share", "mean"),
                                 excl_h=("excl_honest", "mean"), excl_a=("excl_attacker", "mean"), n=("seed", "count"))
        print(f"\n== {tag} | attack={a}"); print(t.round(3).to_string())
        ref = g[g.mpr == "rfcg"].set_index("seed").pdr_post
        for b in ("rfc", "rfcw"):
            x, y = ref.align(g[g.mpr == b].set_index("seed").pdr_post, join="inner")
            p_ = stats.wilcoxon(x, y).pvalue if (x - y).abs().sum() > 0 else np.nan
            print(f"   GeoTrust+ vs {b:4s}: {((x-y).mean()):+.4f}  p={p_:.2e}  n={len(x)}")
            out.append(dict(setting=tag, attack=a, vs=b, d_pdr=(x - y).mean(), p=p_, n=len(x)))
pd.DataFrame(out).to_csv("eval/results/ns3_rfc_wilcoxon.csv", index=False)
