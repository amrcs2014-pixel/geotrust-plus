import pandas as pd, numpy as np
from scipy import stats
d = pd.read_parquet("eval/results/grid_zhou.parquet")
s = d.groupby(["factor", "level", "method", "seed"])[["qos", "e2e", "pdr", "incl", "size", "tx", "fex"]].mean().reset_index()
M = ["FLOOD", "GRR", "TOPO-GRR", "GRR-W", "GT+fixF", "GT+f0"]
pd.set_option("display.width", 220)
for k in ("e2e", "incl", "size", "qos"):
    t = s[s.method.isin(M)].pivot_table(index=["factor", "level"], columns="method", values=k).reindex(columns=M).round(3)
    print(f"\n=== {k}"); print(t.to_string())
rows = []
for (fa, le), g in s.groupby(["factor", "level"]):
    ref = g[g.method == "GT+f0"].set_index("seed")
    for b in ("GRR", "GRR-W", "TOPO-GRR"):
        x = g[g.method == b].set_index("seed")
        a, c = ref.e2e.align(x.e2e, join="inner")
        p = stats.wilcoxon(a, c).pvalue if (a - c).abs().sum() > 0 else np.nan
        rows.append(dict(factor=fa, level=le, vs=b, d_e2e=round(float((a - c).mean()), 4), p=p))
w = pd.DataFrame(rows).pivot_table(index=["factor", "level"], columns="vs", values=["d_e2e", "p"])
print("\n=== Wilcoxon e2e, GeoTrust+ minus baseline"); print(w.round(4).to_string())
s.to_csv("eval/results/summary_zhou_seed.csv", index=False)
