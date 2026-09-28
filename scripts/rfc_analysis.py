import sys, pandas as pd, numpy as np
from scipy import stats
path = sys.argv[1]
d = pd.read_parquet(path)
s = d.groupby(["factor", "level", "method", "seed"])[["e2e_q", "incl", "size", "fex", "tx"]].mean().reset_index()
M = ["OLSR", "TOPO-OLSR", "OLSR-W", "OLSR-G", "GRR", "GT+f0"]
pd.set_option("display.width", 220)
for k in ("e2e_q", "incl", "size"):
    t = s[s.method.isin(M)].pivot_table(index=["factor", "level"], columns="method", values=k).reindex(columns=M).round(3)
    print(f"\n=== {k}"); print(t.to_string())
rows = []
for (fa, le), g in s.groupby(["factor", "level"]):
    a = g[g.method == "OLSR-G"].set_index("seed").e2e_q
    r = {"factor": fa, "level": le}
    for b in ("OLSR", "OLSR-W", "TOPO-OLSR"):
        x, y = a.align(g[g.method == b].set_index("seed").e2e_q, join="inner")
        r["d_" + b] = round(float((x - y).mean()), 3)
        r["p_" + b] = stats.wilcoxon(x, y).pvalue if (x - y).abs().sum() > 0 else np.nan
    rows.append(r)
w = pd.DataFrame(rows)
print("\n=== OLSR+GeoTrust+ minus baseline (e2e with quarantine routing), paired Wilcoxon")
print(w.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
w.to_csv(path.replace(".parquet", "_wilcoxon.csv"), index=False)
