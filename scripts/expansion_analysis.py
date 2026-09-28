"""Ablation and threshold-sensitivity analysis for the expanded paper (reads per-job part files)."""
import glob
import numpy as np, pandas as pd
from scipy import stats

pd.set_option("display.width", 220)


def load(regime):
    fs = glob.glob(f"eval/results/parts_{regime}/*.parquet") or [f"eval/results/grid_{regime}.parquet"]
    if not fs:
        return None
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    pass
    return d.groupby(["factor", "level", "method", "seed"])[["e2e_q", "incl", "fex", "tpr", "size"]].mean().reset_index()


def paired(s, fa, le, a, b, k="e2e_q"):
    g = s[(s.factor == fa) & (s.level == le)]
    x, y = g[g.method == a].set_index("seed")[k].align(g[g.method == b].set_index("seed")[k], join="inner")
    if len(x) < 5:
        return np.nan, np.nan, len(x)
    p = stats.wilcoxon(x, y).pvalue if (x - y).abs().sum() > 0 else 1.0
    return (x - y).mean(), p, len(x)


ABL = ["OLSR", "TOPO-OLSR", "OLSR-W", "OLSR-V", "OLSR-GnoAuth", "OLSR-Gweak", "OLSR-G"]
for reg in ("abl_sparse", "abl_dense"):
    s = load(reg)
    if s is None:
        continue
    print(f"\n######## {reg}  seeds/level:", s.groupby(["factor", "level"]).seed.nunique().to_dict())
    for k in ("e2e_q", "fex", "tpr", "incl"):
        t = s[s.method.isin(ABL)].pivot_table(index=["factor", "level"], columns="method", values=k).reindex(columns=ABL)
        print(f"\n== {k}"); print(t.round(3).to_string())
    print("\n== OLSR-G minus variant (e2e), paired Wilcoxon")
    for fa, le in s[["factor", "level"]].drop_duplicates().itertuples(index=False):
        row = [f"{fa}={le}"]
        for b in ABL[:-1]:
            d, p, n = paired(s, fa, le, "OLSR-G", b)
            row.append(f"{b}:{d:+.3f}({p:.1e})")
        print("  ".join(row))

for reg in ("sens_sparse", "sens_dense"):
    s = load(reg)
    if s is None:
        continue
    print(f"\n######## {reg}  seeds/level:", s.groupby(["factor", "level"]).seed.nunique().to_dict())
    t = s.groupby(["factor", "level", "method"])[["e2e_q", "fex", "tpr", "incl"]].mean().round(3)
    print(t.to_string())
    for fa, le in s[["factor", "level"]].drop_duplicates().itertuples(index=False):
        for b in ("OLSR", "OLSR-W"):
            d, p, n = paired(s, fa, le, "OLSR-G", b)
            print(f"  {fa}={le}: G+ vs {b}: {d:+.3f} p={p:.1e} n={n}")
