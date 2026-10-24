"""One-sided link claims: blame the claimant (v1.0 rule) vs. count them as ordinary disagreement edges, both with the
threshold blame rule (f=2), against the archived greedy-cover results (v1).  GeoTrust+ only, seeds paired.
Writes eval/results/onesided_summary.csv.   Usage: python scripts/onesided_analysis.py"""
import numpy as np, pandas as pd
from scipy import stats

R = "eval/results"
REG = {"sparse": ("grid_onesided_sparse.parquet", "grid_rfc.parquet", "v1_greedy/grid_rfc.parquet"),
       "dense": ("grid_onesided_dense.parquet", "grid_rfcdense.parquet", "v1_greedy/grid_rfcdense.parquet")}
COLS = ["e2e_q", "fex", "tpr"]


def seedmeans(path, method="OLSR-G"):
    d = pd.read_parquet(f"{R}/{path}")
    return d[d.method == method].groupby(["factor", "level", "seed"])[COLS].mean()


rows = []
for reg, (edge_f, claim_f, v1_f) in REG.items():
    E, C, V, W = seedmeans(edge_f), seedmeans(claim_f), seedmeans(v1_f), seedmeans(claim_f, "OLSR-W")
    for key in E.index.droplevel("seed").unique():
        e, c, v, w = (X.xs(key, level=["factor", "level"]) for X in (E, C, V, W))
        s = e.index.intersection(c.index).intersection(v.index)
        r = {"regime": reg, "setting": key[1] if key[0] != "main" else "main", "seeds": len(s)}
        for col in COLS:
            r[f"{col}_v1"], r[f"{col}_claimant"], r[f"{col}_edge"] = v.loc[s, col].mean(), c.loc[s, col].mean(), \
                e.loc[s, col].mean()
            a, b = e.loc[s, col].astype(float), c.loc[s, col].astype(float)
            ok = a.notna() & b.notna()
            r[f"{col}_p_edge_vs_claimant"] = (stats.wilcoxon(a[ok], b[ok]).pvalue
                                             if ok.sum() >= 5 and (a[ok] - b[ok]).abs().sum() > 0 else 1.0)
        r["e2e_q_watchdog"] = w.loc[s, "e2e_q"].mean()
        a, b = e.loc[s, "e2e_q"], w.loc[s, "e2e_q"]
        r["p_edge_vs_watchdog"] = stats.wilcoxon(a, b).pvalue if (a - b).abs().sum() > 0 else 1.0
        rows.append(r)

out = pd.DataFrame(rows)
out.to_csv(f"{R}/onesided_summary.csv", index=False)
show = out[["regime", "setting", "e2e_q_v1", "e2e_q_claimant", "e2e_q_edge", "e2e_q_p_edge_vs_claimant",
            "e2e_q_watchdog", "fex_v1", "fex_claimant", "fex_edge", "tpr_v1", "tpr_claimant", "tpr_edge"]]
with pd.option_context("display.width", 250, "display.max_columns", 30):
    print(show.round(4).to_string(index=False))
