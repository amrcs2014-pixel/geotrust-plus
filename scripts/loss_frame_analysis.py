"""Faithful observation model: (1) robustness to RU loss, bursts and the freshness/persistence parameters
(loss_* campaigns), (2) coordinated framing against the collusion bound (frame_* campaigns, attack A15).
30 seeds; paired Wilcoxon tests of GeoTrust+ against the watchdog.  Writes eval/results/loss_summary.csv and
eval/results/frame_summary.csv.   Usage: python scripts/loss_frame_analysis.py"""
import os
import numpy as np, pandas as pd
from scipy import stats

R = "eval/results"
COLS = ["e2e_q", "fex", "fex_R", "fex_auth", "tpr", "fex_vic", "frame_k", "coll_f"]


def cells(path):
    d = pd.read_parquet(path)
    keep = [c for c in COLS if c in d.columns]
    return d.groupby(["factor", "level", "method", "seed"])[keep].mean()


def summary(reg, extra):
    s = cells(f"{R}/grid_{reg}.parquet")
    rows = []
    for fac, lev in s.index.droplevel(["method", "seed"]).unique():
        c = s.xs((fac, lev), level=["factor", "level"])
        g, w = c.xs("OLSR-G", level="method"), c.xs("OLSR-W", level="method")
        r = {"regime": reg, "setting": "main" if fac == "main" else f"{fac}={lev}",
             "G": g.e2e_q.mean(), "W": w.e2e_q.mean(), "OLSR": c.xs("OLSR", level="method").e2e_q.mean(),
             "TOPO": c.xs("TOPO-OLSR", level="method").e2e_q.mean()}
        diff = (g.e2e_q - w.e2e_q).dropna()
        r["p_GvW"] = stats.wilcoxon(diff).pvalue if diff.abs().sum() > 0 else 1.0
        for k in extra:
            r[k] = g[k].mean() if k in g else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


out = []
for reg in ("loss_sparse", "loss_dense"):
    if os.path.exists(f"{R}/grid_{reg}.parquet"):
        out.append(summary(reg, ["fex", "fex_R", "fex_auth", "tpr"]))
if out:
    L = pd.concat(out)
    L.to_csv(f"{R}/loss_summary.csv", index=False)
    with pd.option_context("display.width", 220):
        print(L.round(4).to_string(index=False))
out = []
for reg in ("frame_sparse", "frame_dense"):
    if os.path.exists(f"{R}/grid_{reg}.parquet"):
        out.append(summary(reg, ["frame_k", "coll_f", "fex_vic", "fex", "tpr"]))
if out:
    F = pd.concat(out).sort_values(["regime", "coll_f", "frame_k"])
    F.to_csv(f"{R}/frame_summary.csv", index=False)
    with pd.option_context("display.width", 220):
        print(F.round(4).to_string(index=False))
