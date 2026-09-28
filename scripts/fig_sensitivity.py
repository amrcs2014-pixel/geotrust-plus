"""Threshold-sensitivity figure: delivery, honest exclusion and liar detection vs. tau_scale."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

SRC = {"sparse": "eval/results/grid_sens_sparse.parquet", "dense": "eval/results/grid_sens_dense.parquet"}
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3))
for reg, path in SRC.items():
    d = pd.read_parquet(path)
    d = d[(d.factor == "tau_scale") | (d.factor == "main")].copy()
    d["x"] = np.where(d.factor == "main", 1.0, pd.to_numeric(d.level, errors="coerce"))
    s = d.groupby(["x", "method", "seed"])[["e2e_q", "fex", "tpr"]].mean().reset_index()
    ls = "-" if reg == "dense" else "--"
    for k, a in zip(("e2e_q", "fex", "tpr"), ax):
        g = s[s.method == "OLSR-G"].groupby("x")[k]
        m, se = g.mean(), g.std(ddof=1) / np.sqrt(g.count())
        h = stats.t.ppf(.975, g.count() - 1) * se
        a.errorbar(m.index, m.values, yerr=h.values, ls=ls, marker="o", ms=3, capsize=2, label=f"GeoTrust+ ({reg})")
        if k == "e2e_q":
            w = s[s.method == "OLSR-W"].groupby("x")[k].mean()
            a.axhline(w.mean(), ls=ls, color="grey", lw=0.8, label=f"watchdog ({reg})")
for a, t in zip(ax, ("end-to-end delivery", "honest exclusion", "liars convicted")):
    a.set_xscale("log", base=2); a.set_xticks([0.5, 1, 2, 4]); a.set_xticklabels(["0.5", "1", "2", "4"])
    a.set_xlabel(r"threshold scale $\tau/\tau_{\rm cal}$"); a.set_title(t, fontsize=9); a.grid(alpha=.3)
ax[0].legend(fontsize=6, loc="lower right")
fig.tight_layout()
fig.savefig("eval/figures/fig_sensitivity.pdf"); fig.savefig("eval/figures/fig_sensitivity.png", dpi=160)
print("ok")
