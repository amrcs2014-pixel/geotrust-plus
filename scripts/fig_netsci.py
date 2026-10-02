"""Network-science figure: (a) delivery loss vs. route capture by insiders (all runs), (b) random vs.
targeted (degree / betweenness) insider placement, sparse and dense swarms (20 seeds, 95% CI)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

NAME = {"OLSR": "OLSR", "TOPO-OLSR": "OLSR + topology", "OLSR-W": "OLSR + watchdog", "OLSR-G": "OLSR + GeoTrust+"}
COL = {"OLSR": "#2a78d6", "TOPO-OLSR": "#eb6834", "OLSR-W": "#1baf7a", "OLSR-G": "#e87ba4"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4), gridspec_kw={"width_ratios": [1.1, 1, 1]})
for k, reg in enumerate(("sparse", "dense")):
    d = pd.read_parquet(f"eval/results/grid_net_{reg}.parquet")
    d["place"] = np.where(d.factor == "main", "random", d.level.astype(str))
    s = d.groupby(["place", "method", "seed"])[["e2e_q", "route_att_q"]].mean().reset_index()
    for m in NAME:
        g = s[s.method == m]
        ax[0].scatter(g.route_att_q, 1 - g.e2e_q, s=5, alpha=.5, color=COL[m], marker="o" if reg == "sparse" else "^",
                      label=NAME[m] if k == 0 else None)
        a = ax[1 + k]
        xs = np.arange(3) + (list(NAME).index(m) - 1.5) * 0.2
        mu, ci = [], []
        for pl in ("random", "degree", "betweenness"):
            v = g[g.place == pl].e2e_q
            mu.append(v.mean()); ci.append(stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
        a.bar(xs, mu, 0.2, yerr=ci, color=COL[m], capsize=1.5, label=NAME[m])
    a.set_xticks(range(3)); a.set_xticklabels(["random", "top\ndegree", "top\nbetweenness"], fontsize=7)
    a.set_ylim(0.5, 1.0); a.set_title(f"({'bc'[k]}) {reg} swarm", fontsize=8); a.set_ylabel("end-to-end delivery")
ax[0].plot([0, 0.5], [0, 0.5], "k--", lw=.7)
ax[0].set_xlabel("fraction of honest flows routed via an insider"); ax[0].set_ylabel("delivery loss")
ax[0].set_title("(a) loss vs. route capture", fontsize=8); ax[0].legend(fontsize=6, frameon=False, loc="lower right", bbox_to_anchor=(1.04, -0.03), handletextpad=0.1, borderaxespad=0.2)
fig.tight_layout()
fig.savefig("eval/figures/fig_netsci.pdf"); fig.savefig("eval/figures/fig_netsci.png", dpi=200)
print("ok")
