"""Numbers of the network-science section: delivery loss vs. route capture (all schemes, placements, seeds), the
insiders' structural betweenness share, route capture per scheme, and the effect of targeted placement.
Reads the net_* campaigns (GT_OBS=ru: faithful observation model, default; GT_OBS=ideal: v1 campaigns).
Usage: python scripts/netsci_capture.py"""
import os
import numpy as np, pandas as pd
from scipy import stats

OBS = os.environ.get("GT_OBS", "ru")
M = ["OLSR", "TOPO-OLSR", "OLSR-W", "OLSR-G"]
for reg in ("sparse", "dense"):
    d = pd.read_parquet(f"eval/results/grid_net_{'ru_' if OBS == 'ru' else ''}{reg}.parquet")
    d["place"] = np.where(d.factor == "main", "random", d.level)
    s = d.groupby(["place", "method", "seed"])[["e2e_q", "route_att_q", "btw_att"]].mean().reset_index()
    loss, cap = 1 - s.e2e_q, s.route_att_q
    fit = stats.linregress(cap, loss)
    print(f"\n[{reg}] loss = {fit.slope:.2f} x capture + {fit.intercept:.3f}  (r = {fit.rvalue:.3f}, n = {len(s)})")
    t = s.groupby(["place", "method"])[["e2e_q", "route_att_q", "btw_att"]].mean().unstack("method")
    print("structural betweenness share of insiders:",
          {p: round(float(t.loc[p, ("btw_att", "OLSR")]), 3) for p in t.index})
    print("route capture:\n", t["route_att_q"][M].round(3).to_string())
    print("delivery:\n", t["e2e_q"][M].round(3).to_string())
    base = s[s.place == "random"].set_index(["method", "seed"]).e2e_q
    for p in ("degree", "betweenness"):
        x = s[s.place == p].set_index(["method", "seed"]).e2e_q
        out = []
        for m in M:
            dlt = (x.xs(m) - base.xs(m)).dropna()
            pv = stats.wilcoxon(dlt).pvalue if dlt.abs().sum() > 0 else 1.0
            out.append(f"{m}: {100 * dlt.mean():+.1f} pp (p={pv:.2g})")
        g, w = x.xs("OLSR-G"), x.xs("OLSR-W")
        gw = (g - w).dropna()
        print(f"targeted ({p}) vs random:", "; ".join(out),
              f"| GeoTrust+ - watchdog: {100 * gw.mean():+.1f} pp (p={stats.wilcoxon(gw).pvalue:.2g})")
    g, w = base.xs("OLSR-G"), base.xs("OLSR-W")
    print(f"random: GeoTrust+ - watchdog: {100 * (g - w).mean():+.1f} pp")
