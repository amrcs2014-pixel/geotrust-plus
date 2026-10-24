"""Effect of the reciprocity blame rule: v1.0 greedy vertex cover vs. the order-independent threshold rule with
collusion bound f = 1, 2, 3 (GeoTrust+ only).  Seeds are paired: the greedy numbers come from the archived campaigns
(grid_rfc / grid_rfcdense, produced with the greedy rule) and are re-checked by the blame=greedy cell of the new
campaign.  Writes eval/results/blame_summary.csv.
Usage: python scripts/blame_analysis.py"""
import numpy as np, pandas as pd
from scipy import stats

R = "eval/results"
REG = {"sparse": ("grid_blame_sparse.parquet", "v1_greedy/grid_rfc.parquet"),
       "dense": ("grid_blame_dense.parquet", "v1_greedy/grid_rfcdense.parquet")}
COLS = ["e2e_q", "fex", "incl", "tpr"]


def seedmeans(d):
    return d[d.method == "OLSR-G"].groupby(["factor", "level", "seed"])[COLS].mean().reset_index()


rows = []
import os
for reg, (new_f, old_f) in REG.items():
    if not os.path.exists(f"{R}/{new_f}"):
        print(f"[{reg}] {new_f} not ready"); continue
    new, old = seedmeans(pd.read_parquet(f"{R}/{new_f}")), seedmeans(pd.read_parquet(f"{R}/{old_f}"))
    # sanity: the greedy cell of the new campaign reproduces the archived main configuration
    g_new = new[new.factor == "blame"].set_index("seed")[COLS]
    g_old = old[old.factor == "main"].set_index("seed")[COLS]
    common = g_new.index.intersection(g_old.index)
    diff = float(np.nanmax(np.abs(g_new.loc[common].values - g_old.loc[common].values)))
    print(f"[{reg}] greedy re-run vs archive: {len(common)} seeds, max |diff| = {diff:.2e}")
    cells = [("main", "None", "threshold f=1")] + [("coll_f", l, f"threshold f={l}") for l in ("2", "3")]
    cells += [("attack", a, "threshold f=1") for a in sorted(new[new.factor == "attack"].level.unique())]
    for fac, lev, rule in cells:
        t = new[(new.factor == fac) & (new.level == lev)].set_index("seed")
        ref = old[(old.factor == ("main" if fac in ("main", "coll_f") else fac))
                  & (old.level == ("None" if fac in ("main", "coll_f") else lev))].set_index("seed")
        s = t.index.intersection(ref.index)
        r = {"regime": reg, "setting": "main" if fac in ("main", "coll_f") else lev, "rule": rule, "seeds": len(s)}
        for c in COLS:
            a, b = t.loc[s, c].astype(float), ref.loc[s, c].astype(float)
            ok = a.notna() & b.notna()
            r[f"{c}_greedy"], r[f"{c}_new"] = b[ok].mean(), a[ok].mean()
            r[f"{c}_p"] = (stats.wilcoxon(a[ok], b[ok]).pvalue if ok.sum() >= 5 and (a[ok] - b[ok]).abs().sum() > 0
                           else 1.0)
        rows.append(r)

out = pd.DataFrame(rows)
out.to_csv(f"{R}/blame_summary.csv", index=False)
show = out[["regime", "setting", "rule", "seeds", "e2e_q_greedy", "e2e_q_new", "e2e_q_p", "fex_greedy", "fex_new",
            "fex_p", "tpr_greedy", "tpr_new"]]
with pd.option_context("display.width", 200, "display.max_columns", 20):
    print(show.round(4).to_string(index=False))
