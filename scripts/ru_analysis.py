"""Main results under the faithful observation model (obs="ru") next to the ideal model (both with the threshold
rule, f=2): GeoTrust+ (OLSR-G), OLSR + watchdog and plain OLSR per cell, 30 seeds, paired Wilcoxon tests of GeoTrust+
against the watchdog.  Writes eval/results/ru_summary.csv.   Usage: python scripts/ru_analysis.py"""
import numpy as np, pandas as pd
from scipy import stats

R = "eval/results"
REG = {"sparse": ("grid_ru_sparse.parquet", "grid_rfc.parquet"),
       "dense": ("grid_ru_dense.parquet", "grid_rfcdense.parquet")}


def seedmeans(path):
    d = pd.read_parquet(f"{R}/{path}")
    return d.groupby(["factor", "level", "method", "seed"])[["e2e_q", "fex", "tpr", "incl"]].mean()


rows = []
for reg, (ru_f, id_f) in REG.items():
    ru, ide = seedmeans(ru_f), seedmeans(id_f)
    for fac, lev in ru.index.droplevel(["method", "seed"]).unique():
        r = {"regime": reg, "setting": "main" if fac == "main" else f"{fac}={lev}"}
        for tag, X in (("ru", ru), ("ideal", ide)):
            try:
                c = X.xs((fac, lev), level=["factor", "level"])
            except KeyError:
                continue
            g, w, o = (c.xs(m, level="method") for m in ("OLSR-G", "OLSR-W", "OLSR"))
            s = g.index.intersection(w.index)
            r[f"G_{tag}"], r[f"W_{tag}"], r[f"OLSR_{tag}"] = g.e2e_q.mean(), w.e2e_q.mean(), o.e2e_q.mean()
            r[f"fex_{tag}"], r[f"tpr_{tag}"], r[f"incl_{tag}"] = g.fex.mean(), g.tpr.mean(), g.incl.mean()
            diff = g.loc[s, "e2e_q"] - w.loc[s, "e2e_q"]
            r[f"p_GvW_{tag}"] = stats.wilcoxon(diff).pvalue if diff.abs().sum() > 0 else 1.0
        rows.append(r)

out = pd.DataFrame(rows)
out.to_csv(f"{R}/ru_summary.csv", index=False)
cols = ["regime", "setting", "G_ideal", "G_ru", "W_ru", "OLSR_ru", "p_GvW_ru", "fex_ideal", "fex_ru", "tpr_ideal",
        "tpr_ru"]
with pd.option_context("display.width", 250, "display.max_rows", 100):
    print(out[[c for c in cols if c in out]].round(4).to_string(index=False))
