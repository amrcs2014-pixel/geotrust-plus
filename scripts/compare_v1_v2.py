"""Compare the v2 campaigns (threshold blame rule, f=2) with the archived v1 campaigns (greedy cover), cell by cell.
Methods that do not use the reciprocity blame must be identical; GeoTrust+ cells report the change.
Usage: python scripts/compare_v1_v2.py [regime ...]"""
import os, sys
import numpy as np, pandas as pd

R = "eval/results"
COLS = ["e2e_q", "fex", "incl", "tpr"]


def load(path):
    d = pd.read_parquet(path)
    keep = [c for c in COLS if c in d.columns]
    return d.groupby(["factor", "level", "method"])[keep].mean()


regimes = sys.argv[1:] or ["rfc", "rfcdense", "abl_sparse", "abl_dense", "sens_sparse", "sens_dense", "net_sparse",
                           "net_dense"]
for reg in regimes:
    new_p, old_p = f"{R}/grid_{reg}.parquet", f"{R}/v1_greedy/grid_{reg}.parquet"
    if reg.startswith(("abl", "sens")) and os.path.exists(f"{R}/v1_greedy/grid_{reg}_parts.parquet"):
        old_p = f"{R}/v1_greedy/grid_{reg}_parts.parquet"
    if not (os.path.exists(new_p) and os.path.exists(old_p)):
        print(f"[{reg}] missing"); continue
    a, b = load(new_p), load(old_p)
    j = a.join(b, lsuffix="_v2", rsuffix="_v1", how="inner")
    meth = j.index.get_level_values("method")
    gt = meth.str.startswith(("OLSR-G", "OLSR-V", "GT"))
    d = (j[[f"{c}_v2" for c in COLS if f"{c}_v2" in j]].values - j[[f"{c}_v1" for c in COLS if f"{c}_v1" in j]].values)
    other = np.nanmax(np.abs(d[~gt])) if (~gt).any() else 0.0
    print(f"\n[{reg}] cells {len(j)}; max |change| in methods without blame: {other:.2e}")
    g = j[gt].copy()
    g["d_e2e_pp"] = 100 * (g["e2e_q_v2"] - g["e2e_q_v1"])
    g["d_fex_pp"] = 100 * (g["fex_v2"] - g["fex_v1"])
    g["d_tpr_pp"] = 100 * (g["tpr_v2"] - g["tpr_v1"]) if "tpr_v2" in g else np.nan
    show = g[["e2e_q_v1", "e2e_q_v2", "d_e2e_pp", "fex_v1", "fex_v2", "d_fex_pp", "d_tpr_pp"]].round(4)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(show.to_string())
    print(f"  GeoTrust+ cells: |d e2e| max {g.d_e2e_pp.abs().max():.2f} pp, |d fex| max {g.d_fex_pp.abs().max():.2f} pp")
