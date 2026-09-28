import pandas as pd, numpy as np
from scipy import stats
from concurrent.futures import ProcessPoolExecutor
from geotrust.config import Config
from geotrust.calib import get_tau
from geotrust.runner import run
M = ("FLOOD", "GRR", "TOPO-GRR", "GRR-W", "GT+f0")
SC = {"main": {}, "A1": dict(attack="A1"), "A10b": dict(attack="A10b"), "A13": dict(attack="A13"), "PHY": dict(attack="PHY")}
def job(a):
    sc, s = a
    cfg = Config(qmin="all", rc=2.0, mean_degree=5.0, seed=s, **SC[sc])
    rows = run(cfg, get_tau(cfg), methods=M)
    for r in rows: r["sc"] = sc
    return pd.DataFrame(rows).groupby(["sc", "method", "seed"])[["e2e", "e2e_q", "incl"]].mean().reset_index()
if __name__ == "__main__":
    with ProcessPoolExecutor(3) as ex:
        df = pd.concat(ex.map(job, [(sc, s) for sc in SC for s in range(10)]))
    pd.set_option("display.width", 200)
    print(df.pivot_table(index="sc", columns="method", values=["e2e", "e2e_q"]).round(3).to_string())
    for sc, g in df.groupby("sc"):
        a = g[g.method == "GT+f0"].set_index("seed").e2e_q
        for b in ("GRR", "GRR-W"):
            c = g[g.method == b].set_index("seed").e2e_q
            x, y = a.align(c, join="inner")
            p = stats.wilcoxon(x, y).pvalue if (x - y).abs().sum() > 0 else float("nan")
            print(f"{sc:5s} GeoTrust+ vs {b:6s} e2e_q diff {(x-y).mean():+.3f} p={p:.4f}")
