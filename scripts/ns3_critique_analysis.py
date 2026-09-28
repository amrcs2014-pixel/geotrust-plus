"""ns-3 critique (Section IV, O2): RFC 3626 vs GRR under Zhou's Eq. (3), 30 drones, 20 seeds per cell."""
import numpy as np, pandas as pd
from scipy import stats

d = pd.read_csv("eval/results/ns3_qall_full.csv")
NAME = {"rfc": "OLSR (RFC 3626)", "grr": "GRR [Zhou]", "grrw": "GRR + watchdog", "geotrust": "GRR + GeoTrust (v1)"}
pd.set_option("display.width", 200)
t = d.groupby(["attack", "mpr"]).agg(pdr_post=("pdr_post", "mean"), tc_node_s=("tc_per_node_s_post", "mean"),
                                     relays=("mpr_size", "mean"), n=("seed", "count")).round(3)
print(t.rename(index=NAME, level=1).to_string())
print("\nRFC 3626 minus GRR (post-onset delivery), paired Wilcoxon")
for a, g in d.groupby("attack"):
    x, y = g[g.mpr == "rfc"].set_index("seed").pdr_post.align(g[g.mpr == "grr"].set_index("seed").pdr_post, join="inner")
    print(f"  {a:12s} {float((x - y).mean()):+.3f}  p={stats.wilcoxon(x, y).pvalue:.1e}  n={len(x)}")
