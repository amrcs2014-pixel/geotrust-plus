"""Two-minute sanity check: one seed of the sparse main configuration (paper Table IV, left).

Expected for seed 0 (end-to-end delivery): OLSR 0.725, OLSR + topology check 0.917, OLSR + watchdog 0.917,
OLSR + GeoTrust+ 0.942. Single seeds vary; the 30-seed means are 0.743 / 0.879 / 0.879 / 0.946.
"""
import sys, time
import pandas as pd
from geotrust.config import Config
from geotrust.calib import get_tau
from geotrust.runner import run

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
cfg = Config(qmin="all", rc=2.0, mean_degree=5.0, seed=seed)
t = time.time()
rows = run(cfg, get_tau(cfg), methods=("OLSR", "TOPO-OLSR", "OLSR-W", "OLSR-G"))
d = pd.DataFrame(rows).groupby("method")[["e2e_q", "incl", "size", "fex"]].mean()
print(d.rename(columns={"e2e_q": "e2e delivery", "incl": "attacker incl.", "size": "|MPR|",
                        "fex": "honest excl."}).round(3).to_string())
print(f"{time.time() - t:.0f} s on {cfg.device}")
