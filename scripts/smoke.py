import time, sys, numpy as np, pandas as pd, torch
from geotrust.config import Config
from geotrust.calib import calibrate
from geotrust.runner import run
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
cfg = Config(seed=1, attack=sys.argv[1] if len(sys.argv) > 1 else "all")
t=time.time(); tau = calibrate(cfg, seeds=3); print("tau", {k: round(v,3) for k,v in tau.items()}, f"{time.time()-t:.1f}s")
t=time.time(); rows = run(cfg, tau); print(f"run {time.time()-t:.1f}s")
df = pd.DataFrame(rows)
print(df.groupby("method")[["incl","size","cov","qos","fex","tpr","ratio","pdr","tx","rt_ms","cap"]].mean().round(3))
print(df[df.method=="GeoTrust"][[c for c in df if c.startswith(("fex_","tpr_"))]].mean().round(3))
