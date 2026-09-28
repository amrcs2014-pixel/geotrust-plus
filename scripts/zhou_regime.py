"""Replicate Zhou et al. Figs. 13-14 regime: Q_min over ALL relays (Eq. 3), R_c = 2 m, sparse swarm."""
import numpy as np, pandas as pd, sys
from concurrent.futures import ProcessPoolExecutor
from geotrust.config import Config
from geotrust.sim import World, honest_claims
from geotrust import mpr

DEG = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0

def job(a):
    n, seed = a
    cfg = Config(n=n, rc=2.0, mean_degree=DEG, seed=seed, attack="none", att_frac=0.0, qmin="all")
    w = World(cfg, np.random.default_rng(seed))
    for r in range(cfg.warmup + 1):
        if r: w.step()
        D = w.true_dist(); meas, adj = w.measure(D); w.update_links(D, adj, meas)
    CD, CP, _, _ = honest_claims(w, meas); Pown = w.fused(meas)
    fwd = {m: set() for m in ("GRR", "ILP", "SG", "OLSR", "FLOOD")}; tot = {m: 0 for m in fwd}
    for i in range(n):
        G1 = np.flatnonzero(adj[i])
        if G1.size == 0: continue
        cl = ~np.isnan(CD[G1]); U = np.flatnonzero(cl.any(0) & ~adj[i]); U = U[U != i]
        C, t = mpr.contributions(Pown[i, G1], np.where(cl[:, U], CP[G1][:, U], np.nan), cfg.gamma, "all")
        for m, fn in (("GRR", mpr.grr_zhou), ("ILP", lambda C, t: mpr.ilp(C, t, f=0)),
                      ("SG", lambda C, t: mpr.robust_greedy(C, t, f=0)), ("OLSR", mpr.olsr),
                      ("FLOOD", mpr.flooding)):
            S = G1[fn(C, t)] if G1.size else []
            fwd[m] |= set(np.asarray(S).tolist()); tot[m] += len(S)
    return [dict(n=n, seed=seed, method=m, forwarders=len(fwd[m]), selections=tot[m]) for m in fwd]

if __name__ == "__main__":
    with ProcessPoolExecutor(6) as ex:
        rows = sum(ex.map(job, [(n, s) for n in (10, 20, 30, 40, 50) for s in range(10)]), [])
    df = pd.DataFrame(rows)
    print(f"mean degree target {DEG}")
    print("Fig.13 analogue: distinct forwarding nodes (mean over 10 instances)")
    print(df.pivot_table(index="n", columns="method", values="forwarders").round(1).to_string())
    print("Fig.14 analogue: sum of selections")
    print(df.pivot_table(index="n", columns="method", values="selections").round(1).to_string())
