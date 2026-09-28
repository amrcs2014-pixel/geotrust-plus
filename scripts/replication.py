"""WP1 replication + theory check (attack-free).
For every honest node instance: relay-set sizes of FLOOD / OLSR / GRR / SG / ILP, robust greedy vs
robust ILP (f=1,2), GRR's bound k and the Wolsey bound 1+ln(G_f(N)/eps).
Writes eval/results/replication.parquet and eval/replication.json (acceptance test)."""
import json, os, time
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from geotrust.config import Config
from geotrust.sim import World, honest_claims
from geotrust import mpr


def instances(cfg):
    rng = np.random.default_rng(cfg.seed)
    w = World(cfg, rng)
    out = []
    for r in range(cfg.warmup + 1):
        if r:
            w.step()
        D = w.true_dist()
        meas, adj = w.measure(D)
        w.update_links(D, adj, meas)
    CD, CP, _, _ = honest_claims(w, meas)
    Pown = w.fused(meas)
    for i in range(cfg.n):
        G1 = np.flatnonzero(adj[i])
        if G1.size == 0:
            continue
        cl = ~np.isnan(CD[G1])
        U = np.flatnonzero(cl.any(0) & ~adj[i])
        U = U[U != i]
        if U.size == 0:
            continue
        Pvu = np.where(cl[:, U], CP[G1][:, U], np.nan)
        out.append(mpr.contributions(Pown[i, G1], Pvu, cfg.gamma, cfg.qmin))
    return out


def job(args):
    n, rc, seed = args
    cfg = Config(n=n, rc=rc, seed=seed, attack="none", att_frac=0.0)
    rows = []
    for C, t in instances(cfg):
        k = int((t > 1e-9).sum())
        if k == 0:
            continue
        r = dict(n=n, rc=rc, seed=seed, n1=C.shape[0], k=k)
        for name, fn in (("FLOOD", mpr.flooding), ("OLSR", mpr.olsr), ("GRR", mpr.grr_zhou), ("GRR2", mpr.grr)):
            r[name] = len(fn(C, t))
        t0 = time.perf_counter(); r["SG"] = len(mpr.robust_greedy(C, t, f=0)); r["t_SG"] = time.perf_counter() - t0
        t0 = time.perf_counter(); r["ILP"] = len(mpr.ilp(C, t, f=0)); r["t_ILP"] = time.perf_counter() - t0
        r["wolsey0"] = mpr.wolsey_bound(C, t, 0)
        for f in (1, 2):
            if f == 2 and C.shape[0] > 25:
                continue
            r[f"RG{f}"] = len(mpr.robust_greedy(C, t, f=f))
            r[f"RILP{f}"] = len(mpr.ilp(C, t, f=f))
            r[f"wolsey{f}"] = mpr.wolsey_bound(C, t, f)
        rows.append(r)
    return rows


if __name__ == "__main__":
    jobs = [(n, rc, s) for rc in (2.0, 4.0) for n in (10, 20, 30, 40, 50) for s in range(10)]
    with ProcessPoolExecutor(6) as ex:
        rows = sum(ex.map(job, jobs), [])
    df = pd.DataFrame(rows)
    os.makedirs("eval/results", exist_ok=True)
    df.to_parquet("eval/results/replication.parquet")
    g = df[df.rc == 2.0]
    acc = {
        "instances": int(len(g)),
        "mean_size": {m: float(g[m].mean()) for m in ("ILP", "SG", "GRR", "GRR2", "OLSR", "FLOOD")},
        "GRR_over_ILP_mean_ratio": float((g.GRR / g.ILP).mean()),
        "GRR2_over_ILP_mean_ratio": float((g.GRR2 / g.ILP).mean()),
        "GRR_over_ILP_mean_ratio_all_rc": float((df.GRR / df.ILP).mean()),
        "SG_over_ILP_mean_ratio_all_rc": float((df.SG / df.ILP).mean()),
        "SG_over_ILP_mean_ratio": float((g.SG / g.ILP).mean()),
        "acceptance_GRR_ILP_le_1.15": bool((g.GRR / g.ILP).mean() <= 1.15),
        "RG1_over_RILP1": float((df.RG1 / df.RILP1).mean()),
        "RG2_over_RILP2": float((df.RG2 / df.RILP2).dropna().mean()),
        "worst_RG1_over_RILP1": float((df.RG1 / df.RILP1).max()),
    }
    json.dump(acc, open("eval/replication.json", "w"), indent=1)
    print(json.dumps(acc, indent=1))
    print(df.groupby(["rc", "n"])[["k", "n1", "ILP", "SG", "GRR", "OLSR", "FLOOD", "RG1", "RILP1"]].mean().round(2))
