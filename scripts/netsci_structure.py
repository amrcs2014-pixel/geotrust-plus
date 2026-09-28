"""Network-science characterisation of the simulated swarms (no attacks, no routing):
the connectivity graph is a 3-D random geometric graph (RGG) with random-waypoint mobility.

Per regime (sparse R_c=2 m / dense R_c=4 m, 50 agents) and speed (0.2 / 1 / 2 m/s), 20 seeds, rounds 10..39:
  degree mean/sd, average clustering, transitivity, giant-component fraction, mean shortest path, diameter,
  betweenness concentration (Gini, top-10% share), link churn per ranging period and per TC interval,
  and *auditability*: for a verifier i and neighbour v, the fraction of v's links whose other end is
  also a neighbour of i (so that i hears both ends and can apply the reciprocity test).
Also an Erdos-Renyi reference with the same mean degree.  Writes eval/results/netsci_structure.csv
"""
import numpy as np, pandas as pd, networkx as nx
from geotrust.config import Config
from geotrust.sim import World

MAIN = {"sparse": dict(qmin="all", rc=2.0, mean_degree=5.0), "dense": dict(qmin="all", rc=4.0, mean_degree=12.0)}
rows = []


def gini(x):
    x = np.sort(np.asarray(x, float))
    return float((2 * np.arange(1, x.size + 1) - x.size - 1) @ x / (x.size * x.sum())) if x.sum() > 0 else np.nan


def metrics(A, prev, prev5):
    n = A.shape[0]
    g = nx.from_numpy_array(A.astype(int))
    deg = A.sum(1)
    comps = sorted(nx.connected_components(g), key=len, reverse=True)
    gc = g.subgraph(comps[0])
    bc = np.array(list(nx.betweenness_centrality(g).values()))
    top = np.sort(bc)[::-1][: max(1, n // 10)].sum() / bc.sum() if bc.sum() > 0 else np.nan
    aud = []
    for i in range(n):
        Ni = np.flatnonzero(A[i])
        for v in Ni:
            Nv = np.flatnonzero(A[v]); Nv = Nv[Nv != i]
            if Nv.size:
                aud.append(np.isin(Nv, Ni).mean())
    iu = np.triu_indices(n, 1)
    churn = lambda P: (A[iu] != P[iu]).sum() / max(1, A[iu].sum()) if P is not None else np.nan
    return dict(deg=deg.mean(), deg_sd=deg.std(), clust=nx.average_clustering(g), trans=nx.transitivity(g),
                giant=len(comps[0]) / n, aspl=nx.average_shortest_path_length(gc) if len(comps[0]) > 1 else np.nan,
                diam=nx.diameter(gc) if len(comps[0]) > 1 else np.nan, btw_gini=gini(bc), btw_top10=top,
                audit=float(np.mean(aud)) if aud else np.nan, churn_T=churn(prev), churn_TC=churn(prev5))


for reg, kw in MAIN.items():
    for speed in (0.2, 1.0, 2.0):
        for seed in range(20):
            cfg = Config(seed=seed, speed=speed, **kw)
            w = World(cfg, np.random.default_rng(seed))
            hist = []
            for r in range(40):
                if r:
                    w.step()
                D = w.true_dist()
                A = (D < cfg.rc) & ~np.eye(cfg.n, dtype=bool)
                hist.append(A)
                if r >= 10 and r % 5 == 0:
                    m = metrics(A, hist[-2], hist[-6])
                    rows.append(dict(regime=reg, speed=speed, seed=seed, round=r, **m))
            # Erdos-Renyi reference with the same mean degree
            p = hist[-1].sum() / (cfg.n * (cfg.n - 1))
            er = nx.gnp_random_graph(cfg.n, p, seed=seed)
            rows.append(dict(regime=reg + "_ER", speed=speed, seed=seed, round=-1,
                             **metrics(nx.to_numpy_array(er).astype(bool), None, None)))
d = pd.DataFrame(rows)
d.to_csv("eval/results/netsci_structure.csv", index=False)
pd.set_option("display.width", 220)
print(d.groupby(["regime", "speed"]).mean(numeric_only=True).drop(columns=["seed", "round"]).round(3).to_string())
