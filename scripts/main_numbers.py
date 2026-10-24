"""Numbers of the main-results section (Table main, attack gains, insider fraction, Table robust) from the
faithful-model campaigns ru_sparse / ru_dense (GT_OBS=ideal: grid_rfc / grid_rfcdense).
Usage: python scripts/main_numbers.py"""
import os
import numpy as np, pandas as pd
from scipy import stats

OBS = os.environ.get("GT_OBS", "ru")
SRC = ({"sparse": "grid_ru_sparse", "dense": "grid_ru_dense"} if OBS == "ru" else
       {"sparse": "grid_rfc", "dense": "grid_rfcdense"})
M = ["GRR", "OLSR", "TOPO-OLSR", "OLSR-W", "OLSR-G"]


def p_of(a, b):
    d = (a - b).dropna()
    return stats.wilcoxon(d).pvalue if d.abs().sum() > 0 else 1.0


def stars(p):
    return "**" if p < 1e-4 else ("*" if p < 0.01 else "")


S = {}
for reg, f in SRC.items():
    d = pd.read_parquet(f"eval/results/{f}.parquet")
    S[reg] = d.groupby(["factor", "level", "method", "seed"])[["e2e_q", "incl", "size", "fex", "fex_auth",
                                                                 "tpr"]].mean()


def cell(reg, fac, lev, m):
    return S[reg].xs((fac, lev, m), level=["factor", "level", "method"])


print("== Table main")
for m in M:
    row = [m]
    for reg in ("sparse", "dense"):
        c = cell(reg, "main", "None", m)
        row += [f"{c.e2e_q.mean():.3f}", f"{c.incl.mean():.3f}", f"{c['size'].mean():.2f}"]
    print(" & ".join(row))
for reg in ("sparse", "dense"):
    g = cell(reg, "main", "None", "OLSR-G").e2e_q
    gains = {m: (100 * (g - cell(reg, "main", "None", m).e2e_q).mean(), p_of(g, cell(reg, "main", "None", m).e2e_q))
             for m in ("OLSR", "OLSR-W", "TOPO-OLSR")}
    c = cell(reg, "main", "None", "OLSR-G")
    print(f"[{reg}] gains: " + "; ".join(f"{m} {v:+.1f} (p={p:.1g})" for m, (v, p) in gains.items()),
          f"| honest excl {100 * c.fex.mean():.1f}% of which TESLA/no usable RU {100 * c.fex_auth.mean():.1f}%",
          f"| liars convicted {100 * c.tpr.mean():.1f}%")

print("\n== Attacks: GeoTrust+ minus watchdog (points, p)")
for reg in ("sparse", "dense"):
    out = []
    for a in ("A1", "A2", "A9", "A10a", "A10b", "A12", "A13", "A14", "PHY"):
        g, w, o = (cell(reg, "attack", a, m).e2e_q for m in ("OLSR-G", "OLSR-W", "OLSR"))
        out.append(f"{a} {100 * (g - w).mean():+.1f} (p={p_of(g, w):.1g}; vs OLSR p={p_of(g, o):.1g})")
    print(f"[{reg}] " + "; ".join(out))

print("\n== Insider fraction 20%")
for reg in ("sparse", "dense"):
    print(reg, {m: round(cell(reg, "att_frac", "0.2", m).e2e_q.mean(), 3) for m in ("OLSR-G", "OLSR", "OLSR-W")})

print("\n== Table robust (sparse)")
for fac, lev, name in [("main", "None", "main (50 agents, 1 m/s)"), ("n", "10", "N=10"), ("n", "30", "N=30"),
                       ("n", "100", "N=100"), ("speed", "0.2", "v=0.2"), ("speed", "2.0", "v=2"),
                       ("n_obst", "8", "8 NLoS")]:
    o, w, g = (cell("sparse", fac, lev, m) for m in ("OLSR", "OLSR-W", "OLSR-G"))
    vo, vw = 100 * (g.e2e_q - o.e2e_q).mean(), 100 * (g.e2e_q - w.e2e_q).mean()
    print(f"{name} & {o.e2e_q.mean():.3f} & {w.e2e_q.mean():.3f} & {g.e2e_q.mean():.3f} & "
          f"${vo:+.1f}${stars(p_of(g.e2e_q, o.e2e_q))} & ${vw:+.1f}${stars(p_of(g.e2e_q, w.e2e_q))} & "
          f"{100 * g.fex.mean():.1f}\\% (p_vsWD={p_of(g.e2e_q, w.e2e_q):.2g})")
