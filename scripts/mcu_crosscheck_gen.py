"""Generate random verifier neighbourhoods and the reciprocity convictions of the reference Python
verifier (geotrust.verify.Verifier.stats + exclude, R rule only) for the C cross-check
(embedded/crosscheck.c).  Writes embedded/crosscheck_cases.txt."""
import numpy as np
from geotrust.config import Config
from geotrust.verify import Verifier

rng = np.random.default_rng(2026)
cfg = Config(device="cpu")
TAU = {"R": 0.0428, "K": np.inf, "P": 1.0, "CM": np.inf}
ver = Verifier(cfg, TAU, device="cpu")
cases = []
for c in range(400):
    n = int(rng.integers(4, 31))                      # neighbours of verifier 0
    N = n + 1 + int(rng.integers(0, 6))               # plus some non-neighbours
    pos = rng.uniform(0, cfg.rc * 1.2, (N, 3))
    true = np.linalg.norm(pos[:, None] - pos[None], axis=2)
    shared = rng.normal(0, 0.03, (N, N)); shared = np.triu(shared, 1); shared += shared.T
    CD = true + shared + rng.normal(0, 0.01, (N, N))
    CD[true > cfg.rc] = np.nan
    np.fill_diagonal(CD, np.nan)
    own = np.full((N, N), np.nan)
    G1 = np.sort(rng.choice(np.arange(1, N), n, replace=False))
    own[0, G1] = true[0, G1] + shared[0, G1] + rng.normal(0, 0.01, n)
    CD[G1, 0] = own[0, G1] + rng.normal(0, 0.014, n)
    liars = rng.choice(G1, max(0, int(rng.binomial(n, 0.15))), replace=False)
    for v in liars:
        kind = rng.integers(0, 4)
        row = np.flatnonzero(~np.isnan(CD[v]))
        if kind == 0:                                  # shorten some links
            sel = row[rng.random(row.size) < 0.6]; CD[v, sel] *= 0.6
        elif kind == 1:                                # fake links to non-listed nodes
            cand = np.flatnonzero(np.isnan(CD[v])); cand = cand[cand != v]
            if cand.size:
                sel = rng.choice(cand, min(3, cand.size), replace=False); CD[v, sel] = rng.uniform(0.3, cfg.rc, sel.size)
        elif kind == 2:                                # omission of links (framing)
            sel = row[rng.random(row.size) < 0.4]; CD[v, sel] = np.nan
        else:                                          # lie to the verifier only
            CD[v, 0] = own[0, v] * 0.5
    strong = float(rng.choice([0.7, 1.0]))
    out, edges, _ = ver.stats(own, CD, np.zeros((N, N)), None, [0], np.random.default_rng(c))
    mask, flags, _ = ver.exclude(0, G1, out, edges, use_cm=False, strong=strong)
    cases.append((n, strong * cfg.rc, CD[np.ix_(G1, G1)], CD[G1, 0], own[0, G1], flags["R"].astype(int)))

with open("embedded/crosscheck_cases.txt", "w") as fh:
    fh.write(f"{len(cases)} {TAU['R']:.6f}\n")
    for n, sd, D, Dvi, own0, R in cases:
        f = lambda a: " ".join("nan" if np.isnan(x) else f"{x:.7g}" for x in np.ravel(a))
        fh.write(f"{n} {sd:.6f}\n{f(D)}\n{f(Dvi)}\n{f(own0)}\n{' '.join(map(str, R))}\n")
print(len(cases), "cases; mean blamed per case:", np.mean([c[5].sum() for c in cases]).round(2))
