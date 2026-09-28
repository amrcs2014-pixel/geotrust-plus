"""Attack library. Insiders hold valid keys and know the true positions of every node.

  A1   PLR forgery          honest distances, PLR claimed 0
  A2   coverage inflation   claims 5 real non-neighbours (rc, 2rc] at a small fixed distance, PLR 0
  A9   wormhole (pairs)     each colluder also claims its partner's neighbours with the partner's data
  A10a coherent fake pos.   all claims computed from a fake position x~ = x + 1 m toward its 2-hop area
  A10b adaptive selective   truthful for d < rc/2; claims 0.6*d for rc/2 <= d < 1.5rc (incl. non-neighbours),
                            PLR consistent with the forged distance -> passes naive PLR/triangle checks
  PHY  A4 Ghost-Peak-class distance reduction on attacker links (p=0.04 per exchange)
       A5 distance enlargement (+1..3 m, p=0.9) on links of the attacker's 3 nearest honest neighbours
Every attacker additionally black-holes traffic it is asked to forward (A7, q=1).
"""
import numpy as np
from .config import ATTACK_TYPES

# A12 stealth inflation (adaptive vs. reciprocity): truthful everywhere, <= 3 fake links to real
# non-neighbours in (rc, 1.5rc] claimed at 0.75-0.95 rc (beyond the strong one-sided rule), PLR consistent
# A13 collusion: colluders confirm each other's FAKE links (both ends report the same distance, so
#      reciprocity passes); every attacker claims every other attacker within 2.5 rc at 0.85 rc
# A14 framing by omission: truthful, but omits its links to its 2 nearest honest neighbours so that their
#      (true) claims become one-sided
EXTRA_TYPES = ("A12", "A13", "A14")
from .sim import plr


class Adversary:
    def __init__(self, cfg, rng, world=None):
        self.cfg, self.rng = cfg, rng
        n = cfg.n
        k = 0 if cfg.attack == "none" or cfg.att_frac <= 0 else max(1, int(round(cfg.att_frac * n)))
        place = getattr(cfg, "att_place", "random")
        if k and place != "random" and world is not None:
            # targeted placement: the k most central nodes of the initial connectivity graph
            import networkx as nx
            D = world.true_dist()
            g = nx.from_numpy_array(((D < cfg.rc) & ~np.eye(n, dtype=bool)).astype(int))
            c = dict(g.degree()) if place == "degree" else nx.betweenness_centrality(g)
            score = np.array([c[v] for v in range(n)], float) + 1e-9 * rng.random(n)   # random tie-break
            self.ids = np.sort(np.argsort(-score)[:k])
        else:
            self.ids = np.sort(rng.choice(n, k, replace=False)) if k else np.array([], int)
        self.is_att = np.zeros(n, bool)
        self.is_att[self.ids] = True
        self.types, self.partner = {}, {}
        if cfg.attack in ATTACK_TYPES + EXTRA_TYPES:
            pool = [cfg.attack] * k
        elif cfg.attack == "all":
            pool = [ATTACK_TYPES[i % len(ATTACK_TYPES)] for i in range(k)]
        else:                                       # "PHY" or "none": no RU forgery
            pool = [None] * k
        for a, t in zip(self.ids, pool):
            self.types[a] = t
        # wormhole colluders are paired; an unpaired one falls back to A10b
        wh = [a for a in self.ids if self.types[a] == "A9"]
        for x, y in zip(wh[0::2], wh[1::2]):
            self.partner[x], self.partner[y] = y, x
        for a in wh:
            if a not in self.partner:
                self.types[a] = "A10b"
        self.phy = cfg.attack in ("PHY", "all") and k > 0
        self.fake = {}          # A2: fixed fake claims per attacker {u: d}

    # ---------------------------------------------------------------- PHY layer
    def phy_hook(self, D):
        rng, rc = self.rng, self.cfg.rc
        honest = ~self.is_att

        def hook(shared, adj):
            if not self.phy:
                return shared
            s = shared.copy()
            for a in self.ids:
                nb = np.flatnonzero(adj[a])
                # A4: distance reduction on links (i, a) -> attacker looks close
                hit = nb[rng.random(nb.size) < 0.04]
                s[a, hit] = s[hit, a] = rng.uniform(0.2, 1.0, hit.size)
                # A5: enlargement on links of the 3 nearest honest neighbours
                hn = nb[honest[nb]]
                tgt = hn[np.argsort(D[a, hn])[:3]]
                for j in tgt:
                    ii = np.flatnonzero(adj[j] & adj[a])        # attacker must be in range of i
                    ii = ii[rng.random(ii.size) < 0.9]
                    add = rng.uniform(1.0, 3.0, ii.size)
                    s[ii, j] += add
                    s[j, ii] += add
            return s
        return hook

    # ------------------------------------------------------------- RU forgery
    def forge(self, world, D, adj, CD, CP):
        """Overwrite attacker rows of claim matrices (in place)."""
        rng, c = self.rng, self.cfg
        rc, n = c.rc, c.n
        for a in self.ids:
            t = self.types[a]
            if t is None:
                continue
            nb = adj[a]
            if t == "A1":
                CP[a, nb] = 0.0
            elif t == "A2":
                fk = self.fake.setdefault(a, {})
                for u in list(fk):
                    if D[a, u] <= rc or D[a, u] > 2 * rc:
                        del fk[u]
                cand = np.flatnonzero((D[a] > rc) & (D[a] <= 2 * rc))
                cand = [u for u in cand if u not in fk]
                for u in rng.permutation(cand)[: max(0, 5 - len(fk))]:
                    fk[u] = rng.uniform(1.0, 0.6 * rc)
                CP[a, nb] = 0.0
                for u, d in fk.items():
                    CD[a, u] = d
                    CP[a, u] = 0.0
            elif t == "A9":
                b = self.partner[a]
                extra = adj[b] & ~nb
                extra[a] = False
                CD[a, extra] = CD[b, extra]
                CP[a, extra] = CP[b, extra]
            elif t == "A10a":
                far = (D[a] > rc) & (D[a] <= 2 * rc)
                tgt = world.pos[far].mean(0) if far.any() else world.pos.mean(0)
                dirn = tgt - world.pos[a]
                xf = world.pos[a] + dirn / max(np.linalg.norm(dirn), 1e-9) * 1.0
                df = np.linalg.norm(world.pos - xf, axis=1) + rng.normal(0, c.sigma, n)
                m = df < rc
                m[a] = False
                CD[a] = np.nan
                CP[a] = np.nan
                CD[a, m] = df[m]
                CP[a, m] = plr(df[m])
            elif t == "A12":
                fk = self.fake.setdefault(a, {})
                for u in list(fk):
                    if D[a, u] <= rc or D[a, u] > 1.5 * rc:
                        del fk[u]
                cand = [u for u in np.flatnonzero((D[a] > rc) & (D[a] <= 1.5 * rc)) if u not in fk]
                for u in rng.permutation(cand)[: max(0, 3 - len(fk))]:
                    fk[u] = rng.uniform(0.75, 0.95) * rc
                for u, d in fk.items():
                    CD[a, u] = d
                    CP[a, u] = plr(d)
            elif t == "A13":
                mates = [b for b in self.ids if b != a and rc < D[a, b] <= 2.5 * rc]
                for b in mates:
                    CD[a, b] = 0.85 * rc
                    CP[a, b] = plr(0.85 * rc)
            elif t == "A14":
                hn = np.flatnonzero(nb & ~self.is_att)
                for u in hn[np.argsort(D[a, hn])[:2]]:
                    CD[a, u] = np.nan
                    CP[a, u] = np.nan
            elif t == "A10b":
                lie = (D[a] >= rc / 2) & (D[a] < 1.5 * rc)
                lie[a] = False
                dl = 0.6 * D[a, lie] + rng.normal(0, c.sigma, lie.sum())
                CD[a, lie] = dl
                CP[a, lie] = plr(dl)
        return CD, CP
