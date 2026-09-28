"""Swarm world: mobility, UWB ranging, link loss, EWMA/fusion and ranging-unit (RU) claims.

Every round produces two *views* of the ranging layer that share all random draws:
  raw   – what an undefended Zhou et al. stack reports/uses,
  gated – every honest node passes its own distance measurements through a kinematic gate
          (part of the GeoTrust deployment).
Claims are stored as dense matrices: row v of CD/CP is v's broadcast ranging unit
(claimed distance / claimed PLR to each neighbour, NaN = not claimed).
"""
import numpy as np
from functools import lru_cache
from .config import Config, PLR_A, PLR_B, PLR_C


NLOS_PENALTY = 2.5


def plr(d):
    return 1.0 / (np.exp(PLR_A * np.asarray(d, float) + PLR_B) + PLR_C)


@lru_cache(maxsize=None)
def cube_side(n, rc, mean_degree, flat):
    """Side length L that gives the requested mean degree (boundary effects included)."""
    rng = np.random.default_rng(12345)
    target = min(mean_degree, 0.6 * (n - 1))
    lo, hi = rc * 0.3, rc * 20
    for _ in range(40):
        L = 0.5 * (lo + hi)
        degs = []
        for _ in range(20):
            p = rng.uniform(0, L, (n, 3))
            if flat:
                p[:, 2] = rng.uniform(0, 0.3, n)
            d = np.linalg.norm(p[:, None] - p[None], axis=-1)
            degs.append(((d < rc).sum(1) - 1).mean())
        if np.mean(degs) > target:
            lo = L
        else:
            hi = L
    return 0.5 * (lo + hi)


class World:
    def __init__(self, cfg: Config, rng: np.random.Generator):
        self.cfg, self.rng = cfg, rng
        n = cfg.n
        self.L = cube_side(n, cfg.rc, cfg.mean_degree, cfg.flat)
        self.zmax = 0.3 if cfg.flat else self.L
        self.pos = self._uniform(n)
        self.wp = self._uniform(n)
        self.spd = rng.uniform(0.5, 1.0, n) * cfg.speed
        self.vel = np.zeros((n, 3))
        # per directed link state
        self.ewma = np.full((n, n), np.nan)
        self.gate = np.full((n, n), np.nan)     # gated distance estimate (GeoTrust deployment)
        self.round = 0
        # NLoS scenario: axis-aligned boxes (1-3 m) block line of sight
        self.boxes = []
        for _ in range(cfg.n_obst):
            size = rng.uniform(1.0, 3.0, 3)
            size[2] = min(size[2], self.zmax) if cfg.flat else size[2]
            lo = rng.uniform(0, self.L, 3) - size / 2
            if cfg.flat:
                lo[2], size[2] = -1.0, self.zmax + 2.0
            self.boxes.append((lo, lo + size))
        self.nlos = np.zeros((n, n), bool)

    def _uniform(self, k):
        p = self.rng.uniform(0, self.L, (k, 3))
        p[:, 2] = self.rng.uniform(0, self.zmax, k)
        return p

    # ------------------------------------------------------------------ mobility
    def step(self):
        c = self.cfg
        d = self.wp - self.pos
        dist = np.linalg.norm(d, axis=1)
        arrive = dist < self.spd * c.T
        self.wp[arrive] = self._uniform(arrive.sum())
        d = self.wp - self.pos
        dist = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
        self.vel = d / dist[:, None] * self.spd[:, None]
        self.pos = self.pos + self.vel * c.T
        self.round += 1

    # ------------------------------------------------------------------ ranging
    def true_dist(self):
        self.nlos = self._nlos()
        return np.linalg.norm(self.pos[:, None] - self.pos[None], axis=-1)

    def _nlos(self):
        """Slab test of every segment (i,j) against every box."""
        n = self.cfg.n
        out = np.zeros((n, n), bool)
        if not self.boxes:
            return out
        p0 = self.pos[:, None, :]
        d = self.pos[None, :, :] - p0
        with np.errstate(divide="ignore", invalid="ignore"):
            for lo, hi in self.boxes:
                t1, t2 = (lo - p0) / d, (hi - p0) / d
                tmin = np.nanmax(np.minimum(t1, t2), axis=-1)
                tmax = np.nanmin(np.maximum(t1, t2), axis=-1)
                out |= (tmax >= np.maximum(tmin, 0)) & (tmin <= 1)
        np.fill_diagonal(out, False)
        return out

    def true_plr(self, D):
        """True channel: Zhou LoS curve; NLoS links lose ~2.5 m-equivalent of link budget."""
        return plr(D + NLOS_PENALTY * self.nlos)

    def measure(self, D, phy_hook=None):
        """Shared DS-TWR estimate (+motion bias) with small independent per-end noise.
        phy_hook(shared) may corrupt the shared exchange (Ghost-Peak / enlargement)."""
        c, n, rng = self.cfg, self.cfg.n, self.rng
        adj = (D < c.rc) & ~np.eye(n, dtype=bool)
        # radial relative speed -> motion bias within a ranging round (Xie et al. 2026)
        u = (self.pos[None] - self.pos[:, None]) / np.maximum(D, 1e-9)[..., None]
        vr = np.abs(((self.vel[None] - self.vel[:, None]) * u).sum(-1))
        noise = rng.normal(0, c.sigma, (n, n))
        bias = rng.uniform(-1, 1, (n, n)) * vr * c.T / 2
        bias += self.nlos * rng.uniform(0.2, 0.8, (n, n))          # NLoS excess delay (positive)
        sh = np.triu(D + noise + bias, 1)
        shared = sh + sh.T
        if phy_hook is not None:
            shared = phy_hook(shared, adj)
        meas = shared + rng.normal(0, c.sigma_ind, (n, n))
        meas = np.maximum(meas, 0.05)
        meas[~adj] = np.nan
        return meas, adj

    def update_links(self, D, adj, meas):
        """Bernoulli losses on the true channel, EWMA of observed PLR, kinematic gate."""
        c = self.cfg
        lost = self.rng.random(D.shape) < self.true_plr(D)
        new = adj & np.isnan(self.ewma)
        self.ewma[new] = plr(meas[new])
        upd = adj & ~new
        self.ewma[upd] = (1 - c.alpha) * self.ewma[upd] + c.alpha * lost[upd]
        self.ewma[~adj] = np.nan
        # kinematic gate: a real distance cannot change by more than g per period
        g = 2 * c.speed * c.T + 4 * np.hypot(c.sigma, c.sigma_ind)
        newg = adj & np.isnan(self.gate)
        self.gate[newg] = meas[newg]
        ug = adj & ~newg
        self.gate[ug] = np.clip(meas[ug], self.gate[ug] - g, self.gate[ug] + g)
        self.gate[~adj] = np.nan
        rx = adj & ~lost            # RU of column node received by row node this round
        return rx

    def fused(self, dist):
        c = self.cfg
        return c.beta * plr(dist) + (1 - c.beta) * self.ewma


def honest_claims(world, meas):
    """Honest RU content for both views: (CD_raw, CP_raw, CD_gated, CP_gated)."""
    Pr = world.fused(meas)
    Pg = world.fused(world.gate)
    return meas.copy(), Pr, world.gate.copy(), Pg
