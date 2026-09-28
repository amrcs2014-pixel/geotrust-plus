"""Geometric verification of neighbour claims (GeoTrust layer 2), batched on the GPU.

Verifier i checks every one-hop neighbour v with four statistics:
  R  reciprocity  : |d~_vu - d~_uv| for u in Gamma1(i) u {i}      (both ends of DS-TWR measure the link)
  CM Cayley-Menger: for u in Gamma2(i) (u's own report is NOT heard by i) take 5-point tuples
                    {v,u,a,b,c}, a,b,c in Gamma1(i) that report both v and u; only d_vu comes from v.
                    z = D / sigma_D with first-order noise propagation through the bordered
                    CM determinant (cofactors); claim score = median |z| over tuples.
                    Near-coplanar neighbourhoods switch to the 4-point (R^2) determinant.
  P  PLR residual : |PLR~_vu - P(d~_vu)|
  K  kinematics   : |d~_vu(t) - d~_vu(t-T)|
Node score rho_v = max_k stat_k / tau_k with tau_k calibrated on attack-free runs
(quantile 1 - 0.01/4 so the union false-exclusion is ~1 %).  v is excluded if rho_v > 1.
"""
from itertools import combinations
import numpy as np
import torch
from .sim import plr

TESTS = ("R", "CM", "P", "K")
MAX_ANCHORS = 6
MIN_TUPLES = 3
CHUNK = 40000
MIN_TARGETS = 2


SD_MAX = 0.2      # keep only tuples whose consistent-distance std error is below 20 cm


def _cm_z(d, sig, dev):
    """Exact Cayley-Menger residual for the suspect entry d[:,0,1] (= d_vu).
    det of the bordered CM matrix is exactly quadratic in x = d_vu^2, so the two distances
    consistent with the other entries (true + mirror) are its roots.  Returns
      e  : signed residual [m] = claimed d_vu - nearest consistent distance (e < 0: optimistic lie)
      sd : first-order std error [m] of that consistent distance (tuple conditioning)."""
    if d.shape[0] == 0:
        return np.zeros(0), np.zeros(0)
    t = torch.as_tensor(d, dtype=torch.float64, device=dev)
    B, k, _ = t.shape
    Bm = torch.ones(B, k + 1, k + 1, dtype=torch.float64, device=dev)
    Bm[:, 0, 0] = 0
    Bm[:, 1:, 1:] = t * t
    x0 = Bm[:, 1, 2].clone()
    vals = []
    for x in (torch.zeros_like(x0), x0, 2 * x0):
        M = Bm.clone()
        M[:, 1, 2] = M[:, 2, 1] = x
        vals.append(torch.linalg.det(M))
    f0, f1, f2 = vals
    a = (f2 - 2 * f1 + f0) / (2 * x0 ** 2)
    b = (f1 - f0) / x0 - a * x0
    disc = b * b - 4 * a * f0
    sq = torch.sqrt(torch.clamp(disc, min=0))
    roots = torch.stack([(-b - sq) / (2 * a), (-b + sq) / (2 * a)], 1)
    dr = torch.sqrt(torch.clamp(roots, min=0))
    e = t[:, 0, 1][:, None] - dr
    j = e.abs().argmin(1, keepdim=True)
    e = e.gather(1, j).squeeze(1)
    xr = roots.gather(1, j).squeeze(1)
    dsel = dr.gather(1, j).squeeze(1)
    M = Bm.clone()
    M[:, 1, 2] = M[:, 2, 1] = xr
    idx = torch.arange(k + 1, device=dev)
    var = torch.zeros(B, dtype=torch.float64, device=dev)
    for jj, ll in combinations(range(k), 2):
        if (jj, ll) == (0, 1):
            continue
        r, c = jj + 1, ll + 1
        cof = ((-1) ** (r + c)) * torch.linalg.det(M[:, idx != r][:, :, idx != c])
        var += (2 * cof * 2 * t[:, jj, ll] * sig) ** 2     # d det/d(d^2) * std(d^2)
    slope = torch.abs(2 * a * xr + b)
    sd = torch.sqrt(var) / slope / (2 * torch.clamp(dsel, min=0.05))
    sd = torch.where(disc > 0, sd, torch.full_like(sd, np.inf))   # complex roots: no consistent distance
    sd = torch.nan_to_num(sd, nan=np.inf)
    e = torch.nan_to_num(e, nan=0.0)
    return e.cpu().numpy(), sd.cpu().numpy()


def _pair(CD, a, b):
    x, y = CD[a, b], CD[b, a]
    if np.isnan(x):
        return y
    if np.isnan(y):
        return x
    return 0.5 * (x + y)


def planar(CD, i, G1, rng):
    """Detect a near-coplanar neighbourhood via normalised tetrahedron volumes."""
    if len(G1) < 3:
        return False
    vols = []
    for _ in range(20):
        a, b, c = rng.choice(G1, 3, replace=False)
        P = [i, a, b, c]
        d = np.array([[0 if p == q else _pair(CD, p, q) for q in P] for p in P])
        if np.isnan(d).any():
            continue
        Bm = np.ones((5, 5))
        Bm[0, 0] = 0
        Bm[1:, 1:] = d ** 2
        v2 = np.linalg.det(Bm) / 288.0
        e = d[np.triu_indices(4, 1)].mean()
        vols.append(abs(v2) / e ** 6)
    # random 3-D tetrahedra: median ~1.3e-3; 0.3 m-thick formation: ~3e-5
    return len(vols) > 5 and np.median(vols) < 2e-4


class Verifier:
    def __init__(self, cfg, tau=None, device=None):
        self.cfg = cfg
        dev = device or cfg.device
        self.dev = torch.device(dev if (dev.startswith("cpu") or torch.cuda.is_available()) else "cpu")
        if tau is not None and getattr(cfg, "tau_scale", 1.0) != 1.0:
            tau = {k: (v * cfg.tau_scale if not k.startswith("_") else v) for k, v in tau.items()}
            # like calibration, never set the kinematic threshold below the physical motion bound
            g = 2 * cfg.speed * cfg.T + 4 * np.hypot(cfg.sigma, cfg.sigma_ind)
            if "K" in tau:
                tau["K"] = max(tau["K"], 1.05 * g)
        self.tau = tau
        c = cfg
        self.sig = float(np.sqrt(c.sigma ** 2 + c.sigma_ind ** 2 + (c.speed * c.T) ** 2 / 12))

    def stats(self, own, CD, CP, CDprev, verifiers, rng):
        """own: verifier's own (gated) distances (n,n).  CD/CP: claims as seen by the verifiers.
        Returns dict {(i,v): {test: value}} for every verifier i and neighbour v."""
        n = CD.shape[0]
        has = ~np.isnan(CD)
        out, edges = {}, {}
        tuples, keys = [], {}                      # CM tuples shared across verifiers
        per_claim = {}                             # (i,v,u) -> list of tuple ids
        for i in verifiers:
            G1 = np.flatnonzero(~np.isnan(own[i]))
            inG1 = np.zeros(n, bool)
            inG1[G1] = True
            flat = planar(CD, i, G1, rng)
            k = 4 if flat else 5
            for v in G1:
                st = {}
                cl = np.flatnonzero(has[v])
                # R: reciprocity edges (blame is assigned later on the disagreement graph)
                # a one-sided link claim (only one end lists the other) is a disagreement too
                el = edges.setdefault(i, [])
                el.append((v, i, abs(CD[v, i] - own[i, v]) if has[v, i] else np.inf, None))
                for u in G1[G1 > v]:
                    if has[v, u] and has[u, v]:
                        el.append((v, u, abs(CD[v, u] - CD[u, v]), None))
                    elif has[v, u]:
                        el.append((v, u, np.inf, (v, CD[v, u])))     # (claimant, claimed distance)
                    elif has[u, v]:
                        el.append((v, u, np.inf, (u, CD[u, v])))
                # P: one-sided PLR residual (claimed link better than its distance implies);
                #    used to calibrate the PLR clamp, not for exclusion
                st["P"] = np.nanmax(plr(CD[v, cl]) - CP[v, cl]) if cl.size else np.nan
                # K: one-sided kinematics - claimed distance shrinking faster than physically possible
                both = cl[~np.isnan(CDprev[v, cl])] if CDprev is not None else []
                st["K"] = np.max(CDprev[v, both] - CD[v, both]) if len(both) else np.nan
                out[(i, v)] = st
                # CM tuples for two-hop claims
                for u in cl[~inG1[cl] & (cl != i)]:
                    anc = G1[(G1 != v) & has[G1, u] & has[G1, v]]
                    if anc.size < k - 2:
                        continue
                    if anc.size > MAX_ANCHORS:
                        anc = rng.choice(anc, MAX_ANCHORS, replace=False)
                    lst = []
                    for tri in combinations(sorted(anc), k - 2):
                        if any(np.isnan(_pair(CD, a, b)) for a, b in combinations(tri, 2)):
                            continue
                        key = (v, u) + tri
                        if key not in keys:
                            keys[key] = len(tuples)
                            tuples.append(key)
                        lst.append(keys[key])
                    if lst:
                        per_claim[(i, v, u)] = lst
        # ---- GPU batch: build distance tensors (only d_vu comes from v's own claim)
        z, sd = np.zeros(0), np.zeros(0)
        if tuples:
            by_k = {}
            for tid, key in enumerate(tuples):
                by_k.setdefault(len(key), []).append(tid)
            z = np.zeros(len(tuples))
            sd = np.full(len(tuples), np.inf)
            for k, tids in by_k.items():
                d = np.zeros((len(tids), k, k))
                for r, tid in enumerate(tids):
                    P = tuples[tid]
                    v, u = P[0], P[1]
                    for x in range(k):
                        for y in range(x + 1, k):
                            p, q = P[x], P[y]
                            if x == 0 and y == 1:
                                val = CD[v, u]                  # the claim under test
                            elif x == 0:
                                val = CD[q, v]                  # anchor's own report of v
                            elif x == 1:
                                val = CD[q, u]                  # anchor's own report of u
                            else:
                                val = _pair(CD, p, q)
                            d[r, x, y] = d[r, y, x] = val
                for c0 in range(0, len(tids), CHUNK):          # bounded host/GPU memory
                    sl = tids[c0:c0 + CHUNK]
                    z[sl], sd[sl] = _cm_z(d[c0:c0 + CHUNK], self.sig, self.dev)
        cm, self.claims = {}, {}
        for (i, v, u), lst in per_claim.items():
            lst = [t for t in lst if sd[t] < SD_MAX]          # well-conditioned tuples only
            if len(lst) < MIN_TUPLES:
                continue
            s = -float(np.median(z[lst]))           # > 0 [m]: claim shorter than geometry supports
            cm[(i, v)] = max(cm.get((i, v), -np.inf), s)
            # keep tuple members + scores so exclude() can drop tuples with blamed anchors
            self.claims.setdefault((i, v), []).append(
                (u, np.array([tuples[t][2:] for t in lst]), -z[lst]))
        for key, st in out.items():
            st["CM"] = cm.get(key, np.nan)
        return out, edges, self.claims

    def exclude(self, i, G1, out, edges, claims=None, use_cm=True, strong=None):
        """Boolean mask over G1: True = excluded.  Also returns the per-test flags.
        Reciprocity blame: a disagreement with the verifier's own measurement blames v;
        other disagreements form a graph and nodes are blamed greedily (vertex cover) only while
        they disagree with >= 2 partners, so an honest node next to one liar is not condemned."""
        tau = self.tau
        flags = {k: np.zeros(len(G1), bool) for k in TESTS}
        pos = {v: j for j, v in enumerate(G1)}
        for j, v in enumerate(G1):
            x = out.get((i, v), {}).get("K", np.nan)
            flags["K"][j] = (not np.isnan(x)) and x > tau["K"]
        E = [e for e in edges.get(i, []) if e[2] > tau["R"]]
        blamed = {v for v, u, r, _ in E if u == i}
        rest = [(v, u) for v, u, r, _ in E if u != i and v not in blamed and u not in blamed]
        while rest:
            deg = {}
            for v, u in rest:
                deg[v] = deg.get(v, 0) + 1
                deg[u] = deg.get(u, 0) + 1
            x = max(deg, key=deg.get)
            if deg[x] < 2:
                break
            blamed.add(x)
            rest = [(v, u) for v, u in rest if x not in (v, u)]
        # a one-sided claim well inside the range (honest links there never lose reciprocity)
        # blames the claimant, unless the disagreement was already explained above
        for v, u, r, one in E:
            if one is not None and u != i and v not in blamed and u not in blamed                     and one[1] < (self.cfg.strong_frac if strong is None else strong) * self.cfg.rc + 1e-9:
                blamed.add(one[0])
        for v in blamed:
            if v in pos:
                flags["R"][pos[v]] = True
        # CM with consensus attribution.  Tuples whose anchors are already blamed are discarded and
        # a claim is judged only with >= MIN_TUPLES clean tuples.  A flagged claim (v,u) is always
        # dropped; v itself is blamed only if another relay's claim about the same u passes -
        # otherwise the anomaly may sit on u's links (e.g. PHY enlargement of u) and v is spared.
        claims = self.claims if claims is None else claims
        if not use_cm:
            claims = {}
        bad_nodes = set(G1[flags["R"] | flags["K"]].tolist())
        drops, passed = set(), {}
        for _ in range(2):
            flagged, passed, pess = {}, {}, set()
            for v in G1:
                if v in bad_nodes:
                    continue
                for u, anc, s in claims.get((i, v), []):
                    keep = ~np.isin(anc, list(bad_nodes)).any(1) if bad_nodes else np.ones(len(s), bool)
                    if keep.sum() < MIN_TUPLES:
                        continue
                    med = np.median(s[keep])
                    if med > tau["CM"]:
                        flagged.setdefault(u, set()).add(int(v))
                    else:
                        passed.setdefault(u, set()).add(int(v))
                        if med < -tau["CM"]:
                            pess.add(u)          # someone's claim about u is too LONG
            drops = {(v, u) for u, vs in flagged.items() for v in vs}
            # two-sided disagreement about u -> u's links are corrupted (PHY): drop, don't blame
            cnt = {}
            for u, vs in flagged.items():
                if passed.get(u) and u not in pess:
                    for v in vs:
                        cnt[v] = cnt.get(v, 0) + 1
            # a liar contradicts the geometry on several targets; a PHY victim's corrupted links
            # implicate an honest relay through few targets only
            new = {v for v, c in cnt.items() if c >= MIN_TARGETS}
            if new <= bad_nodes:
                break
            bad_nodes |= new
        for v in bad_nodes:
            if v in pos and not (flags["R"][pos[v]] or flags["K"][pos[v]]):
                flags["CM"][pos[v]] = True
        mask = flags["R"] | flags["CM"] | flags["K"]
        self.verified_targets = set(passed)
        return mask, flags, drops

    def clamp(self, CD, CP):
        """PLR clamp: a claimed PLR may not be better than its claimed distance implies."""
        return np.fmax(CP, plr(CD) - self.tau["P"])
