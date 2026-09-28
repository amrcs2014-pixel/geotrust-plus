"""One simulation run -> list of metric rows (one per evaluation round x method)."""
import time
from dataclasses import dataclass
import numpy as np
from .config import Config
from .sim import World, honest_claims
from .attacks import Adversary
from .verify import Verifier, TESTS
from . import mpr


@dataclass(frozen=True)
class Spec:
    view: str = "raw"            # raw | gated | lag (lag = TESLA-authenticated previous-round RUs)
    verify: object = False       # False | True | "topo"
    alg: str = "greedy"          # flood | olsr | grr | grr2 | ilp | greedy
    f: object = 0                # int | "cfg" | "adaptive"
    cm: bool = True              # Cayley-Menger layer
    wd: bool = False             # data-plane watchdog
    strong: object = None        # override strong_frac (None = cfg)
    corroborate: bool = False


SPEC = {
    "FLOOD": Spec(alg="flood"),
    "OLSR": Spec(alg="olsr"),
    "GRR": Spec(alg="grr_zhou"),                                # Zhou et al. Alg. 1
    "GRR2": Spec(alg="grr"),                                    # my earlier reconstruction (sensitivity)
    "ILP": Spec(alg="ilp"),
    "SG": Spec(alg="greedy"),
    "TOPO-GRR": Spec(verify="topo", alg="grr_zhou"),
    "GRR-W": Spec(alg="grr_zhou", wd=True),                          # watchdog alone on the base scheme
    "GT-R": Spec(f="cfg"),                                      # robust cover only
    "GT-V": Spec(view="lag", verify=True),                      # verification only (f=0)
    "GT-noCM": Spec(view="lag", verify=True, f="cfg", cm=False),
    "GeoTrust": Spec(view="lag", verify=True, f="cfg"),         # fixed-f design of the plan
    "GeoTrust-noAuth": Spec(view="gated", verify=True, f="cfg"),
    "GeoTrust-RILP": Spec(view="lag", verify=True, alg="ilp", f="cfg"),
    "GeoTrust-C": Spec(view="lag", verify=True, f="cfg", corroborate=True),
    # ---- revised design. Headline GeoTrust+ = GT+f0 (verification + watchdog + strong rule, f=0, no CM);
    #      the robust cover is the opt-in "guaranteed mode" (GT+fixF), CM an optional extension (GT+)
    "GT+f0": Spec(view="lag", verify=True, f=0, wd=True, strong=1.0, cm=False),
    "GT+noCM": Spec(view="lag", verify=True, f="adaptive", wd=True, strong=1.0, cm=False),
    "GT+fixF": Spec(view="lag", verify=True, f="cfg", wd=True, strong=1.0, cm=False),
    "GT+": Spec(view="lag", verify=True, f=0, wd=True, strong=1.0, cm=True),
    "GT+noWD": Spec(view="lag", verify=True, f=0, wd=False, strong=1.0, cm=False),
    "GT+weak": Spec(view="lag", verify=True, f=0, wd=True, cm=False),  # strong_frac from cfg (0.7)
    # ---- GeoTrust+ on classic OLSR (RFC 3626 MPR heuristic)
    "OLSR-W": Spec(alg="olsr", wd=True),                                           # watchdog only
    "TOPO-OLSR": Spec(verify="topo", alg="olsr"),                                  # topological trust
    "OLSR-G": Spec(view="lag", verify=True, alg="olsr", wd=True, strong=1.0, cm=False),  # GeoTrust+ on RFC
    # ---- ablations of GeoTrust+ on RFC 3626
    "OLSR-V": Spec(view="lag", verify=True, alg="olsr", wd=False, strong=1.0, cm=False),       # evidence only
    "OLSR-Gweak": Spec(view="lag", verify=True, alg="olsr", wd=True, cm=False),                # strong rule off (0.7 R)
    "OLSR-GnoAuth": Spec(view="gated", verify=True, alg="olsr", wd=True, strong=1.0, cm=False),  # no TESLA lag
}
METHODS = tuple(SPEC)
CORE = ("FLOOD", "OLSR", "GRR", "GRR2", "ILP", "SG", "TOPO-GRR", "GRR-W", "GT-R", "GT-V", "GT-noCM",
        "GeoTrust", "GeoTrust-noAuth", "GT+f0", "GT+noCM", "GT+fixF", "GT+", "GT+noWD", "GT+weak")


def _select(alg, C, t, f):
    if alg == "flood":
        return mpr.flooding(C, t)
    if alg == "olsr":
        return mpr.olsr(C, t)
    if alg == "grr":
        return mpr.grr(C, t)
    if alg == "grr_zhou":
        return mpr.grr_zhou(C, t)
    if alg == "ilp":
        return mpr.ilp(C, t, f=f)
    return mpr.robust_greedy(C, t, f=f)


def topo_reject(CD, adj_i, i, v):
    """Adnane-style topological consistency: every claimed link to a node the verifier can hear
    must be reciprocated, and v must list i."""
    if np.isnan(CD[v, i]):
        return True
    for u in np.flatnonzero(~np.isnan(CD[v])):
        if adj_i[u] and np.isnan(CD[u, v]):
            return True
    return False


def flood_trials(cfg, rng, PT, adj, is_att, mpr_sets, trials):
    """OLSR MPR flooding with common random numbers across methods.  Attackers drop w.p. q_drop."""
    n = cfg.n
    honest = np.flatnonzero(~is_att)
    succ = [rng.random((n, n)) >= PT for _ in range(trials)]
    fwd = [rng.random(n) >= cfg.q_drop for _ in range(trials)]
    srcs = rng.choice(honest, trials)
    res = {}
    for m, sets in mpr_sets.items():
        pdr, tx = [], []
        for s, ok, fw in zip(srcs, succ, fwd):
            got = np.zeros(n, bool)
            got[s] = True
            q, ntx = [s], 0
            while q:
                x = q.pop()
                ntx += 1
                for y in np.flatnonzero(adj[x] & ok[x] & ~got):
                    got[y] = True
                    if (not is_att[y] or fw[y]) and y in sets.get(x, ()):
                        q.append(y)
            pdr.append((got[honest].sum() - 1) / max(1, honest.size - 1))
            tx.append(ntx / n)
        res[m] = (float(np.mean(pdr)), float(np.mean(tx)))
    return res


def e2e_trials(cfg, rng, PT, adj, is_att, mpr_sets, att_sets, flows=20, tries=3):
    """End-to-end unicast over OLSR-style routes: shortest paths on the MPR-link topology
    (links (x, m) for m in MPR(x), as advertised in TCs) plus the source's own one-hop links.
    Per hop up to `tries` MAC attempts; attackers drop w.p. q_drop.  Common random numbers."""
    n = cfg.n
    honest = np.flatnonzero(~is_att)
    pairs = [tuple(rng.choice(honest, 2, replace=False)) for _ in range(flows)]
    draws = rng.random((flows, n, tries))
    drops = rng.random((flows, n)) < cfg.q_drop
    res = {}
    for m, sets in mpr_sets.items():
        nbr = [set() for _ in range(n)]
        for x, S in list(sets.items()) + list(att_sets.items()):
            for y in S:
                nbr[x].add(y)
                nbr[y].add(x)
        ok, hops, att = [], [], []
        for k, (s, d) in enumerate(pairs):
            prev = {s: None}
            q = [s]
            first = True
            while q and d not in prev:
                nq = []
                for x in q:
                    cand = np.flatnonzero(adj[x]) if first else nbr[x]
                    for y in cand:
                        y = int(y)
                        if y not in prev and adj[x, y]:
                            prev[y] = x
                            nq.append(y)
                q, first = nq, False
            if d not in prev:
                ok.append(0.0)
                continue
            path = [d]
            while prev[path[-1]] is not None:
                path.append(prev[path[-1]])
            path = path[::-1]
            good = True
            for a, b in zip(path[:-1], path[1:]):
                if is_att[a] and a != s and drops[k, a]:
                    good = False
                    break
                if not (draws[k, a] >= PT[a, b]).any():
                    good = False
                    break
            ok.append(float(good))
            hops.append(len(path) - 1)
            att.append(float(is_att[path[1:-1]].any()) if len(path) > 2 else 0.0)
        res[m] = (float(np.mean(ok)), float(np.mean(hops)) if hops else np.nan,
                  float(np.mean(att)) if att else np.nan)
    return res


def e2e_quarantine(cfg, rng, PT, adj, is_att, mpr_sets, att_sets, quar, flows=20, tries=3):
    """Hop-by-hop OLSR-style forwarding in which every honest node computes its next hop on the
    MPR-link topology with the nodes IT has quarantined removed (verification exclusions and
    watchdog verdicts).  Attackers forward normally but drop w.p. q_drop.  Common random numbers."""
    n = cfg.n
    honest = np.flatnonzero(~is_att)
    pairs = [tuple(rng.choice(honest, 2, replace=False)) for _ in range(flows)]
    draws = rng.random((flows, n, tries))
    drops = rng.random((flows, n)) < cfg.q_drop
    res, cap = {}, {}
    for m, sets in mpr_sets.items():
        nbr = [set() for _ in range(n)]
        for x, S in list(sets.items()) + list(att_sets.items()):
            for y in S:
                nbr[x].add(y)
                nbr[y].add(x)
        Q = quar.get(m, {})

        def next_hop(x, d):
            bad = Q.get(x, set())
            prev = {x: None}
            q = [x]
            first = True
            while q and d not in prev:
                nq = []
                for a in q:
                    cand = np.flatnonzero(adj[a]) if first else nbr[a]
                    for y in cand:
                        y = int(y)
                        if y not in prev and adj[a, y] and (y not in bad or y == d):
                            prev[y] = a
                            nq.append(y)
                q, first = nq, False
            if d not in prev:
                return None
            y = d
            while prev[y] != x:
                y = prev[y]
            return y

        ok, via = [], []
        for k, (s, d) in enumerate(pairs):
            x, good, seen, hit = s, True, 0, False
            while x != d:
                if is_att[x] and x != s:
                    hit = True
                if is_att[x] and x != s and drops[k, x]:
                    good = False
                    break
                y = next_hop(x, d)
                seen += 1
                if y is None or seen > 3 * n or not (draws[k, x] >= PT[x, y]).any():
                    good = False
                    break
                x = y
            ok.append(float(good))
            via.append(float(hit))
        res[m] = float(np.mean(ok))
        cap[m] = float(np.mean(via))
    return res, cap


class RunTimeout(Exception):
    pass


def run(cfg: Config, tau=None, collect_null=False, methods=CORE, budget_s=None):
    t_start = time.time()
    rng = np.random.default_rng(cfg.seed)
    world = World(cfg, rng)
    adv = Adversary(cfg, np.random.default_rng(cfg.seed + 10_000), world)
    ver = Verifier(cfg, tau)
    vrng = np.random.default_rng(cfg.seed + 20_000)
    frng = np.random.default_rng(cfg.seed + 30_000)
    wrng = np.random.default_rng(cfg.seed + 40_000)
    n, is_att = cfg.n, adv.is_att
    honest_ids = np.flatnonzero(~is_att)
    hist = []
    rows, null = [], {}
    wd_state = {m: {} for m in methods if SPEC[m].wd}      # (i,v) -> [k_obs, mu, var]
    wd_flag = {m: set() for m in methods if SPEC[m].wd}
    prev_sets = {}
    for r in range(cfg.rounds):
        if r:
            world.step()
        D = world.true_dist()
        PT = world.true_plr(D)
        meas, adj = world.measure(D, adv.phy_hook(D))
        world.update_links(D, adj, meas)
        CDr, CPr, CDg, CPg = honest_claims(world, meas)
        adv.forge(world, D, adj, CDr, CPr)
        CDg[is_att], CPg[is_att] = CDr[is_att], CPr[is_att]
        hist.append((CDg, CPg, world.gate.copy()))
        hist = hist[-3:]
        if budget_s and time.time() - t_start > budget_s:
            raise RunTimeout(f"round {r} after {time.time() - t_start:.0f}s")
        if r < cfg.warmup or (r - cfg.warmup) % cfg.eval_every:
            continue
        views = {"raw": (CDr, CPr), "gated": (CDg, CPg), "lag": hist[-2][:2]}
        own = {"raw": meas, "gated": world.gate, "lag": hist[-2][2]}
        ownP = {"raw": world.fused(meas), "gated": world.fused(world.gate), "lag": world.fused(world.gate)}
        vstats, vtime = {}, {}
        need = {SPEC[m].view for m in methods if SPEC[m].verify is True}
        if collect_null:
            need = {"gated", "lag"}
        for vw in need:
            CD, CP = views[vw]
            prev = hist[-2][0] if vw == "gated" else (hist[-3][0] if len(hist) >= 3 else None)
            t0 = time.perf_counter()
            vstats[vw] = ver.stats(own[vw], CD, CP, prev, honest_ids, vrng)
            vtime[vw] = time.perf_counter() - t0
        if collect_null:
            for vw in need:
                out, edges, _ = vstats[vw]
                for k in ("CM", "P", "K"):
                    null.setdefault(k, []).append(np.array([st[k] for st in out.values()], np.float32))
                null.setdefault("R", []).append(np.array([e[2] for el in edges.values() for e in el], np.float32))
            continue
        # ---------------- watchdog: overhearing during the window since the last evaluation
        for m in wd_state:
            for i, S in prev_sets.get(m, {}).items():
                for v in S:
                    if not adj[i, v]:
                        continue
                    p_true = (1 - PT[i, v]) * (1 - PT[v, i]) * (1 - cfg.q_drop * is_att[v])
                    k = wrng.binomial(cfg.wd_msgs, p_true)
                    e = world.ewma[i, v]
                    p_exp = (1 - (e if not np.isnan(e) else 0.0)) ** 2
                    st = wd_state[m].setdefault((i, v), [0.0, 0.0, 0.0])
                    st[0] += k
                    st[1] += cfg.wd_msgs * p_exp
                    st[2] += cfg.wd_msgs * p_exp * (1 - p_exp)
                    if st[1] >= 10 and (st[0] - st[1]) / np.sqrt(st[2] + 1.0) < -cfg.wd_z:
                        wd_flag[m].add((i, v))
        # ---------------- relays chosen by attackers (for routing; honest-looking GRR)
        att_sets = {}
        for a in np.flatnonzero(is_att):
            G1 = np.flatnonzero(adj[a])
            if G1.size == 0:                             # isolated attacker: no relays
                att_sets[int(a)] = set()
                continue
            cl = ~np.isnan(CDr[G1])
            U = np.flatnonzero(cl.any(0) & ~adj[a])
            U = U[U != a]
            C, t = mpr.contributions(world.fused(meas)[a, G1], np.where(cl[:, U], CPr[G1][:, U], np.nan),
                                     cfg.gamma, cfg.qmin)
            S = mpr.grr_zhou(C, t)
            att_sets[int(a)] = set(G1[S].tolist()) if len(S) else set()
        # ---------------- selection per honest verifier and method
        sets = {m: {} for m in methods}
        quar = {}
        acc = {m: {} for m in methods}

        def put(m, k, x):
            acc[m].setdefault(k, []).append(x)

        ilp_size = {}
        for i in honest_ids:
            G1 = np.flatnonzero(adj[i])
            if G1.size == 0:
                continue
            att_nb = is_att[G1].any()
            hv = G1[~is_att[G1]]
            U_true = np.flatnonzero(adj[hv].any(0) & ~adj[i] & ~is_att)
            U_true = U_true[U_true != i]
            Pt_iv = PT[i]
            for m in methods:
                sp = SPEC[m]
                CD, CP = views[sp.view]
                mask = np.ones(G1.size, bool)
                qx = np.zeros(G1.size, bool)                        # quarantine evidence (routing)
                drops, evidence = set(), False
                if sp.verify == "topo":
                    mask = np.array([not topo_reject(CD, adj[i], i, v) for v in G1])
                    qx = ~mask
                elif sp.verify:
                    CP = ver.clamp(CD, CP)
                    exc, fl, drops = ver.exclude(i, G1, *vstats[sp.view], use_cm=sp.cm, strong=sp.strong)
                    qx = exc.copy()
                    evidence = bool(exc.any() or drops)
                    if sp.view == "lag":
                        pend = np.isnan(own["lag"][i, G1])          # TESLA: no authenticated RU yet
                        if (~is_att[G1]).any():
                            put(m, "fex_auth", float(pend[~is_att[G1]].mean()))
                        exc = exc | pend
                    mask = ~exc
                    hmask = ~is_att[G1]
                    if hmask.any():
                        put(m, "fex", float(exc[hmask].mean()))
                        for k in TESTS:
                            put(m, "fex_" + k, float(fl[k][hmask].mean()))
                    if (~hmask).any():
                        put(m, "tpr", float(exc[~hmask].mean()))
                        for k in TESTS:
                            put(m, "tpr_" + k, float(fl[k][~hmask].mean()))
                if sp.wd:
                    wdx = np.array([(i, v) in wd_flag[m] for v in G1])
                    if (~is_att[G1]).any():
                        put(m, "fex_wd", float(wdx[~is_att[G1]].mean()))
                    if is_att[G1].any():
                        put(m, "tpr_wd", float(wdx[is_att[G1]].mean()))
                    evidence = evidence or bool(wdx.any())
                    mask = mask & ~wdx
                    qx = qx | wdx
                quar.setdefault(m, {})[int(i)] = set(G1[qx].tolist())
                cl = ~np.isnan(CD[G1])
                U = np.flatnonzero(cl.any(0) & ~adj[i])
                U = U[U != i]
                Pvu = np.where(cl[:, U], CP[G1][:, U], np.nan)
                Pvu[~mask] = np.nan
                if sp.corroborate:
                    sup = (~np.isnan(Pvu)).sum(0)
                    ok = np.array([(u in ver.verified_targets) or sup[c] >= 2 for c, u in enumerate(U)], bool)
                    Pvu[:, ~ok] = np.nan
                if drops:
                    col = {u: c for c, u in enumerate(U)}
                    row_ = {v: j for j, v in enumerate(G1)}
                    for v, u in drops:
                        if v in row_ and u in col:
                            Pvu[row_[v], col[u]] = np.nan
                if sp.f == "cfg":
                    fm = cfg.f
                elif sp.f == "adaptive":
                    fm = cfg.f if evidence else 0
                    put(m, "f_on", float(fm > 0))
                else:
                    fm = sp.f
                C, t = mpr.contributions(ownP[sp.view][i, G1], Pvu, cfg.gamma, cfg.qmin)
                t0 = time.perf_counter()
                S = _select(sp.alg, C, t, fm)
                put(m, "rt", time.perf_counter() - t0)
                if fm > 0 and C.size and sp.alg == "greedy":
                    _, T = mpr.robust_targets(C, t, fm)
                    need_u = t > 1e-9
                    if need_u.any():
                        put(m, "cap", float((T[:, need_u] < t[need_u] - 1e-9).any(0).mean()))
                Sid = G1[S] if len(S) else np.array([], int)
                sets[m][int(i)] = set(Sid.tolist())
                put(m, "size", len(Sid))
                if m in ("ILP", "GeoTrust-RILP"):
                    ilp_size[(m, i)] = len(Sid)
                if att_nb:
                    put(m, "incl", float(is_att[Sid].any()))
                # true two-hop reliability; attackers deliver w.p. 1 - q_drop
                keep = 1 - cfg.q_drop * is_att
                for u in U_true:
                    rel = hv[adj[hv, u]]
                    pf = Pt_iv[rel] + (1 - Pt_iv[rel]) * PT[rel, u]
                    qstar = np.prod(pf) if cfg.qmin == "all" else pf.min()
                    rs = Sid[adj[Sid, u]] if Sid.size else Sid
                    put(m, "cov", float((~is_att[rs]).any()) if rs.size else 0.0)
                    fail = np.prod(1 - keep[rs] * (1 - Pt_iv[rs]) * (1 - PT[rs, u])) if rs.size else 1.0
                    put(m, "qos", float(fail <= min(cfg.gamma * qstar, 1.0) * (1 + 1e-9)))
        for i in honest_ids:
            for m, ref in (("GRR", "ILP"), ("GRR2", "ILP"), ("SG", "ILP"), ("OLSR", "ILP"),
                           ("GeoTrust", "GeoTrust-RILP")):
                if m in sets and (ref, i) in ilp_size and ilp_size[(ref, i)] > 0:
                    put(m, "ratio", len(sets[m][int(i)]) / ilp_size[(ref, i)])
        prev_sets = {m: sets[m] for m in wd_state}
        fl = flood_trials(cfg, frng, PT, adj, is_att, sets, cfg.floods)
        ee = e2e_trials(cfg, frng, PT, adj, is_att, sets, att_sets)
        eq, eqc = e2e_quarantine(cfg, frng, PT, adj, is_att, sets, att_sets, quar)
        btw_att = np.nan                                   # insiders' share of betweenness in the true graph
        if is_att.any():
            import networkx as nx
            bc = nx.betweenness_centrality(nx.from_numpy_array(adj.astype(int)))
            tot = sum(bc.values())
            btw_att = float(sum(bc[int(a)] for a in np.flatnonzero(is_att)) / tot) if tot > 0 else np.nan
        for m in methods:
            a = acc[m]
            mean = lambda k: float(np.mean(a[k])) if a.get(k) else np.nan
            row = dict(cfg.asdict(), round=r, method=m, pdr=fl[m][0], tx=fl[m][1],
                       e2e=ee[m][0], hops=ee[m][1], e2e_att=ee[m][2], e2e_q=eq[m], route_att_q=eqc[m], btw_att=btw_att,
                       rt_ms=1e3 * mean("rt"), ratio_max=float(np.max(a["ratio"])) if a.get("ratio") else np.nan,
                       verify_s=vtime.get(SPEC[m].view, np.nan) if SPEC[m].verify is True else np.nan,
                       n_att=int(is_att.sum()))
            for k in ("incl", "size", "cov", "qos", "fex", "tpr", "ratio", "cap", "fex_auth", "f_on",
                      "fex_wd", "tpr_wd") + tuple(p + t for t in TESTS for p in ("fex_", "tpr_")):
                row[k] = mean(k)
            rows.append(row)
    return ({k: np.concatenate(v) for k, v in null.items()} if collect_null else rows)


WD_Z = 3.5
