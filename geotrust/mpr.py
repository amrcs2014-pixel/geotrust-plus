"""Relay (MPR) selection algorithms.

Every algorithm is a pure function of the contribution matrix C (candidates x two-hop nodes)
and targets t, returning indices into the candidate list.

Contribution of relay v to two-hop node u (Zhou et al.):
    c_vu = -log( P_iv + (1 - P_iv) P_vu ),   t_u = -log min(gamma * Q_min(u), 1)
with Q_min(u) the best single-relay two-hop PLR, so one best relay always satisfies u.
"""
from itertools import combinations
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

EPS = 1e-9


def contributions(P_iv, P_vu, gamma, qmin="single"):
    """P_iv: (n,), P_vu: (n,k) with NaN where v does not claim u. Returns C (n,k), t (k,).
    qmin="all":    Q_min(u) = prod over ALL relays of the two-hop failure prob, t_u = sum_v c_vu - ln gamma
                   (Zhou et al. Alg. 1 as implemented in the RangeGuard ns-3 patch);
    qmin="single": Q_min(u) = best single-relay failure prob, t_u = max_v c_vu - ln gamma."""
    P_iv = np.clip(P_iv, 1e-4, 1.0)
    P_vu = np.clip(P_vu, 1e-4, 1.0)
    if P_vu.size == 0:                                   # isolated node or no two-hop targets
        return np.zeros(P_vu.shape), np.zeros(P_vu.shape[1])
    fail = P_iv[:, None] + (1 - P_iv[:, None]) * P_vu
    C = np.where(np.isnan(fail), 0.0, -np.log(np.clip(fail, 1e-12, 1.0)))
    base = C.sum(0) if qmin == "all" else C.max(0)
    t = np.maximum(0.0, base - np.log(gamma))
    t[C.max(0) <= 0] = 0.0
    return C, t


def grr_zhou(C, t, **_):
    """Zhou et al. (2026) Algorithm 1, ported 1:1 from the RangeGuard ns-3 patch (GrrMprSet):
    every target ranks relays by contribution; in round r each target in turn adds its r-th ranked relay
    (if not yet selected and useful); stop as soon as every target is satisfied."""
    n, k = C.shape
    if n == 0 or k == 0:
        return []
    R = np.argsort(-C, axis=0, kind="stable")          # R[r, i] = r-th best relay of target i
    chosen = np.zeros(n, bool)
    acc = np.zeros(k)
    S = []
    if np.all(acc + 1e-9 >= t):
        return S
    for r in range(n):
        for i in range(k):
            j = R[r, i]
            if chosen[j] or C[j, i] <= 0:
                continue
            chosen[j] = True
            S.append(int(j))
            acc += C[j]
            if np.all(acc + 1e-9 >= t):
                return S
    return S


# --------------------------------------------------------------------- baselines
def flooding(C, t, **_):
    return list(range(C.shape[0]))


def grr(C, t, hardest_first=False, **_):
    """Greedy Round-Robin (reconstruction of Zhou et al. Alg.): visit two-hop nodes in turn;
    an unsatisfied node adds its best not-yet-selected relay; residuals of all nodes drop.
    hardest_first: visit targets in increasing number of candidate relays (variant GRR2)."""
    n, k = C.shape
    r = t.copy()
    S, chosen = [], np.zeros(n, bool)
    order = np.argsort((C > 0).sum(0), kind="stable") if hardest_first else range(k)
    while (r > EPS).any():
        progress = False
        for u in order:
            if r[u] <= EPS:
                continue
            cand = np.where(chosen, -1.0, C[:, u])
            v = int(cand.argmax())
            if cand[v] <= 0:
                r[u] = 0.0                      # cannot be improved further
                continue
            S.append(v)
            chosen[v] = True
            r -= C[v]
            progress = True
        if not progress:
            break
    return S


def olsr(C, t, **_):
    """RFC 3626 §8.3.1 MPR heuristic (link quality ignored: coverage = claimed adjacency)."""
    A = C > 0
    need = t > 0
    n = A.shape[0]
    S = set()
    cov = np.zeros(A.shape[1], bool)
    for u in np.flatnonzero(need):
        rel = np.flatnonzero(A[:, u])
        if rel.size == 1:
            S.add(int(rel[0]))
    for v in S:
        cov |= A[v]
    deg = A.sum(1)
    while (need & ~cov).any():
        reach = (A & (need & ~cov)).sum(1)
        reach[list(S)] = -1
        if reach.max() <= 0:
            break
        best = np.flatnonzero(reach == reach.max())
        v = int(best[np.argmax(deg[best])])
        S.add(v)
        cov |= A[v]
    return sorted(S)


# --------------------------------------------------- submodular (robust) cover
def _fail_sets(n, f):
    Fs = [()]
    for j in range(1, min(f, n) + 1):
        Fs += list(combinations(range(n), j))
    M = np.ones((len(Fs), n))
    for r, F in enumerate(Fs):
        M[r, list(F)] = 0.0
    return M


def robust_targets(C, t, f):
    """Capped targets T[F,u] = min(t_u, sum_{v not in F} c_vu) for every |F| <= f."""
    M = _fail_sets(C.shape[0], f)
    return M, np.minimum(t[None, :], M @ C)


def robust_value(C, T, M, S):
    x = np.zeros(C.shape[0])
    x[list(S)] = 1.0
    return np.minimum(T, (M * x) @ C).sum()


def robust_greedy(C, t, f=1, w=None, prune=True, **_):
    """R-GRR: greedy on G_f(S) = sum_{|F|<=f} sum_u min(T_Fu, sum_{v in S\\F} c_vu).
    G_f is monotone submodular and G_f(S) = G_f(N) <=> S is f-robust feasible (with capped targets),
    so greedy enjoys Wolsey's logarithmic ratio.  f = 0 gives the plain submodular greedy (SG)."""
    n, k = C.shape
    if n == 0 or k == 0 or not (t > EPS).any():
        return []
    M, T = robust_targets(C, t, f)
    goal = T.sum()
    cur = np.zeros_like(T)
    S, chosen = [], np.zeros(n, bool)
    tie = (w if w is not None else np.ones(n)) * 1e-9
    while goal - cur.clip(max=T).sum() > 1e-7:
        # gains[x] = sum_F sum_u min(T, cur + M[F,x] C[x]) - min(T, cur)
        new = np.minimum(T[None], cur[None] + M.T[:, :, None] * C[:, None, :])
        gains = (new - np.minimum(T, cur)[None]).sum((1, 2))
        gains[chosen] = -1
        x = int(np.argmax(gains + tie))
        if gains[x] <= 1e-12:
            break
        S.append(x)
        chosen[x] = True
        cur = cur + M[:, x:x + 1] * C[x][None]
    if prune:
        S = _prune(C, T, M, S, goal)
    return S


def _prune(C, T, M, S, goal):
    """Reverse-delete: drop relays whose removal keeps every (capped) constraint satisfied."""
    S = list(S)
    for x in sorted(S, key=lambda v: C[v].sum()):
        rest = [v for v in S if v != x]
        if robust_value(C, T, M, rest) >= goal - 1e-7:
            S = rest
    return S


def ilp(C, t, f=0, **_):
    """Exact (robust) minimum relay cover with HiGHS; capped targets as in robust_greedy."""
    n, k = C.shape
    if n == 0 or not (t > EPS).any():
        return []
    M, T = robust_targets(C, t, f)
    rows, lb = [], []
    for r in range(M.shape[0]):
        for u in range(k):
            if T[r, u] > EPS:
                rows.append(M[r] * C[:, u])
                lb.append(T[r, u] - 1e-7)
    A = np.array(rows)
    res = milp(np.ones(n), constraints=LinearConstraint(A, lb, np.inf),
               integrality=np.ones(n), bounds=Bounds(0, 1),
               options={"time_limit": 10})
    if res.x is None:
        return robust_greedy(C, t, f=f)
    return [int(v) for v in np.flatnonzero(res.x > 0.5)]


def wolsey_bound(C, t, f, eps=1e-2):
    """Instance ratio bound 1 + ln(G_f(N) / eps) for eps-discretised inputs."""
    M, T = robust_targets(C, t, f)
    G = T.sum()
    return 1 + np.log(max(G, eps) / eps)
