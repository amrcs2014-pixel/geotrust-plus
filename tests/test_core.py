import itertools
import numpy as np
import pytest
from geotrust.sim import plr
from geotrust import mpr, auth
from geotrust.verify import _cm_z


def test_plr_sanity():
    # P(d) of Zhou et al.: ~0 at short range, rising sharply after 5 m, ~1 at 8 m
    assert plr(2) < 0.002
    assert 0.35 < plr(6) < 0.5
    assert plr(8) > 0.9


def rand_instance(rng, n=10, k=8):
    P_iv = rng.uniform(0.0, 0.3, n)
    P_vu = np.where(rng.random((n, k)) < 0.5, rng.uniform(0, 0.6, (n, k)), np.nan)
    P_vu[rng.integers(0, n, k), np.arange(k)] = rng.uniform(0, 0.6, k)   # every u coverable
    return mpr.contributions(P_iv, P_vu, 1.2)


@pytest.mark.parametrize("f", [0, 1, 2])
def test_robust_greedy_certified(f):
    rng = np.random.default_rng(f)
    for _ in range(30):
        C, t = rand_instance(rng)
        S = mpr.robust_greedy(C, t, f=f)
        M, T = mpr.robust_targets(C, t, f)
        # every |F|<=f failure leaves the (capped) targets satisfied
        for F in itertools.chain.from_iterable(itertools.combinations(S, j) for j in range(f + 1)):
            rest = [v for v in S if v not in F]
            got = C[rest].sum(0) if rest else np.zeros(C.shape[1])
            cap = np.minimum(t, C[[v for v in range(C.shape[0]) if v not in F]].sum(0))
            assert np.all(got >= cap - 1e-6)


@pytest.mark.parametrize("f", [0, 1])
def test_ilp_not_worse_than_greedy(f):
    rng = np.random.default_rng(7)
    for _ in range(20):
        C, t = rand_instance(rng)
        assert len(mpr.ilp(C, t, f=f)) <= len(mpr.robust_greedy(C, t, f=f))


def test_Gf_submodular():
    """Diminishing returns of G_f(S) = sum_F sum_u min(T, sum_{S\\F} c) on random instances."""
    rng = np.random.default_rng(3)
    for _ in range(200):
        C, t = rand_instance(rng, n=7, k=5)
        f = int(rng.integers(0, 3))
        M, T = mpr.robust_targets(C, t, f)
        A = [v for v in range(7) if rng.random() < 0.3]
        B = sorted(set(A) | {v for v in range(7) if rng.random() < 0.4})
        x = [v for v in range(7) if v not in B]
        if not x:
            continue
        x = x[0]
        gA = mpr.robust_value(C, T, M, A + [x]) - mpr.robust_value(C, T, M, A)
        gB = mpr.robust_value(C, T, M, B + [x]) - mpr.robust_value(C, T, M, B)
        assert gA >= gB - 1e-9


def _dist(P):
    return np.linalg.norm(P[:, None] - P[None], axis=-1)


def test_cm_sign():
    rng = np.random.default_rng(0)
    P = rng.uniform(0, 4, (200, 5, 3))
    d = np.stack([_dist(p) for p in P])
    z0 = _cm_z(d, 0.03, "cpu")[0]
    assert np.median(np.abs(z0)) < 1e-3                  # exact geometry -> ~0
    ds = d.copy()
    ds[:, 0, 1] = ds[:, 1, 0] = d[:, 0, 1] * 0.6         # v claims to be closer to u
    zs = _cm_z(ds, 0.03, "cpu")[0]
    assert (zs < 0).mean() > 0.9                         # optimistic lie -> negative z
    dl = d.copy()
    dl[:, 0, 1] = dl[:, 1, 0] = d[:, 0, 1] + 1.5         # pessimistic (enlarged)
    el, sdl = _cm_z(dl, 0.03, "cpu")
    ok = sdl < 0.2                                       # the verifier only uses well-conditioned tuples
    # sign is right for ~82 % of large enlargements (mirror root); what matters is that an enlargement
    # is almost never mistaken for an optimistic lie at the calibrated threshold (~0.8 m)
    assert (el[ok] > 0).mean() > 0.75
    assert (-el[ok] > 0.8).mean() < 0.1


def test_tesla_forgery_rejected():
    ch = auth.Chain(length=64, seed=b"k" * 16)
    rx = auth.Receiver(ch.key(0))
    ok = []
    for r in range(1, 6):
        payload = f"RU{r}".encode()
        fr = auth.frame(ch, r, payload)
        if r == 3:                                        # attacker alters payload, keeps MAC
            fr = b"RUX" + fr[3:]
        ok += rx.receive(r, fr)
    assert b"RU3" not in ok and b"RUX" not in ok
    assert b"RU2" in ok and b"RU4" in ok
    assert auth.overhead_table([20])[0]["auth_bytes"] == 24
