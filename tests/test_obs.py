"""Faithful observation model (geotrust/obs.py): two-way-ranging freshness, omission, verifier views, persistence."""
import numpy as np
from geotrust.config import Config
from geotrust.obs import Observation


def setup(n=4, **kw):
    cfg = Config(n=n, device="cpu", obs="ru", **kw)
    ob = Observation(cfg, np.random.default_rng(0))
    D = np.full((n, n), 1.0)
    np.fill_diagonal(D, 0.0)
    adj = ~np.eye(n, dtype=bool)
    return cfg, ob, D, adj


def period(ob, D, adj, loss=0.0, omit=None):
    ob.omit = omit or {}
    rx, ok = ob.links(D, adj, np.full(D.shape, loss))
    n = D.shape[0]
    CD, CR, fresh = ob.measure(D, D, np.zeros(n, bool), ok)
    CP = np.zeros_like(CD)
    return rx, ok, CD, CR, fresh, CP


def full_period(ob, D, adj, **kw):
    """A complete period as in the runner: links, measurement, then delivery (advances the period counter)."""
    rx, ok, CD, CR, fresh, CP = period(ob, D, adj, **kw)
    ob.deliver(rx, CD, CP, CR)
    return rx, ok, CD, CR, fresh, CP


def test_exchange_needs_both_directions_and_previous_period():
    cfg, ob, D, adj = setup()
    _, ok, CD, *_ = period(ob, D, adj)
    assert not ok.any() and np.isnan(CD).all()           # first period: no earlier RU, nothing measured
    _, ok, CD, *_ = period(ob, D, adj)
    assert ok[~np.eye(4, dtype=bool)].all()              # second period: every exchange completes
    assert np.isfinite(CD[~np.eye(4, dtype=bool)]).all()


def test_omission_prevents_the_victim_from_measuring():
    cfg, ob, D, adj = setup(claim_ttl=1)
    full_period(ob, D, adj)
    full_period(ob, D, adj)
    for _ in range(3):
        _, ok, CD, *_ = full_period(ob, D, adj, omit={0: {1}})
    assert not ok[1, 0] and ok[0, 1]                     # 1 can no longer range with 0; 0 still ranges with 1
    assert np.isnan(CD[1, 0]) and np.isfinite(CD[0, 1])  # 1's entry for 0 expired (ttl = 1): no one-sided claim


def test_views_are_usable_one_period_after_reception_and_expire():
    cfg, ob, D, adj = setup(fresh_w=1)
    rx, ok, CD, CR, fresh, CP = period(ob, D, adj)
    assert not ob.usable().any()
    ob.deliver(rx, CD, CP, CR)
    assert ob.usable()[~np.eye(4, dtype=bool)].all()     # received last period: usable now (TESLA)
    none = np.zeros_like(rx)
    ob.deliver(none, CD, CP, CR)
    assert ob.usable()[~np.eye(4, dtype=bool)].all()     # age 1 <= fresh_w
    ob.deliver(none, CD, CP, CR)
    assert not ob.usable().any()                         # age 2 > fresh_w: unknown, not evidence


def test_one_sided_persistence_counts_consecutive_periods():
    cfg, ob, D, adj = setup()
    CD = np.full((4, 4), np.nan)
    CD[0, 1] = 1.0                                       # 0 lists 1; 1 lists nobody
    CR = np.zeros((4, 4), int)
    rx = adj.copy()
    for t in range(3):
        ob.deliver(rx, CD, np.zeros((4, 4)), CR)
        ob.update_streaks(np.zeros((4, 4), bool))
        assert ob.streak[2, 0, 1] == t + 1               # at verifier 2: claimant 0, partner 1
    ob.deliver(np.zeros_like(rx), CD, np.zeros((4, 4)), CR)
    ob.deliver(np.zeros_like(rx), CD, np.zeros((4, 4)), CR)
    ob.deliver(np.zeros_like(rx), CD, np.zeros((4, 4)), CR)
    ob.update_streaks(np.zeros((4, 4), bool))
    assert ob.streak[2, 0, 1] == 0                       # reports expired: unknown resets the streak
