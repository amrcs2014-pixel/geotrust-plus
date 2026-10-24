"""Reciprocity blame rule: the v1.0 greedy cover can frame honest agents under collusion; the threshold rule cannot
(as long as at most coll_f insiders lie to any honest agent) and does not depend on the order of the edges."""
import itertools
import numpy as np
import pytest
from geotrust.config import Config
from geotrust.verify import Verifier

TAU = {"R": 0.04, "K": 10.0, "CM": 10.0, "P": 0.01}


def blamed(edges, nodes, **cfg):
    """Run Verifier.exclude for verifier 0 on a disagreement graph given as (a, b) pairs."""
    ver = Verifier(Config(device="cpu", **cfg), tau=TAU)
    E = {0: [(a, b, 1.0, None) for a, b in edges]}
    G1 = np.array(nodes)
    mask, _, _ = ver.exclude(0, G1, {}, E, claims={}, use_cm=False)
    return set(G1[mask].tolist())


# insiders 1, 2 both lie to honest 3, 4 (a K_{2,2}); edges listed so that an honest node comes first
K22 = [(3, 1), (3, 2), (4, 1), (4, 2)]


def test_greedy_cover_frames_honest_agents_under_collusion():
    """Counterexample to Proposition 2 of v1.0 (PLOS ONE editorial assessment, 2026-10)."""
    assert blamed(K22, [1, 2, 3, 4], blame="greedy") == {3, 4}       # both honest convicted, both insiders spared


def test_threshold_rule_with_matching_collusion_bound_is_not_framed():
    # two colluders per honest agent -> coll_f = 2: no honest conviction, the two-link liars stay below threshold
    assert blamed(K22, [1, 2, 3, 4], blame="threshold", coll_f=2) == set()
    # a liar on coll_f + 1 = 3 links is convicted
    assert blamed(K22 + [(1, 5)], [1, 2, 3, 4, 5], blame="threshold", coll_f=2) == {1}


def test_threshold_rule_is_order_independent():
    edges = K22 + [(1, 5), (2, 6), (7, 8)]
    nodes = list(range(1, 9))
    ref = blamed(edges, nodes, blame="threshold", coll_f=1)
    assert ref == {1, 2, 3, 4}        # assumption violated (2 colluders > coll_f): all four reach degree 2
    for perm in itertools.islice(itertools.permutations(edges), 0, 5040, 97):
        assert blamed(list(perm), nodes, blame="threshold", coll_f=1) == ref
        assert blamed([(b, a) for a, b in perm], nodes, blame="threshold", coll_f=1) == ref


def test_one_sided_edge_rule_prevents_framing_by_omission():
    """Insider 1 omits its links to honest 3 and 4, so their true claims look one-sided (A14)."""
    ver = lambda **cfg: Verifier(Config(device="cpu", coll_f=2, **cfg), tau=TAU)
    one = lambda claimant: (claimant, 1.0)              # (claimant, claimed distance) well inside the range
    E = {0: [(1, 3, np.inf, one(3)), (1, 4, np.inf, one(4))]}
    G1 = np.array([1, 3, 4])
    m, _, _ = ver(one_sided="claimant").exclude(0, G1, {}, E, claims={}, use_cm=False)
    assert set(G1[m].tolist()) == {3, 4}                # v1.0 rule: both honest claimants convicted
    m, _, _ = ver(one_sided="edge").exclude(0, G1, {}, E, claims={}, use_cm=False)
    assert set(G1[m].tolist()) == set()                 # edge rule: two omissions stay within the lie budget f=2
    E[0].append((1, 5, np.inf, one(5)))
    m, _, _ = ver(one_sided="edge").exclude(0, np.array([1, 3, 4, 5]), {}, E, claims={}, use_cm=False)
    assert set(np.array([1, 3, 4, 5])[m].tolist()) == {1}   # a third omission convicts the omitter only


def test_geometry_rule_blames_only_impossible_claims():
    """One-sided claims carry the verifier's own triangle bounds (lo, hi); only claims outside them convict."""
    ver = Verifier(Config(device="cpu", coll_f=2, one_sided="geometry"), tau=TAU)
    m_ = 4.0 * np.sqrt(3.0) * ver.sig
    E = {0: [(1, 3, np.inf, (3, 1.0, 0.5, 3.0)),         # honest claim, plausible (omission by 1): nobody blamed
             (2, 4, np.inf, (2, 0.3, 2.0, 4.0))]}        # fake short link between agents >= 2 m apart: blame 2
    G1 = np.array([1, 2, 3, 4])
    m, _, _ = ver.exclude(0, G1, {}, E, claims={}, use_cm=False)
    assert set(G1[m].tolist()) == {2}
    E = {0: [(2, 4, np.inf, (2, 2.0 - 0.9 * m_, 2.0, 4.0))]}   # within the noise margin: not blamed
    m, _, _ = ver.exclude(0, G1, {}, E, claims={}, use_cm=False)
    assert not m.any()


@pytest.mark.parametrize("f", [1, 2, 3])
def test_threshold_rule_guarantees_on_random_graphs(f):
    """Proposition (revised): if every honest agent is lied to by at most f insiders, no honest agent is blamed,
    and every insider that lies to at least f+1 distinct agents is blamed - for any graph and edge order."""
    rng = np.random.default_rng(f)
    for _ in range(300):
        n_ins, n_hon = int(rng.integers(1, 6)), int(rng.integers(2, 12))
        ins, hon = list(range(1, n_ins + 1)), list(range(n_ins + 1, n_ins + n_hon + 1))
        edges = set()
        for h in hon:                                   # at most f insiders lie to each honest agent
            for v in rng.choice(ins, size=min(f, n_ins, int(rng.integers(0, f + 1))), replace=False):
                edges.add((int(v), h))
        for a, b in itertools.combinations(ins, 2):     # insiders may also disagree among themselves
            if rng.random() < 0.3:
                edges.add((a, b))
        edges = list(edges)
        rng.shuffle(edges)
        got = blamed(edges, ins + hon, blame="threshold", coll_f=f)
        assert not got & set(hon)
        deg = {v: len({b for a, b in edges if a == v} | {a for a, b in edges if b == v}) for v in ins}
        assert {v for v in ins if deg[v] >= f + 1} <= got
