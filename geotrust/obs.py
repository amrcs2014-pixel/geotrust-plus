"""Faithful observation model for ranging reports (cfg.obs == "ru").

The ideal model of v1.0 gave every verifier every agent's complete and current claims in every period.  Here:

* RU loss.  The RU that agent v broadcasts in a period reaches neighbour u with probability 1 - p_uv, where
  p_uv = 1 - (1 - P(d_uv)) (1 - ru_loss) combines the distance-dependent loss of the channel with an extra loss rate.
  With ru_burst = B > 1 losses are bursty: a link repeats its previous outcome with probability 1 - 1/B and otherwise
  draws afresh (same marginal loss rate, mean burst length about B).
* Two-way ranging freshness.  u obtains a new distance to v in period t only if it receives v's RU of period t, which
  carries v's reception timestamp of u's RU of period t-1, and v received that RU: ok[u,v] = rx_t[u,v] & rx_{t-1}[v,u].
  An insider that omits u from its RU (A14) therefore removes u's ability to measure the link.  Entries older than
  claim_ttl periods expire from an agent's table; every distance carries the period of the exchange it comes from.
* Verifier views.  Verifier i holds the last RU it received from each neighbour.  Because of TESLA a RU becomes usable
  one period after reception, and it stays usable for fresh_w periods; older or missing RUs are unknown, not evidence.
* Persistence.  At every verifier, the number of consecutive periods in which a link claim is one-sided (the claimant
  lists the partner, the partner's usable RU does not list the claimant) is tracked; a one-sided claim becomes
  evidence only after `persist` periods.  The verifier's own measurement counts as its claim about the neighbour.
"""
import numpy as np


class Observation:
    def __init__(self, cfg, rng):
        self.cfg, self.rng = cfg, rng
        n = cfg.n
        self.t = 0
        self.rx_prev = np.zeros((n, n), bool)
        self.lost_prev = np.zeros((n, n), bool)
        self.dtab = np.full((n, n), np.nan)          # agent u's last measured distance to v (honest table)
        self.rtab = np.full((n, n), -10 ** 6)        # period of that measurement (exchange label)
        # verifier views: held RU of v at i (claimed distances, PLR claims and labels), and its age in periods
        self.Hd = np.full((n, n, n), np.nan, np.float32)
        self.Hp = np.full((n, n, n), np.nan, np.float32)
        self.Hr = np.full((n, n, n), -10 ** 6, np.int32)
        # previous claim of an older exchange per (i, v, u), for the labelled kinematic test
        self.Hpd = np.full((n, n, n), np.nan, np.float32)
        self.Hpr = np.full((n, n, n), -10 ** 6, np.int32)
        self.hage = np.full((n, n), 10 ** 6)
        self.streak = np.zeros((n, n, n), np.int16)  # [i, claimant, partner] consecutive one-sided periods
        self.K = cfg.claim_ttl + cfg.fresh_w + 4     # own measurements kept by exchange label for this many periods
        self.obuf = np.full((self.K, n, n), np.nan)
        self.omit = {}                               # insider -> set of honest agents it omits (A14)

    # ------------------------------------------------------------------ channel and two-way ranging
    def links(self, D, adj, plr_true):
        """Draw this period's RU receptions and the two-way exchanges that complete."""
        c = self.cfg
        p = 1 - (1 - plr_true) * (1 - c.ru_loss)
        lost = self.rng.random(D.shape) < p
        if c.ru_burst > 1:
            keep = self.rng.random(D.shape) < 1 - 1 / c.ru_burst
            lost = np.where(keep, self.lost_prev, lost)
        self.lost_prev = lost
        rx = adj & ~lost                              # rx[u, v]: u received v's RU in this period
        ok = rx & self.rx_prev.T                      # ok[u,v] = rx_t[u,v] & rx_{t-1}[v,u]
        for a, victims in self.omit.items():          # a's RU carries no timestamps for an omitted u,
            for u in victims:                         # so u cannot complete the exchange with a
                ok[u, a] = False
        self.rx_prev = rx
        return rx, ok

    def measure(self, gated, raw, is_att, ok):
        """Update the agents' distance tables with the exchanges that completed (honest agents store the gated
        estimate, insiders the raw one); return the claims before forgery, their labels and the freshness mask."""
        c = self.cfg
        meas = np.where(is_att[:, None], raw, gated)
        self.obuf[self.t % self.K] = np.where(ok, meas, np.nan)
        self.dtab[ok] = meas[ok]
        self.rtab[ok] = self.t
        fresh = (self.t - self.rtab) <= c.claim_ttl
        CD = np.where(fresh, self.dtab, np.nan)
        CR = np.where(fresh, self.rtab, -10 ** 6)
        np.fill_diagonal(CD, np.nan)
        return CD, CR, fresh

    # ------------------------------------------------------------------ verifier views
    def usable(self):
        """Mask [i, v]: verifier i holds a usable (authenticated, fresh) RU of v.  Views are read before this period's
        delivery, so hage = 0 means 'received in the previous period' (key disclosed now, TESLA)."""
        return self.hage <= self.cfg.fresh_w

    def own_at(self, i, labels):
        """Verifier i's own distance to each agent in the exchange with the given labels (NaN if not kept)."""
        labels = np.asarray(labels)
        ok = (labels >= 0) & (self.t - labels < self.K) & (labels <= self.t)
        out = np.full(labels.shape, np.nan)
        idx = np.flatnonzero(ok)
        out[idx] = self.obuf[labels[idx] % self.K, i, idx]
        return out

    def view(self, i):
        """Claims usable at verifier i: rows of unknown agents are NaN.  Returns CD_i, CP_i, CR_i, known_i and
        CDprev_i, defined so that CDprev_i - CD_i is the per-period shrink of each claim between its two last
        exchanges (NaN if there is no earlier exchange): the kinematic test of the ideal model, made label-aware."""
        known = self.usable()[i]
        CD = np.where(known[:, None], self.Hd[i], np.nan).astype(float)
        CP = np.where(known[:, None], self.Hp[i], np.nan).astype(float)
        CR = np.where(known[:, None], self.Hr[i], -10 ** 6)
        gap = (self.Hr[i] - self.Hpr[i]).astype(float)
        rate = np.where((gap >= 1) & (gap <= 4 * (self.cfg.claim_ttl + 1)),
                        (self.Hpd[i] - self.Hd[i]) / np.maximum(gap, 1), np.nan)
        CDp = np.where(known[:, None], self.Hpd[i], np.nan).astype(float)     # previous exchange of each claim
        CRp = np.where(known[:, None], self.Hpr[i], -10 ** 6)
        return CD, CP, CR, known, CD + rate, CDp, CRp

    def update_streaks(self, own_fresh):
        """One-sided persistence at every verifier, on the views usable in this period (before delivery)."""
        n = self.cfg.n
        known = self.usable()                                       # [i, v]
        listed = ~np.isnan(self.Hd) & known[:, :, None]             # [i, v, u]: v's usable RU at i lists u
        both = known[:, :, None] & known[:, None, :]
        one = listed & ~np.transpose(listed, (0, 2, 1)) & both      # v lists u, u's usable RU does not list v
        idx = np.arange(n)
        # the verifier's own measurement is its claim: i "lists" v; v's usable RU omits i
        own_one = own_fresh & known & ~listed[idx, :, idx]          # [i, v]
        one[idx, idx, :] = own_one
        self.streak = np.where(one, self.streak + 1, 0).astype(np.int16)

    def deliver(self, rx, CD, CP, CR):
        """RUs broadcast in this period reach the verifiers that received them (usable from the next period)."""
        self.hage += 1
        newer = rx[:, :, None] & (CR[None] != self.Hr) & ~np.isnan(CD[None]) & ~np.isnan(self.Hd)
        self.Hpd = np.where(newer, self.Hd, self.Hpd)
        self.Hpr = np.where(newer, self.Hr, self.Hpr)
        self.Hd = np.where(rx[:, :, None], CD[None].astype(np.float32), self.Hd)
        self.Hp = np.where(rx[:, :, None], CP[None].astype(np.float32), self.Hp)
        self.Hr = np.where(rx[:, :, None], CR[None].astype(np.int32), self.Hr)
        self.hage = np.where(rx, 0, self.hage)
        self.t += 1
