"""Experiment configuration for GeoTrust-MPR simulations."""
from dataclasses import dataclass, asdict, replace

# Zhou et al. (2026) empirical LoS PLR model  P(d) = 1 / (exp(a d + b) + c)
PLR_A, PLR_B, PLR_C = -1.61, 9.97, 1.01

ATTACK_TYPES = ("A1", "A2", "A9", "A10a", "A10b")


@dataclass(frozen=True)
class Config:
    n: int = 50                 # swarm size
    rc: float = 4.0             # communication range [m]
    mean_degree: float = 12.0   # target mean one-hop degree (sets cube side)
    speed: float = 1.0          # max speed v_max [m/s]
    T: float = 0.1              # ranging period [s]
    sigma: float = 0.03         # ranging noise shared by both ends of a DS-TWR exchange [m]
    sigma_ind: float = 0.01     # independent per-end timestamp noise [m]
    att_frac: float = 0.10      # fraction of insider attackers
    attack: str = "all"         # none | A1 | A2 | A9 | A10a | A10b | PHY | all
    f: int = 1                  # robustness parameter of the relay cover
    gamma: float = 1.2
    qmin: str = "single"        # "single": best single-relay Q_min (primary) | "all": product over all relays (RangeGuard port of Alg. 1)
    beta: float = 0.7           # fusion weight of P(d) vs EWMA
    alpha: float = 0.3          # EWMA factor
    rounds: int = 40
    warmup: int = 10
    eval_every: int = 5
    flat: bool = False          # fixed-altitude (near-coplanar) formation
    floods: int = 10            # MPR-flooding trials per evaluation round
    ilp: bool = True            # run exact ILP oracles
    q_drop: float = 1.0         # attacker drop probability on forwarded traffic (1 = black-hole)
    n_obst: int = 0             # number of box obstacles (NLoS scenario)
    strong_frac: float = 0.7    # one-sided claims shorter than strong_frac*rc blame the claimant
    wd_msgs: int = 25           # data broadcasts per verifier per watchdog window (20 ms period)
    wd_z: float = 3.5           # watchdog conviction threshold (one-sided z)
    tau_scale: float = 1.0      # sensitivity study: multiplies every calibrated verification threshold
    att_place: str = "random"   # insider placement: random | degree | betweenness (targeted, on the initial graph)
    seed: int = 0
    device: str = "cuda:0"

    def key(self):
        """Signature used for the attack-free calibration of verification thresholds."""
        return (f"n{self.n}_rc{self.rc}_v{self.speed}_s{self.sigma}_d{self.mean_degree}_flat{int(self.flat)}"
                + (f"_obst{self.n_obst}" if self.n_obst else ""))

    def asdict(self):
        return asdict(self)

    def with_(self, **kw):
        return replace(self, **kw)
