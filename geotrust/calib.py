"""Attack-free calibration of verification thresholds tau_k (quantile 1 - alpha/4 per test).
Calibration seeds (100000+) are disjoint from evaluation seeds."""
import json
import os
import numpy as np
from .runner import run
from .verify import TESTS

CALIB_FILE = os.path.join(os.path.dirname(__file__), "..", "eval", "calibration.json")
ALPHA = 0.01


def calibrate(cfg, seeds=10, alpha=ALPHA):
    null = {}
    for s in range(seeds):
        r = run(cfg.with_(att_frac=0.0, attack="none", seed=100_000 + s), collect_null=True)
        for k, v in r.items():
            null.setdefault(k, []).append(v[~np.isnan(v)])
    tau = {}
    for k in TESTS:
        x = np.concatenate(null[k]) if k in null else np.zeros(0)
        tau[k] = float(np.quantile(x, 1 - alpha / len(TESTS))) if x.size else float("inf")
        tau[k] = max(tau[k], 1e-6)
    # an honest (gated) claim cannot shrink faster than the kinematic gate allows
    g = 2 * cfg.speed * cfg.T + 4 * np.hypot(cfg.sigma, cfg.sigma_ind)
    tau["K"] = float(max(tau["K"], 1.05 * g))
    tau["_n"] = int(sum(len(x) for x in null.get("CM", [])))
    return tau


def load():
    if os.path.exists(CALIB_FILE):
        with open(CALIB_FILE) as fh:
            return json.load(fh)
    return {}


def get_tau(cfg, seeds=10):
    db = load()
    k = cfg.key()
    if k not in db:
        db[k] = calibrate(cfg, seeds)
        db = {**load(), k: db[k]}
        os.makedirs(os.path.dirname(CALIB_FILE), exist_ok=True)
        with open(CALIB_FILE, "w") as fh:
            json.dump(db, fh, indent=1)
    return db[k]
