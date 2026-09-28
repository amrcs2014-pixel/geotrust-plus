"""Full evaluation grid (one-factor-at-a-time around the main configuration) on GPU0.
Usage:  python scripts/run_grid.py --seeds 10 --workers 4
Writes eval/results/grid.parquet (one row per seed x eval-round x method)."""
import argparse, os, time, json
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from geotrust.config import Config
from geotrust.calib import calibrate, CALIB_FILE, load
from geotrust.runner import run, RunTimeout

MAIN = Config()
# faithful regime of Zhou et al. (Eq. 3: Q_min over ALL one-hop relays; R_c = 2 m; sparse swarm)
ZHOU_MAIN = Config(qmin="all", rc=2.0, mean_degree=5.0)
ZHOU_OFAT = {
    "n": [10, 30, 100], "att_frac": [0.0, 0.05, 0.2],
    "attack": ["A1", "A2", "A9", "A10a", "A10b", "A12", "A13", "A14", "PHY"], "speed": [0.2, 2.0], "n_obst": [8],
}
RFC_METHODS = ("FLOOD", "OLSR", "OLSR-W", "TOPO-OLSR", "OLSR-G", "GRR", "GRR-W", "GT+f0")
ABL_METHODS = ("OLSR", "TOPO-OLSR", "OLSR-W", "OLSR-V", "OLSR-Gweak", "OLSR-GnoAuth", "OLSR-G")
ABL_OFAT = {"attack": ["A1", "A10b", "A14", "PHY"], "att_frac": [0.2]}
SENS_METHODS = ("OLSR", "OLSR-W", "OLSR-G")
NET_METHODS = ("OLSR", "TOPO-OLSR", "OLSR-W", "OLSR-G")
NET_OFAT = {"att_place": ["degree", "betweenness"]}
SENS_OFAT = {"tau_scale": [0.5, 0.75, 1.5, 2.0, 4.0], "wd_z": [2.5, 4.5]}
RFC_DENSE_OFAT = {"attack": ["A1", "A2", "A9", "A10a", "A10b", "A12", "A13", "A14", "PHY"], "att_frac": [0.0, 0.05, 0.2]}
OFAT = {
    "n": [10, 30, 100, 200], "rc": [2.0, 8.0], "speed": [0.2, 2.0], "att_frac": [0.0, 0.05, 0.2],
    "attack": ["A1", "A2", "A9", "A10a", "A10b", "A12", "A13", "A14", "PHY"], "sigma": [0.10], "f": [0, 2],
    "gamma": [1.0, 1.4], "flat": [True], "q_drop": [0.5], "n_obst": [8], "qmin": ["all"],
}
METHODS = ("FLOOD", "OLSR", "GRR", "GRR2", "ILP", "SG", "TOPO-GRR", "GRR-W", "GT-R", "GT-V", "GT-noCM",
           "GeoTrust", "GeoTrust-noAuth", "GeoTrust-RILP", "GT+f0", "GT+noCM", "GT+fixF", "GT+", "GT+noWD",
           "GT+weak")


def configs(regime="default"):
    main, ofat = {"zhou": (ZHOU_MAIN, ZHOU_OFAT), "rfc": (ZHOU_MAIN, ZHOU_OFAT),
                  "rfcdense": (MAIN.with_(qmin="all"), RFC_DENSE_OFAT),
                  "abl_sparse": (ZHOU_MAIN, ABL_OFAT), "abl_dense": (MAIN.with_(qmin="all"), ABL_OFAT),
                  "sens_sparse": (ZHOU_MAIN, SENS_OFAT), "sens_dense": (MAIN.with_(qmin="all"), SENS_OFAT),
                  "net_sparse": (ZHOU_MAIN, NET_OFAT), "net_dense": (MAIN.with_(qmin="all"), NET_OFAT),
                  }.get(regime, (MAIN, OFAT))
    out = [("main", None, main)]
    for k, vals in ofat.items():
        for v in vals:
            out.append((k, v, main.with_(**{k: v})))
    return out

def calib_job(cfg):
    return cfg.key(), calibrate(cfg, seeds=10)

def log(msg):
    with open("eval/results/jobs.log", "a") as fh:
        fh.write(time.strftime("%H:%M:%S") + " " + msg + "\n")


PARTS = "eval/results/parts"


def part_path(factor, level, seed, parts=None):
    return os.path.join(parts or PARTS, f"{factor}={level}_s{seed}.parquet")


def run_job(args):
    """Runs one job and writes its rows to its own part file; only a short status crosses the
    process boundary (large results jammed the Windows process-pool result pipe)."""
    factor, level, cfg, tau, parts, methods = args
    t0 = time.time()
    log(f"START {factor}={level} seed={cfg.seed}")
    try:
        rows = run(cfg, tau, methods=methods, budget_s=900)
    except RunTimeout as e:
        log(f"TIMEOUT {factor}={level} seed={cfg.seed} {e}")
        return "timeout"
    for r in rows:
        r["factor"], r["level"], r["wall_s"] = factor, str(level), time.time() - t0
    tmp = part_path(factor, level, cfg.seed, parts) + ".tmp"
    pd.DataFrame(rows).to_parquet(tmp)
    os.replace(tmp, part_path(factor, level, cfg.seed, parts))        # atomic: a part file is always complete
    log(f"END {factor}={level} seed={cfg.seed} {time.time() - t0:.0f}s")
    return "ok"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="eval/results/grid.parquet")
    ap.add_argument("--regime", default="default",
                    choices=["default", "zhou", "rfc", "rfcdense", "abl_sparse", "abl_dense", "sens_sparse", "sens_dense",
                             "net_sparse", "net_dense"])
    ap.add_argument("--n200", choices=["include", "skip", "only"], default="include",
                    help="N=200 runs need ~2.3 GB each: run them separately with few workers")
    a = ap.parse_args()
    cfgs = configs(a.regime)
    parts = PARTS if a.regime == "default" else PARTS + "_" + a.regime
    db = load()
    todo = {c.key(): c for _, _, c in cfgs if c.key() not in db}
    t0 = time.time()
    os.makedirs(os.path.dirname(CALIB_FILE), exist_ok=True)
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(calib_job, c): k for k, c in todo.items()}
        for fu in as_completed(futs):
            k, tau = fu.result()
            db[k] = tau
            json.dump(db, open(CALIB_FILE, "w"), indent=1)          # checkpoint every calibration
            print(f"[calib] {k} {json.dumps({x: round(y, 3) for x, y in tau.items()})}", flush=True)
    # N=200 is a scaling check: 10 seeds; everything else a.seeds
    methods = (ABL_METHODS if a.regime.startswith("abl") else SENS_METHODS if a.regime.startswith("sens")
               else NET_METHODS if a.regime.startswith("net")
               else RFC_METHODS if a.regime.startswith("rfc") else METHODS)
    jobs = [(f, l, c.with_(seed=s), db[c.key()], parts, methods) for f, l, c in cfgs
            for s in range(10 if (f == "n" and l == 200) else a.seeds)]
    os.makedirs(parts, exist_ok=True)
    base = a.out.replace(".parquet", "_base.parquet")           # runs finished before part files existed
    if os.path.exists(a.out) and not os.path.exists(base):
        os.replace(a.out, base)
    have = set()
    if os.path.exists(base):
        old = pd.read_parquet(base, columns=["factor", "level", "seed"])
        have = set(zip(old.factor, old.level, old.seed))
    jobs = [j for j in jobs if (j[0], str(j[1]), j[2].seed) not in have
            and not os.path.exists(part_path(j[0], j[1], j[2].seed, parts))]
    big = lambda j: j[0] == "n" and str(j[1]) == "200"
    if a.n200 == "skip":
        jobs = [j for j in jobs if not big(j)]
    elif a.n200 == "only":
        jobs = [j for j in jobs if big(j)]
    print(f"[resume] {len(have)} base runs, {len(os.listdir(parts))} part files, {len(jobs)} to go", flush=True)
    # interleave heavy (large N) and light jobs so progress stays visible
    jobs.sort(key=lambda j: (j[2].seed, j[0], str(j[1])))
    done = 0
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(run_job, j) for j in jobs]
        for fu in as_completed(futs):
            st = fu.result(); done += 1
            print(f"[run] {done}/{len(jobs)} {st} {time.time()-t0:.0f}s", flush=True)
    # merge base + parts into the final table
    frames = [pd.read_parquet(base)] if os.path.exists(base) else []
    frames += [pd.read_parquet(os.path.join(parts, p)) for p in sorted(os.listdir(parts)) if p.endswith(".parquet")]
    full = pd.concat(frames, ignore_index=True).drop_duplicates(["factor", "level", "seed", "round", "method"])
    full.to_parquet(a.out)
    print("done", len(full), "rows", full.drop_duplicates(["factor", "level", "seed"]).shape[0], "runs")
