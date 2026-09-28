"""Draw the ns-3 topology snapshot (top view of the 15 x 15 x 3 m arena) for OLSR, OLSR + watchdog and
OLSR + GeoTrust+ under the same seed and mobility.
Usage: python fig_ns3_topology.py [attack] [seed] [--paper]   (--paper: 2 x 2 layout, no title, larger fonts)"""
import sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PAPER = "--paper" in sys.argv
args = [a for a in sys.argv[1:] if a != "--paper"]
ATT = args[0] if len(args) > 0 else "shrink"
SEED = args[1] if len(args) > 1 else "3"
FS = 1.35 if PAPER else 1.0
SCHEMES = [("rfc", "OLSR (RFC 3626)"), ("rfcw", "OLSR + watchdog"), ("rfcg", "OLSR + GeoTrust+")]
NAMES = {"shrink": "adaptive distance liars (A10b)", "fakelinks": "fake links (A2)", "claim": "claim-all spoofing",
         "stealth": "stealth fake links (A12)", "wormhole": "wormhole (A9)"}

if PAPER:
    fig, grid = plt.subplots(2, 2, figsize=(8.2, 8.6))
    axes = [grid[0, 0], grid[0, 1], grid[1, 0]]
    grid[1, 1].axis("off")
else:
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.0))
for ax, (m, title) in zip(axes, SCHEMES):
    d = pd.read_csv(f"eval/results/ns3_topo_{ATT}_{m}_s{SEED}.csv")
    nodes = d[d.kind == "node"].astype({"a": int, "b": int}).set_index("a")
    meta = d[d.kind == "meta"].iloc[0]
    R = float(meta.x)
    P = nodes[["x", "y", "z"]].to_numpy(float)
    att = nodes.b.to_numpy().astype(bool)
    n = len(nodes)
    D = np.linalg.norm(P[:, None] - P[None], axis=2)
    # true links (3-D distance below the link range)
    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] < R:
                ax.plot(P[[i, j], 0], P[[i, j], 1], color="#d9d9d9", lw=0.5, zorder=1)
    # fake links advertised by insiders
    for _, r in d[d.kind == "fake"].iterrows():
        a, b = int(r.a), int(r.b)
        ax.plot(P[[a, b], 0], P[[a, b], 1], color="#d62728", lw=0.8, ls=(0, (3, 2)), alpha=0.7, zorder=2)
    # MPR selections of honest nodes: selector -> MPR
    mp = d[d.kind == "mpr"].astype({"a": int, "b": int})
    mp = mp[~att[mp.a.to_numpy()]]
    for _, r in mp.iterrows():
        a, b = int(r.a), int(r.b)
        col = "#d62728" if att[b] else "#1f77b4"
        ax.annotate("", xy=P[b, :2], xytext=P[a, :2], zorder=3,
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.1 if att[b] else 0.8, alpha=0.85,
                                    shrinkA=4, shrinkB=5, mutation_scale=7))
    share = att[mp.b.to_numpy()].mean() if len(mp) else 0.0
    # convictions: how many honest verifiers exclude each node
    ex = d[d.kind == "excl"].astype({"a": int, "b": int})
    conv = ex.groupby("b").size().reindex(range(n), fill_value=0).to_numpy()
    wrong = int(conv[~att].sum())
    ax.scatter(P[~att, 0], P[~att, 1], s=46, c="#4d4d4d", edgecolors="white", linewidths=0.6, zorder=4)
    ax.scatter(P[att, 0], P[att, 1], s=90, marker="^", c="#d62728", edgecolors="black", linewidths=0.6, zorder=5)
    for i in np.flatnonzero(att & (conv > 0)):
        ax.scatter(P[i, 0], P[i, 1], s=260, facecolors="none", edgecolors="black", linewidths=1.2, zorder=5)
        ax.text(P[i, 0] + 0.35, P[i, 1] + 0.35, f"x{conv[i]}", fontsize=7 * FS, zorder=6)
    for i in range(n):
        ax.text(P[i, 0], P[i, 1] - 0.62, str(i), fontsize=5.5, ha="center", color="#555", zorder=6)
    if PAPER:
        tag = "abc"[[s[0] for s in SCHEMES].index(m)]
        ax.set_title(f"({tag}) {title}\ninsiders hold {share:.0%} of MPR links\nhonest drones excluded: {wrong}",
                     fontsize=8 * FS)
    else:
        ax.set_title(f"{title}\ninsiders hold {share:.0%} of MPR links; honest drones excluded: {wrong}", fontsize=8)
    ax.set_xlim(-0.8, 15.8); ax.set_ylim(-0.8, 15.8); ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=8 * FS); ax.set_ylabel("y (m)", fontsize=8 * FS); ax.tick_params(labelsize=7 * FS)
handles = [Line2D([], [], color="#d9d9d9", lw=1, label=f"true link (< {R:g} m)"),
           Line2D([], [], color="#d62728", lw=1, ls="--", label="fake link advertised by insider"),
           Line2D([], [], color="#1f77b4", lw=1, marker=">", markersize=4, label="MPR choice -> honest relay"),
           Line2D([], [], color="#d62728", lw=1.2, marker=">", markersize=4, label="MPR choice -> insider"),
           Line2D([], [], ls="", marker="o", color="#4d4d4d", label="honest drone"),
           Line2D([], [], ls="", marker="^", color="#d62728", markeredgecolor="black", label="insider"),
           Line2D([], [], ls="", marker="o", markerfacecolor="none", markeredgecolor="black", markersize=10,
                  label="insider convicted (xk = by k verifiers)")]
if PAPER:
    grid[1, 1].legend(handles=handles, loc="center", fontsize=9, frameon=False)
    fig.tight_layout()
    out = f"eval/figures/fig_ns3_topology_{ATT}_s{SEED}_paper"
else:
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"ns-3 topology snapshot (30 drones, top view of 15 x 15 x 3 m arena, t = 50 s, "
                 f"10% insiders: {NAMES.get(ATT, ATT)}, seed {SEED})", fontsize=10)
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
    out = f"eval/figures/fig_ns3_topology_{ATT}_s{SEED}"
fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=200)
print(out)
