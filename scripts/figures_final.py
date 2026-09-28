"""Figures for the revised paper (GeoTrust+ on RFC 3626 OLSR; faithful Zhou regime)."""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

FIG = "eval/figures"
os.makedirs(FIG, exist_ok=True)
STYLE = {"OLSR": ("#2a78d6", "o", "OLSR (RFC 3626)"), "TOPO-OLSR": ("#eb6834", "s", "OLSR + topology check"),
         "OLSR-W": ("#1baf7a", "^", "OLSR + watchdog"), "OLSR-G": ("#e87ba4", "*", "OLSR + GeoTrust+"),
         "GRR": ("#8a8a8a", "D", "GRR [Zhou]")}
M = ["OLSR", "TOPO-OLSR", "OLSR-W", "OLSR-G"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "axes.edgecolor": "#888", "lines.linewidth": 2})


def seedlevel(path):
    d = pd.read_parquet(path)
    return d.groupby(["factor", "level", "method", "seed"])[["e2e_q", "incl", "size"]].mean().reset_index()


def mci(x):
    x = np.asarray(x, float)
    return x.mean(), stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def attack_bars(s, fname, title):
    atk = ["A1", "A2", "A9", "A10a", "A10b", "A12", "A13", "A14", "PHY", "all"]
    fig, axs = plt.subplots(2, 1, figsize=(7.2, 4.8))
    w = 0.8 / len(M)
    for ax, k, yl in zip(axs, ["incl", "e2e_q"], ["attacker inclusion", "end-to-end PDR"]):
        for j, m in enumerate(M):
            vals = []
            for a in atk:
                g = s[(s.method == m) & (((s.factor == "attack") & (s.level == a)) | ((a == "all") & (s.factor == "main")))]
                vals.append(mci(g[k]))
            ax.bar(np.arange(len(atk)) + (j - 1.5) * w, [v[0] for v in vals], w * 0.9, yerr=[v[1] for v in vals],
                   color=STYLE[m][0], label=STYLE[m][2], capsize=1.5, error_kw=dict(lw=0.8))
        ax.set_xticks(range(len(atk)), atk); ax.set_ylabel(yl)
    axs[1].set_ylim(0.5, 1.02)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(f"{FIG}/{fname}.png", dpi=200); fig.savefig(f"{FIG}/{fname}.pdf"); plt.close(fig)


def frac_plot(sparse, dense, fname):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    for ax, s, ttl in zip(axs, (sparse, dense), ("sparse swarm (R$_c$=2 m)", "dense swarm (R$_c$=4 m)")):
        for m in M + ["GRR"]:
            xs, ys, es = [], [], []
            for lv, fac in ((0.0, ("att_frac", "0.0")), (0.05, ("att_frac", "0.05")), (0.10, ("main", "None")), (0.2, ("att_frac", "0.2"))):
                g = s[(s.method == m) & (s.factor == fac[0]) & (s.level == fac[1])]
                mu, h = mci(g.e2e_q); xs.append(lv); ys.append(mu); es.append(h)
            c, mk, lab = STYLE[m]
            ax.errorbar(xs, ys, yerr=es, color=c, marker=mk, ms=6, capsize=2, label=lab, markeredgecolor="white",
                        markeredgewidth=0.8, linestyle="--" if m == "GRR" else "-")
        ax.set_title(ttl, fontsize=8); ax.set_xlabel("insider fraction")
    axs[0].set_ylabel("end-to-end PDR")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=5, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(f"{FIG}/{fname}.png", dpi=200); fig.savefig(f"{FIG}/{fname}.pdf"); plt.close(fig)


def zhou_diag(fname):
    """Why the base scheme needs rethinking: GRR ~ flooding under Eq. (3), and flooding is hurt as much."""
    d = pd.read_parquet("eval/results/grid_zhou.parquet")
    s = d.groupby(["factor", "level", "method", "seed"])[["e2e", "size", "incl"]].mean().reset_index()
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.5))
    meths = [("FLOOD", "#b0b0b0", "flooding"), ("GRR", "#8a8a8a", "GRR [Zhou]"), ("OLSR", "#2a78d6", "OLSR (RFC 3626)")]
    for j, (m, c, lab) in enumerate(meths):
        g = s[(s.method == m) & (s.factor == "main")]
        axs[0].bar(j, g["size"].mean(), color=c, width=0.6)
        g0 = s[(s.method == m) & (s.factor == "att_frac") & (s.level == "0.0")]
        axs[1].bar(j - 0.18, g0.e2e.mean(), 0.34, color=c, alpha=0.5, label="no attack" if j == 0 else None)
        axs[1].bar(j + 0.18, g.e2e.mean(), 0.34, color=c, label="10% insiders" if j == 0 else None)
    for ax in axs:
        ax.set_xticks(range(3), [x[2] for x in meths], fontsize=7)
    axs[0].set_ylabel("relays per node |S|"); axs[1].set_ylabel("end-to-end PDR (no quarantine)")
    axs[1].set_ylim(0.6, 1.0); axs[1].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(f"{FIG}/{fname}.png", dpi=200); fig.savefig(f"{FIG}/{fname}.pdf"); plt.close(fig)


if __name__ == "__main__":
    sp, de = seedlevel("eval/results/grid_rfc.parquet"), seedlevel("eval/results/grid_rfcdense.parquet")
    attack_bars(de, "fig_rfc_attacks_dense", "dense")
    attack_bars(sp, "fig_rfc_attacks_sparse", "sparse")
    frac_plot(sp, de, "fig_rfc_fraction")
    zhou_diag("fig_zhou_diag")
    print("ok")
