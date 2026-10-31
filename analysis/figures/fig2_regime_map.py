"""Figure 2: where the real traits sit on the theoretical map of attainable spectral gain (a phase diagram).

Background: the oracle gap (Theorem 2), i.e. the largest in-sample R2 any eigenvalue-indexed spectral filter could add over
the best ridge, as a function of the power-law exponent gamma and the heritability h2, computed on the eigenvalues of the
mouse GRM (REML basis, intercept only), shown as discrete bands. Points: the 61 real traits at their REML gamma-hat and fitted
h2 (their bootstrap CIs are in Figure 6). Each trait's own oracle gap, computed on its own panel's spectrum (gap_real.json),
agrees closely with the band at its position (checked for the traits with the largest gaps).
usage: python fig2_regime_map.py       writes out/fig2_regime_map.pdf/.png
"""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from figstyle import setup, save, INK, DARKGREY, DOUBLE, MM, RES
from fig_theory_sim_test import spectrum, gap

setup()
LEVELS = [0, 0.0005, 0.001, 0.005, 0.01, 0.02, 1.0]
BANDS = ["#FFFFFF", "#EEF3FA", "#D6E4F3", "#B0CBE7", "#7FA9D6", "#4F84BE"]          # ColorBrewer-like Blues, light end
SHAPE = {"rice": ("o", "Rice"), "wheat": ("s", "Wheat"), "mice": ("^", "Mouse HS"), "rdp1": ("D", "Rice RDP1"), "sorghum": ("v", "Sorghum")}


def J(fn): return json.load(open(os.path.join(RES, fn)))


def main():
    L = spectrum("mice"); L = L[L > 1e-6]
    gs = np.linspace(0.1, 3.2, 94); hs = np.linspace(0.005, 0.995, 56)
    cache = os.path.join(RES, "fig1_gap_grid.npy")
    Z = np.load(cache) if os.path.exists(cache) and np.load(cache).shape == (len(hs), len(gs)) else None
    if Z is None:
        Z = np.array([[gap(L, g, h)[0] for g in gs] for h in hs]); np.save(cache, Z)

    fig = plt.figure(figsize=(DOUBLE, 74 * MM))
    ax = fig.add_axes([0.075, 0.15, 0.74, 0.76]); cax = fig.add_axes([0.835, 0.15, 0.016, 0.42])
    cmap, norm = ListedColormap(BANDS), BoundaryNorm(LEVELS, len(BANDS))
    cf = ax.contourf(gs, hs, Z, levels=LEVELS, cmap=cmap, norm=norm, zorder=0)
    ax.contour(gs, hs, Z, levels=LEVELS[1:-1], colors="white", linewidths=0.6, zorder=1)
    ax.axvline(1.0, color=INK, lw=0.6, ls=(0, (4, 2)), zorder=2)

    E = {(x["dataset"], x["trait"]): x for x in J("e7_reml_boot.json")}
    G = {(x["dataset"], x["trait"]): x for x in J("gap_real.json")}
    for k, x in E.items():
        m = SHAPE[k[0]][0]
        ax.plot(x["gamma"], G[k]["h2"], marker=m, ms=4.2 if m != "D" else 3.6, mfc="white", mec=INK, mew=0.6, lw=0, zorder=4)

    ax.set_xlim(0.1, 3.2); ax.set_ylim(0.0, 1.02)
    ax.set_xticks([0.25, 0.5, 1, 1.5, 2, 2.5, 3]); ax.set_xticklabels(["0.25", "0.5", "1", "1.5", "2", "2.5", "3"])
    ax.set_xlabel(r"Power-law exponent $\gamma$ (points: REML $\hat{\gamma}$)")
    ax.set_ylabel(r"Heritability $h^2$ (points: REML estimate)")
    for s in ("top", "right"): ax.spines[s].set_visible(True)
    for s in ax.spines.values(): s.set_linewidth(0.6)

    cb = fig.colorbar(cf, cax=cax, ticks=LEVELS[:-1]); cb.outline.set_linewidth(0.5)
    cb.ax.set_yticklabels(["0", "0.0005", "0.001", "0.005", "0.01", "0.02"]); cb.ax.tick_params(labelsize=6, length=2)
    cb.set_label("Oracle gap (in-sample R²)", fontsize=6.6)
    h = [Line2D([], [], marker=m, mfc="white", mec=INK, mew=0.6, lw=0, ms=4.2 if m != "D" else 3.6, label=lab) for m, lab in SHAPE.values()]
    h.append(Line2D([], [], color=INK, lw=0.6, ls=(0, (4, 2)), label=r"$\gamma=1$ (GBLUP)"))
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.825, 0.92), fontsize=6.3, handletextpad=0.4, borderaxespad=0)
    save(fig, "fig2_regime_map")


if __name__ == "__main__":
    main()
