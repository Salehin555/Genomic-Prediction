"""Figures 3-5 (numbering of Planning/12): theory (analytic, on real GRM spectra), simulation regime map (E1), calibration of the REML test.
usage: python fig_theory_sim_test.py      writes out/fig3_theory, out/fig4_simulation, out/fig5_test_calibration (.pdf, .png)
"""
import os, sys, json
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from figstyle import setup, save, value_grid, zero_line, panel_label, subtitle, sat, METHOD, LABEL, TOL, DARKGREY, SINGLE, DOUBLE, MM, RES

setup()


def J(fn): return json.load(open(os.path.join(RES, fn)))


# ------------------------------------------------------------------ Figure 2: theory
def spectrum(name):
    from e13_reml_diag import load, reml_basis
    D = load(name); T = next(iter(D["traits"].values())); n = len(D["G"])
    lam, _ = reml_basis(D["G"], np.ones((n, 1))); return np.sort(lam)[::-1]


def weights(lam, g, h2=0.5):
    tau = lam ** g; tau = tau / tau.mean() * h2 / (1 - h2); return tau, tau / (tau + 1.0)


def gap(lam, g, h2=0.5):
    tau, w = weights(lam, g, h2)
    f = lambda ld: np.mean((tau + 1) * (lam / (lam + np.exp(ld)) - w) ** 2)
    r = minimize_scalar(f, bounds=(-15, 15), method="bounded", options={"xatol": 1e-10})
    return r.fun / np.mean(tau + 1), np.exp(r.x)


def fig2():
    lam = {k: spectrum(k) for k in ("mice", "rice")}
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 52 * MM), gridspec_kw=dict(wspace=0.32))
    ax = axs[0]; L = lam["mice"]; L = L[L > 1e-6]
    cols = {0.5: TOL["yellow"], 1.0: DARKGREY, 2.0: TOL["blue"]}
    for g, c in cols.items():
        _, w = weights(L, g); ax.plot(L, w, color=c, lw=1.3, label=f"Bayes weights, γ = {g:g}")
        if g != 1.0:
            _, d = gap(L, g); ax.plot(L, L / (L + d), color=c, lw=0.9, ls=(0, (3, 2)))
    ax.set_xscale("log"); ax.set_xlabel("GRM eigenvalue λᵢ (mouse panel, mean 1)"); ax.set_ylabel("Shrinkage weight wᵢ")
    ax.set_ylim(-0.02, 1.02); value_grid(ax)
    h = [Line2D([], [], color=c, lw=1.3, label=f"γ = {g:g}" + (" (= rrBLUP)" if g == 1.0 else "")) for g, c in cols.items()] + \
        [Line2D([], [], color=DARKGREY, lw=0.9, ls=(0, (3, 2)), label="best ridge λ/(λ+δ*)")]
    ax.legend(handles=h, loc="upper left"); panel_label(ax, "a"); subtitle(ax, "Shrinkage weights on the mouse GRM")
    ax = axs[1]; gs = np.linspace(0.3, 3.0, 55)
    for k, c, lab in (("mice", TOL["purple"], "Mouse GRM"), ("rice", TOL["green"], "Rice GRM")):
        L = lam[k]; L = L[L > 1e-6]; G = np.array([gap(L, g)[0] for g in gs])
        ax.plot(gs, G * 1e3, color=c, lw=1.3, label=lab)
        A = (gap(L, 1.05)[0] + gap(L, 0.95)[0]) / (2 * 0.05 ** 2)
        near = gs[(gs > 0.6) & (gs < 1.4)]; ax.plot(near, A * (near - 1) ** 2 * 1e3, color=c, lw=0.8, ls=(0, (1, 1.5)))
    ax.axvline(1, color=DARKGREY, lw=0.6, ls=(0, (3, 2)))
    ax.set_xlabel("Power-law exponent γ"); ax.set_ylabel("Oracle gap over best ridge (R², ×10$^{-3}$)")
    value_grid(ax); ax.set_ylim(bottom=0)
    h = ax.get_legend_handles_labels()[0] + [Line2D([], [], color=DARKGREY, lw=0.8, ls=(0, (1, 1.5)), label="quadratic with the curvature at γ = 1")]
    ax.legend(handles=h, loc="upper center"); panel_label(ax, "b"); subtitle(ax, "Oracle gap over the best ridge")
    save(fig, "fig3_theory")


# ------------------------------------------------------------------ Figure 3: simulation regime map
def fig3():
    R = J("e1_regimes.json"); ns = sorted({r["n"] for r in R})
    scen = [("g0.5", "γ = 0.5"), ("g1", "γ = 1 (GBLUP model)"), ("g2", "γ = 2"), ("g3", "γ = 3"), ("spiked", "Population structure only")]
    meths = ["oracle", "safe_CV", "power_CV", "gammaBLUP_ML"]
    fig, axs = plt.subplots(2, 3, figsize=(DOUBLE, 92 * MM), gridspec_kw=dict(wspace=0.38, hspace=0.55))
    for i, (s, title) in enumerate(scen):
        ax = axs.flat[i]
        for m in meths:
            mu, lo, hi = [], [], []
            for n in ns:
                v = np.array([r["r2"][m] - r["r2"]["rrBLUP_CV"] for r in R if r["scn"] == s and r["n"] == n]) * 1e3
                se = v.std(ddof=1) / np.sqrt(len(v)); mu.append(v.mean()); lo.append(v.mean() - 1.96 * se); hi.append(v.mean() + 1.96 * se)
            ls = (0, (3, 2)) if m == "oracle" else "-"
            ax.plot(ns, mu, color=METHOD[m], ls=ls, marker="o" if m != "oracle" else None, ms=2.6, lw=1.1, zorder=3)
            ax.fill_between(ns, lo, hi, color=METHOD[m], alpha=0.26, lw=0, zorder=2)
        ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns]); ax.minorticks_off()
        zero_line(ax); value_grid(ax)
        subtitle(ax, title)
        if i % 3 == 0: ax.set_ylabel("ΔR² vs rrBLUP (×10$^{-3}$)")
        if i >= 3 or i == 2: ax.set_xlabel("Training size n")
        panel_label(ax, "abcde"[i], x=-0.24, y=1.06)
    axs.flat[5].axis("off")
    h = [Line2D([], [], color=METHOD[m], ls=(0, (3, 2)) if m == "oracle" else "-", marker=None if m == "oracle" else "o", ms=2.6, label=LABEL[m]) for m in meths]
    h.append(Patch(color=DARKGREY, alpha=0.32, label="95% CI of the mean"))
    axs.flat[5].legend(handles=h, loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=7)
    save(fig, "fig4_simulation")


# ------------------------------------------------------------------ Figure 4: test calibration
def fig4():
    R = J("e1_reml_test.json"); S = J("e13_reml_size.json"); ns = sorted({r["n"] for r in R})
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 56 * MM), gridspec_kw=dict(wspace=0.3, width_ratios=[1, 1.25]))
    ax = axs[0]
    sets = [("Simulation, γ = 1 (all n)", [r["p_chi2"] for r in R if r["scn"] == "g1"], DARKGREY),
            ("Rice GRM, exact null", [r["p_reml"] for r in S if r["data"] == "rice"], TOL["green"]),
            ("Mouse GRM + covariates, exact null", [r["p_reml"] for r in S if r["data"] == "mice"], TOL["purple"]),
            ("Mouse, fixed effects not projected out", [r["p_old"] for r in S if r["data"] == "mice"], TOL["red"])]
    m = min(len(p) for _, p, _ in sets); e = -np.log10((np.arange(1, m + 1) - 0.5) / m)
    k = np.arange(1, m + 1); lo = -np.log10(stats.beta.ppf(0.975, k, m - k + 1)); hi = -np.log10(stats.beta.ppf(0.025, k, m - k + 1))
    ax.fill_between(e, lo, hi, color="#000000", alpha=0.08, lw=0, label=f"95% band (n = {m})")
    for lab, p, c in sets:
        p = np.sort(np.clip(np.array(p, float), 1e-12, 1)); q = -np.log10((np.arange(1, len(p) + 1) - 0.5) / len(p))
        ax.plot(q, -np.log10(p), ".", color=c, ms=2.2, label=lab, alpha=0.9)
    lim = max(3.2, e.max() + 0.1); ax.plot([0, lim], [0, lim], color=DARKGREY, lw=0.6)
    ax.set_xlim(0, lim); ax.set_ylim(0, None); ax.set_xlabel("Expected −log$_{10}$ p (uniform)"); ax.set_ylabel("Observed −log$_{10}$ p")
    ax.legend(loc="upper left", fontsize=5.9, markerscale=2.5); value_grid(ax); panel_label(ax, "a"); subtitle(ax, "Size: p-values under γ = 1")
    ax = axs[1]
    curves = [("g0.5", "γ = 0.5", "#D55E00"), ("g0.75", "γ = 0.75", "#E69F00"), ("g1", "γ = 1 (size)", DARKGREY),
              ("g1.25", "γ = 1.25", "#9AD0F3"), ("g1.5", "γ = 1.5", "#56B4E9"), ("g2", "γ = 2", "#0072B2"),
              ("g3", "γ = 3", "#003D63"), ("spiked", "Structure only", "#CC79A7")]          # Okabe-Ito: warm below 1, blues above
    for s, lab, c in curves:
        y = [np.mean([r["p_chi2"] < .05 for r in R if r["scn"] == s and r["n"] == n]) for n in ns]
        ax.plot(ns, y, color=c, marker="o", ms=2.6, lw=1.1, ls="-" if s != "spiked" else (0, (3, 2)), label=lab)
    ax.axhline(0.05, color=DARKGREY, lw=0.6, ls=(0, (1, 1.5)), label="Nominal 5% level")
    ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns]); ax.minorticks_off()
    ax.set_ylim(0, 1.02); ax.set_xlabel("Training size n"); ax.set_ylabel("Rejection rate of γ = 1 (5% level)")
    value_grid(ax); ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=6.2); panel_label(ax, "b"); subtitle(ax, "Power by training size")
    save(fig, "fig5_test_calibration")


if __name__ == "__main__":
    which = sys.argv[1:] or ["2", "3", "4"]
    for w in which: {"2": fig2, "3": fig3, "4": fig4}[w]()
