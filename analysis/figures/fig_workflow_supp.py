"""Supplementary figures S1 (LOO ablation, E10), S2 (runtime, E8), S3 (Gibbs chains, E5).
The bootstrap stability of gamma-hat (planned as a separate supplementary figure) is shown in Figure 5.
usage: python fig_workflow_supp.py [S1 S2 S3]          (Figure 1: fig1_workflow.py)
"""
import os, sys, json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from figstyle import setup, save, value_grid, zero_line, panel_label, subtitle, METHOD, LABEL, TOL, DARKGREY, INK, SINGLE, DOUBLE, MM, RES

setup()


def J(fn): return json.load(open(os.path.join(RES, fn)))


# ------------------------------------------------------------------ S1: LOO ablation
def figS1():
    E = J("e10_ablation.json"); variants = [("original", "LOO-tuned,\nno safeguard"), ("bounded_only", "Leverage\nbound only"), ("guard_only", "Inner-CV\nguard only"), ("both", "Bound +\nguard")]
    series = [("wheat", "smooth", ("s", INK)), ("wheat", "grouped", ("s", "white")), ("rice", "smooth", ("o", INK)), ("rice", "grouped", ("o", "white"))]
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 60 * MM), gridspec_kw=dict(wspace=0.3))
    for j, (ds, fam, c) in enumerate(series):
        for i, (v, _) in enumerate(variants):
            r = np.array([x["r2"] for x in E if x["dataset"] == ds and x["family"] == fam and x["variant"] == v])
            se = r.std(ddof=1) / np.sqrt(len(r)); x = i + (j - 1.5) * 0.14
            axs[0].errorbar(x, r.mean(), yerr=1.96 * se, fmt=c[0], color=INK, mfc=c[1], mec=INK, ms=3.6, lw=0.8, mew=0.7)
            l = [x_["loo"] for x_ in E if x_["dataset"] == ds and x_["family"] == fam and x_["variant"] == v and x_["loo"] == x_["loo"]]
            if l: axs[1].plot(i + (j - 1.5) * 0.14, np.mean(l), c[0], mfc=c[1], mec=INK, mew=0.7, ms=3.6)
    for ax in axs:
        ax.set_xticks(range(4)); ax.set_xticklabels([v[1] for v in variants], fontsize=6.2); value_grid(ax)
    zero_line(axs[0]); axs[0].set_ylabel("Held-out R² (mean ± 95% CI over folds)"); panel_label(axs[0], "a"); subtitle(axs[0], "Held-out accuracy")
    axs[1].set_ylabel("In-sample LOO MSE of the selected profile"); axs[1].set_xlim(-0.5, 2.5); axs[1].set_xticks(range(3)); axs[1].set_xticklabels([v[1] for v in variants[:3]], fontsize=6.2)
    panel_label(axs[1], "b"); subtitle(axs[1], "In-sample LOO criterion")
    h = [Line2D([], [], marker=c[0], mfc=c[1], mec=INK, mew=0.7, lw=0, ms=3.6, label=f"{ds.capitalize()}, {fam} family") for ds, fam, c in series]
    axs[0].legend(handles=h, loc="lower right", fontsize=6)
    save(fig, "figS1_loo_ablation")


# ------------------------------------------------------------------ S2: runtime
def figS2():
    a = J("e8_scaling_run1.json")["results"]; b = J("e8_scaling_run2.json")["results"]
    fig, ax = plt.subplots(figsize=(SINGLE, 62 * MM))
    curves = [("eigh_s", "Eigendecomposition", "#000000"), ("rrBLUP_CV_s", "rrBLUP (tuned)", METHOD["rrBLUP"]),
              ("power_CV_s", "Power filter (tuned)", METHOD["power"]), ("safe_CV_s", "Safe selector (nested CV)", METHOD["safe"])]
    for k, lab, c in curves:
        for run, mk, fill in ((b, "o", True), (a, "s", False)):
            pts = [(r["n"], r[k]) for r in run if k in r]
            ax.plot(*zip(*pts), marker=mk, ms=3, color=c, mfc=c if fill else "white", lw=1.1 if fill else 0.6, ls="-" if fill else ":", label=lab if fill else None)
    ax.set_xscale("log"); ax.set_yscale("log"); ns = [r["n"] for r in b]; ax.set_xticks(ns); ax.set_xticklabels([f"{n:,}" for n in ns]); ax.minorticks_off()
    ax.set_xlabel("Number of individuals n"); ax.set_ylabel("Wall-clock time (s), 12-thread CPU"); value_grid(ax)
    h = ax.get_legend_handles_labels()[0] + [Line2D([], [], color=DARKGREY, marker="s", mfc="white", lw=0.6, ls=":", ms=3, label="repeat run")]
    ax.legend(handles=h, loc="upper left", fontsize=6)
    save(fig, "figS2_runtime")


# ------------------------------------------------------------------ S3: Gibbs chains
def figS3():
    models = ("BRR", "BayesC", "BayesR"); panels = ("rice", "wheat", "mice")
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 60 * MM), gridspec_kw=dict(wspace=0.3))
    ax = axs[0]
    for i, p in enumerate(panels):
        for j, m in enumerate(models):
            for tag, alpha, off in (("", 0.5, -0.13), ("_long", 1.0, 0.13)):
                z = np.array([abs(f["geweke"]) for T in J(f"e4_{p}_kfold_bayes{tag}.json") for f in T["info"][m] if isinstance(f, dict) and f.get("geweke") is not None])
                z = z[np.isfinite(z)]
                ax.bar(i * 4 + j + off, np.mean(z > 1.96), width=0.26, color=METHOD[m], alpha=alpha, lw=0)
    ax.axhline(0.05, color=DARKGREY, lw=0.6, ls=(0, (1, 1.5)))
    ax.set_xticks([i * 4 + 1 for i in range(3)]); ax.set_xticklabels(["Rice", "Wheat", "Mouse HS"])
    ax.set_ylabel("Share of chains with |Geweke z| > 1.96"); value_grid(ax); panel_label(ax, "a"); subtitle(ax, "Geweke convergence diagnostic")
    h = [Line2D([], [], color=METHOD[m], lw=5, label=m) for m in models] + [Line2D([], [], color=DARKGREY, lw=5, alpha=0.5, label="short chain (3,000)"),
         Line2D([], [], color=DARKGREY, lw=5, label="long chain (12,000–20,000)"), Line2D([], [], color=DARKGREY, lw=0.6, ls=(0, (1, 1.5)), label="5% expected")]
    ax.legend(handles=h, loc="upper right", fontsize=5.8, ncol=2)
    ax = axs[1]
    for p, mk in zip(panels, ("o", "s", "^")):
        for m in models:
            a = {t["trait"]: t for t in J(f"e4_{p}_kfold_bayes.json")}; b = {t["trait"]: t for t in J(f"e4_{p}_kfold_bayes_long.json")}
            for t in b:
                L = len(b[t]["r2"][m]); ax.plot(np.nanmean(np.array(a[t]["r2"][m])[:L]), np.nanmean(b[t]["r2"][m]), mk, color=METHOD[m], ms=3, mfc="white", mew=0.8)
    lim = ax.get_xlim(); lo, hi = min(lim[0], ax.get_ylim()[0]), max(lim[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], color=DARKGREY, lw=0.6); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Held-out R², short chain (same folds)"); ax.set_ylabel("Held-out R², long chain"); value_grid(ax, "both"); panel_label(ax, "b"); subtitle(ax, "Accuracy, short vs long chain")
    h = [Line2D([], [], color=DARKGREY, marker=mk, mfc="white", lw=0, ms=3, label=n) for n, mk in (("Rice", "o"), ("Wheat", "s"), ("Mouse HS", "^"))]
    ax.legend(handles=h, loc="upper left", fontsize=6)
    save(fig, "figS3_gibbs")


if __name__ == "__main__":
    for w in (sys.argv[1:] or ["S1", "S2", "S3"]): {"S1": figS1, "S2": figS2, "S3": figS3}[w]()
