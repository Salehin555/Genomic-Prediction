"""Figures 6-9 (numbering of Planning/12): real data on five panels (61 traits).
  5  gamma-hat with 95% REML bootstrap CI per trait, coloured by the REML test decision
  6  per-trait Delta R2 vs rrBLUP (power filter, safe selector, KRR, BayesR) with corrected-t 95% CI; filled = BH q < 0.05
  7  why GBLUP is hard to beat: oracle in-sample gap per trait vs the observed held-out gain of the power filter
  8  mouse panel: k-fold vs cage-out gains (power filter, KRR, epistatic kernel) and BayesR on the lipid traits
usage: python fig_real.py [5 6 7 8]
"""
import os, sys, json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from figstyle import setup, save, value_grid, zero_line, panel_label, subtitle, METHOD, LABEL, TOL, DARKGREY, INK, DECISION, SINGLE, DOUBLE, MM, RES
from analyze_all import bh

setup()
PANELS = ("rice", "wheat", "mice", "rdp1", "sorghum")
PNAME = {"rice": "Rice", "wheat": "Wheat", "mice": "Mouse HS", "rdp1": "Rice RDP1", "sorghum": "Sorghum"}


def J(fn): return json.load(open(os.path.join(RES, fn)))
def by_trait(fn): return {t["trait"]: t for t in J(fn)}


def rejected():
    out = {(r["dataset"], r["trait"]): r["p_boot"] < .05 for r in J("e13_reml_real.json") if r["dataset"] in ("rice", "wheat", "mice")}
    for p in ("rdp1", "sorghum"):
        for r in J(f"prereg_predictions_{p}.json")["traits"]: out[(p, r["trait"])] = r["p_boot"] < .05
    return out


def order():
    """Traits in panel order; within a panel by gamma-hat."""
    E = {(x["dataset"], x["trait"]): x for x in J("e7_reml_boot.json")}
    return [k for p in PANELS for k in sorted([k for k in E if k[0] == p], key=lambda k: E[k]["gamma"])], E


def delta_ci(b, v, n):
    """Mean fold difference and Nadeau-Bengio corrected 95% CI (n_tr = 0.8 n, n_te = 0.2 n)."""
    L = min(len(b), len(v)); d = np.asarray(v[:L], float) - np.asarray(b[:L], float); d = d[~np.isnan(d)]; Jn = len(d)
    se = np.sqrt((1 / Jn + 0.25) * d.var(ddof=1)); t = stats.t.ppf(0.975, Jn - 1)
    p = 2 * stats.t.sf(abs(d.mean() / se), Jn - 1) if se > 0 else 1.0
    return d.mean(), d.mean() - t * se, d.mean() + t * se, p


def panel_axis(ax, keys, ymin_text, shade_top=1.0):
    xs = np.arange(len(keys)); bounds = []
    for p in PANELS:
        idx = [i for i, k in enumerate(keys) if k[0] == p]
        if idx: bounds.append((p, idx[0], idx[-1]))
    for j, (p, a, b) in enumerate(bounds):
        if j > 0: ax.axvline(a - 0.5, color="#BDBDBD", lw=0.5, zorder=0)              # thin separator between panels
        ax.text((a + b) / 2, ymin_text, PNAME[p], ha="center", va="top", fontsize=6.3, transform=ax.get_xaxis_transform())
    ax.set_xlim(-0.8, len(keys) - 0.2); ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    return xs


# ------------------------------------------------------------------ Figure 5
def fig5():
    keys, E = order(); rej = rejected()
    fig, ax = plt.subplots(figsize=(DOUBLE, 62 * MM))
    xs = panel_axis(ax, keys, -0.03, shade_top=2 / 3.2)
    ax.axhspan(2, 3.2, color="#EEF3FA", lw=0, zorder=0); ax.axhspan(0, 0.5, color="#EEF3FA", lw=0, zorder=0)
    ax.text(len(keys) - 0.6, 2.95, "γ ≥ 2 and γ ≤ 0.5 (shaded): the simulation shows a gain", ha="right", va="top", fontsize=6.3, color=DARKGREY, style="italic")
    ax.axhline(1, color=DARKGREY, lw=0.7, ls=(0, (3, 2)), zorder=1)
    for x, k in zip(xs, keys):
        g, (lo, hi) = E[k]["gamma"], E[k]["ci"]; c = DECISION["reject"] if rej[k] else DECISION["keep"]
        top = min(hi, 3.0)
        ax.plot([x, x], [lo, top], color=c, lw=0.9, alpha=0.95, zorder=2)
        if hi > 3.0: ax.plot(x, 3.0, marker="^", color=c, ms=3, zorder=3)
        ax.plot(x, g, "o", color=c, ms=3.2, mec="white", mew=0.4, zorder=4)
    ax.set_ylim(0, 3.2); ax.set_ylabel(r"Estimated $\hat{\gamma}$ (95% bootstrap CI)"); value_grid(ax)
    h = [Line2D([], [], color=DECISION["reject"], marker="o", lw=0.9, ms=3.2, label="REML test rejects γ = 1 (p < 0.05)"),
         Line2D([], [], color=DECISION["keep"], marker="o", lw=0.9, ms=3.2, label="not rejected"),
         Line2D([], [], color=DARKGREY, marker="^", lw=0, ms=3, label="CI continues above 3"),
         Line2D([], [], color=DARKGREY, lw=0.7, ls=(0, (3, 2)), label="γ = 1 (GBLUP model)")]
    ax.legend(handles=h, loc="upper left", ncol=4, bbox_to_anchor=(0.0, 1.10))
    save(fig, "fig6_gamma_real")


# ------------------------------------------------------------------ Figure 6
def gains_all():
    """{method: {(panel, trait): (mean, lo, hi, q)}} with BH within panel and method."""
    out = {m: {} for m in ("power_CV", "safe_CV", "KRR_RBF", "BayesR")}
    for p in PANELS:
        S = by_trait(f"e4_{p}_kfold_spectral.json"); K = by_trait(f"e4_{p}_kfold_kernel.json")
        B = by_trait(f"e4_{p}_kfold_bayes_long.json") if os.path.exists(os.path.join(RES, f"e4_{p}_kfold_bayes_long.json")) else {}
        for m in out:
            src = S if m in ("power_CV", "safe_CV") else K if m == "KRR_RBF" else B
            rows = {t: delta_ci(S[t]["r2"]["rrBLUP_CV"], src[t]["r2"][m], S[t]["n"]) for t in src}
            if not rows: continue
            q = bh([r[3] for r in rows.values()])
            for (t, r), qq in zip(rows.items(), q): out[m][(p, t)] = (r[0], r[1], r[2], qq)
    return out


def fig6():
    keys, _ = order(); G = gains_all()
    meths = [("power_CV", "Power filter"), ("safe_CV", "Safe selector"), ("KRR_RBF", "Kernel ridge regression (RBF)"), ("BayesR", "BayesR (long chains)")]
    fig, axs = plt.subplots(4, 1, figsize=(DOUBLE, 130 * MM), gridspec_kw=dict(hspace=0.36))
    for i, (ax, (m, title)) in enumerate(zip(axs, meths)):
        xs = panel_axis(ax, keys, -0.04 if i == 3 else -0.02)
        if i < 3:
            for t in ax.texts: t.set_visible(False)
        for x, k in zip(xs, keys):
            if k not in G[m]: continue
            mu, lo, hi, q = G[m][k]; c = METHOD[m]
            ax.plot([x, x], [lo, hi], color=c, lw=0.8, alpha=0.85, zorder=2)
            ax.plot(x, mu, "o", ms=3.0, color=c, mfc=c if q < .05 else "white", mew=0.8, zorder=3)
        if m == "BayesR":
            a = [i for i, k in enumerate(keys) if k[0] in ("rdp1", "sorghum")]
            ax.text((a[0] + a[-1]) / 2, 0.8, "not run on these panels", ha="center", va="center", fontsize=6.3, color=DARKGREY, transform=ax.get_xaxis_transform())
        zero_line(ax); value_grid(ax); ax.set_ylabel("ΔR² vs rrBLUP", fontsize=6.5)
        ax.set_title(title, loc="left", fontsize=7, pad=4, color=METHOD[m], fontweight="bold")
        panel_label(ax, "abcd"[i], x=-0.075, y=0.98)
    h = [Line2D([], [], color=DARKGREY, marker="o", lw=0, ms=3.2, label="BH q < 0.05"),
         Line2D([], [], color=DARKGREY, marker="o", mfc="white", lw=0, ms=3.2, label="not significant"),
         Line2D([], [], color=DARKGREY, lw=0.8, label="95% CI (corrected resampled t)")]
    axs[0].legend(handles=h, loc="lower right", ncol=3, bbox_to_anchor=(1.0, 1.0))
    save(fig, "fig7_gains_real")


# ------------------------------------------------------------------ Figure 7
def fig7():
    gap = {(x["dataset"], x["trait"]): x for x in J("gap_real.json")}; G = gains_all()["power_CV"]
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 62 * MM), gridspec_kw=dict(wspace=0.3, width_ratios=[1, 1.35]))
    floor = 1e-5
    ax = axs[0]; v = np.sort(np.maximum([g["gap_r2"] for g in gap.values()], floor))
    ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color="#000000", lw=1.2)
    for thr, lab in ((0.001, "0.001"), (0.005, "0.005")):
        share = np.mean(np.array([g["gap_r2"] for g in gap.values()]) < thr)
        ax.axvline(thr, color=DARKGREY, lw=0.6, ls=(0, (3, 2)))
        ax.text(thr / 1.2 if thr < .003 else thr * 1.2, 0.93 if thr < .003 else 0.30, f"{share:.0%} of traits\nbelow {lab}", fontsize=6, color=DARKGREY,
                ha="right" if thr < .003 else "left", va="top", bbox=dict(fc="white", ec="none", pad=0.6))
    ax.set_xscale("log"); ax.set_xlim(floor * 0.8, 0.05); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Oracle gap: largest in-sample R² any\nspectral filter could add over the best ridge"); ax.set_ylabel("Cumulative share of the 61 traits")
    value_grid(ax); panel_label(ax, "a"); subtitle(ax, "Oracle gap over 61 traits")
    ax = axs[1]; shp = {"rice": "o", "wheat": "s", "mice": "^", "rdp1": "D", "sorghum": "v"}      # datasets by shape, as in Figure 2
    for k, g in gap.items():
        if k not in G: continue
        mu, lo, hi, q = G[k]; x = max(g["gap_r2"], floor)
        ax.plot([x, x], [lo, hi], color="#BDBDBD", lw=0.6, zorder=2)
        m = shp[k[0]]; ax.plot(x, mu, m, ms=3.6 if m != "D" else 3.0, mfc="white", mec=INK, mew=0.6, zorder=3)
    xx = np.logspace(np.log10(floor), np.log10(0.03), 50); ax.plot(xx, xx, color="#000000", lw=0.7, ls=(0, (3, 2)))
    ax.text(0.03, 0.031, "observed gain = gap", fontsize=6, ha="right", va="bottom")
    zero_line(ax); ax.set_xscale("log"); ax.set_xlim(floor * 0.8, 0.04)
    ax.set_xlabel("Oracle gap (R², log scale; 0 shown at 10$^{-5}$)"); ax.set_ylabel("Observed held-out ΔR² of the power filter")
    value_grid(ax)
    h = [Line2D([], [], marker=m, mfc="white", mec=INK, mew=0.6, lw=0, ms=3.6 if m != "D" else 3.0, label=PNAME[p]) for p, m in shp.items()]
    ax.legend(handles=h, loc="lower left", ncol=2); panel_label(ax, "b"); subtitle(ax, "Observed gain against the oracle gap")
    save(fig, "fig8_oracle_gap")


# ------------------------------------------------------------------ Figure 8
def fold_r2(rec, m):
    y = np.array(rec["y"]); p = np.array(rec["p"][m]); return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def mouse_delta(scheme, m):
    S = by_trait(f"e4_mice_{scheme}_spectral.json"); out = {}
    if m == "epi":
        E = by_trait("e11_epi_mice.json" if scheme == "kfold" else "e11_epi_mice_cage.json")
        for t in E:
            base = {(r["seed"], r["fold"]): fold_r2(r, "rrBLUP_CV") for r in S[t]["preds"]}
            d = [f["r2"] - base[(f["seed"], f["fold"])] for f in E[t]["folds"] if (f["seed"], f["fold"]) in base]
            out[t] = float(np.mean(d))
        return out
    src = S if m == "power_CV" else by_trait(f"e4_mice_{scheme}_kernel.json")
    for t in src:
        b = np.array(S[t]["r2"]["rrBLUP_CV"]); v = np.array(src[t]["r2"][m]); L = min(len(b), len(v)); out[t] = float(np.nanmean(v[:L] - b[:L]))
    return out


def fig8():
    rej = rejected()
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 64 * MM), gridspec_kw=dict(wspace=0.32, width_ratios=[1.15, 1]))
    ax = axs[0]
    for j, m in enumerate(("power_CV", "KRR_RBF", "epi")):
        a, b = mouse_delta("kfold", m), mouse_delta("cage", m); ts = [t for t in a if t in b]
        x0, x1 = 3 * j, 3 * j + 1.4
        for t in ts:
            c = DECISION["reject"] if rej[("mice", t)] else DECISION["keep"]
            ax.plot([x0, x1], [a[t], b[t]], color=c, lw=0.55, alpha=0.65, zorder=2)
        for x, d in ((x0, a), (x1, b)):
            v = np.array([d[t] for t in ts]); se = v.std(ddof=1) / np.sqrt(len(v)); tq = stats.t.ppf(0.975, len(v) - 1)
            ax.errorbar(x, v.mean(), yerr=tq * se, fmt="o", color=METHOD[m], ms=4.2, lw=1.4, zorder=4)
        ax.text((x0 + x1) / 2, -0.12, LABEL[m], ha="center", va="top", fontsize=6.5, color=METHOD[m], fontweight="bold", transform=ax.get_xaxis_transform())
    ax.set_xticks([x for j in range(3) for x in (3 * j, 3 * j + 1.4)]); ax.set_xticklabels(["k-fold", "cage-out"] * 3, fontsize=6.2)
    zero_line(ax); value_grid(ax); ax.set_ylabel("ΔR² vs rrBLUP (19 mouse traits)"); ax.set_xlim(-0.6, 8.0)
    h = [Line2D([], [], color=DECISION["reject"], lw=0.8, label="trait: test rejects γ = 1"), Line2D([], [], color=DECISION["keep"], lw=0.8, label="trait: not rejected"),
         Line2D([], [], color=DARKGREY, marker="o", lw=1.4, ms=4, label="mean ± 95% CI")]
    ax.legend(handles=h, loc="lower left", fontsize=6); panel_label(ax, "a", x=-0.13, y=1.02); subtitle(ax, "k-fold vs cage-out cross-validation")
    ax = axs[1]; S = by_trait("e4_mice_kfold_spectral.json"); B = by_trait("e4_mice_kfold_bayes_long.json")
    rows = {t: delta_ci(np.array(S[t]["r2"]["rrBLUP_CV"])[:5], B[t]["r2"]["BayesR"], S[t]["n"]) for t in B}
    q = dict(zip(rows, bh([r[3] for r in rows.values()])))
    ts = sorted(rows, key=lambda t: rows[t][0])
    for y, t in enumerate(ts):
        mu, lo, hi, _ = rows[t]; c = METHOD["BayesR"]
        ax.plot([lo, hi], [y, y], color=c, lw=0.8, alpha=0.85); ax.plot(mu, y, "o", color=c, mfc=c if q[t] < .05 else "white", ms=3.2, mew=0.8)
    ax.set_yticks(range(len(ts))); ax.set_yticklabels([t.split(".", 1)[1].replace("Tot.", "Total ") for t in ts], fontsize=5.6)
    for lab in ax.get_yticklabels():
        if rej[("mice", "Biochem." + lab.get_text().replace("Total ", "Tot.")) if ("mice", "Biochem." + lab.get_text().replace("Total ", "Tot.")) in rej else ("mice", "Obesity." + lab.get_text())]: lab.set_color(DECISION["reject"])
    zero_line(ax, axis="x"); value_grid(ax, axis="x"); ax.set_xlabel("ΔR² of BayesR vs rrBLUP (long chains, 5 folds)")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    h = [Line2D([], [], color=METHOD["BayesR"], marker="o", lw=0, ms=3.2, label="BH q < 0.05"), Line2D([], [], color=METHOD["BayesR"], marker="o", mfc="white", lw=0, ms=3.2, label="not significant"),
         Line2D([], [], color=DECISION["reject"], marker="s", lw=0, ms=3.0, label="orange name: test rejects γ = 1")]
    ax.legend(handles=h, loc="lower right", fontsize=6); panel_label(ax, "b", x=-0.36, y=1.02); subtitle(ax, "BayesR vs rrBLUP by trait")
    save(fig, "fig9_mouse")


if __name__ == "__main__":
    for w in (sys.argv[1:] or ["5", "6", "7", "8"]): {"5": fig5, "6": fig6, "7": fig7, "8": fig8}[w]()
