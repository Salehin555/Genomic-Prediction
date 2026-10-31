"""Figure 1: methodological overview in the style of method figures in ML/statistics papers (math objects as nodes, thin
arrows labelled with the operation, flat colours, real-data thumbnails). Drawn in millimetres at 174 x 46 mm; thumbnails are rasterised at 600 dpi.

Thumbnails are real mouse data: genotype block, phenotype (Biochem.Sodium), a block of the GRM, the leading eigenvectors and
eigenvalues of a 300-animal subset (for display only); the scatter is the REML-basis fit for Sodium on all animals.
usage: python fig_method_flow.py       writes out/fig1_method_flow.pdf/.png
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from figstyle import setup, save, TOL, INK, DARKGREY, MM

setup()
plt.rcParams.update({"mathtext.fontset": "cm", "font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"], "axes.unicode_minus": False,
                     "axes.formatter.use_mathtext": True, "mathtext.default": "it"})
W, H = 174.0, 46.0
BLUES = LinearSegmentedColormap.from_list("b", ["#F4F8FC", "#BBD1EA", "#6F9BC9", "#2E5E99"])
DIV = LinearSegmentedColormap.from_list("d", ["#2E5E99", "#BBD1EA", "#F7F7F7", "#F6C09A", "#C8551B"])
ORANGE = "#E07A2F"


def arrow(ax, a, b, label=None, dy=2.2, ls="-", lw=0.7, color=INK, fs=6.6):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7, lw=lw, color=color, ls=ls, shrinkA=0, shrinkB=0, zorder=4))
    if label:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + dy, label, ha="center", va="bottom", fontsize=fs, color=color, linespacing=1.15)


def img(ax, A, x0, y0, w, h, cmap, vmin=None, vmax=None):
    ax.imshow(A, extent=(x0, x0 + w, y0, y0 + h), cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", aspect="auto", zorder=2, rasterized=True)
    ax.add_patch(Rectangle((x0, y0), w, h, fc="none", ec=INK, lw=0.5, zorder=3))


def brace(ax, x0, x1, y, label, below=True):
    s = -1 if below else 1
    ax.plot([x0, x0, x1, x1], [y - s * 0.6, y, y, y - s * 0.6], color=DARKGREY, lw=0.45)
    ax.text((x0 + x1) / 2, y + s * 0.6, label, ha="center", va="top" if below else "bottom", fontsize=6.2, color=DARKGREY)


def data():
    from e13_reml_diag import load, reml_basis
    from spectral_filters import ml_fit
    D = load("mice"); T = D["traits"]["Biochem.Sodium"]; rows = T["rows"]; G = D["G"][np.ix_(rows, rows)]
    lam, A = reml_basis(G, T["C"]); z = A @ T["y"]; z2 = (z / z.std()) ** 2
    c, s, g, _ = ml_fit(lam, z2); c0, s0, _, _ = ml_fit(lam, z2, gamma=1.0)
    rng = np.random.default_rng(0); sub = np.sort(rng.choice(len(rows), 300, replace=False))
    Zm = D["Z"][rows][sub[:36]][:, rng.choice(D["Z"].shape[1], 60, replace=False)]
    Gs = G[np.ix_(sub, sub)]; ev, U = np.linalg.eigh(Gs - Gs.mean(0) - Gs.mean(1)[:, None] + Gs.mean()); ev, U = ev[::-1], U[:, ::-1]
    return dict(lam=lam, z2=z2, fit=(c, s, g), fit1=(c0, s0), Zm=Zm, y=T["y"][sub[:36]], Gb=Gs[:36, :36], U=U[:36, :8], ev=ev[:8])


def main():
    d = data()
    fig = plt.figure(figsize=(W * MM, H * MM)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    yc = 25.0                                                          # centre line of the pipeline
    T = dict(ha="center", va="bottom", fontsize=7.5)

    # ---- (a) data: genotypes M, phenotype y, fixed effects X
    img(ax, d["Zm"], 3.0, yc - 8, 12, 16, BLUES, -2, 2)
    img(ax, d["y"][:, None], 16.2, yc - 8, 1.6, 16, BLUES)
    Xb = np.c_[np.ones(36), (np.arange(36) % 2), (np.arange(36) // 12 == 1)].astype(float)
    img(ax, Xb, 19.0, yc - 8, 3.6, 16, LinearSegmentedColormap.from_list("x", ["#FBF6E3", "#D9B44A"]))
    for x, t in ((9.0, r"$Z$"), (17.0, r"$y$"), (20.8, r"$C$")): ax.text(x, yc + 8.8, t, **T)
    brace(ax, 3.0, 15.0, yc - 9.2, r"$p$ markers"); ax.text(2.0, yc, r"$n$", ha="right", va="center", fontsize=6.6, color=DARKGREY)

    arrow(ax, (24.5, yc), (36.5, yc), r"$G=ZZ^{\top}\!/p$" "\n" r"$Q\perp\mathrm{col}(C)$", dy=1.4)

    # ---- (b) spectral basis  Q'GQ = U Lambda U'
    img(ax, d["Gb"], 38.0, yc - 6.5, 13, 13, BLUES); ax.text(44.5, yc + 7.4, r"$Q^{\top}GQ$", **T)
    ax.text(53.0, yc, r"$=$", ha="center", va="center", fontsize=8)
    Uv = d["U"] / np.abs(d["U"]).max()
    img(ax, Uv, 55.0, yc - 6.5, 5.0, 13, DIV, -1, 1); ax.text(57.5, yc + 7.4, r"$U$", **T)
    L = np.zeros((8, 8)); np.fill_diagonal(L, np.log(d["ev"] / d["ev"].min()) + 0.3)
    img(ax, L, 61.5, yc - 2.5, 5.0, 5.0, LinearSegmentedColormap.from_list("o", ["#FFFFFF", "#F9D3B4", ORANGE, "#A3480F"]))
    ax.text(64.0, yc + 3.4, r"$\Lambda$", **T)
    img(ax, Uv.T, 68.0, yc - 2.5, 13, 5.0, DIV, -1, 1); ax.text(74.5, yc + 3.4, r"$U^{\top}$", **T)
    ax.text(71.5, yc - 4.6, r"$\lambda_1\geq\lambda_2\geq\cdots$", ha="center", va="top", fontsize=6.4, color=DARKGREY)

    arrow(ax, (83.0, yc), (95.0, yc), r"$z=U^{\top}Q^{\top}y$", dy=1.4)

    # ---- (c) power-law fit (real data, Sodium)
    a = fig.add_axes([100.5 / W, (yc - 8.5) / H, 21 / W, 17 / H]); a.patch.set_alpha(0)
    lam, z2 = d["lam"], d["z2"]; (c, s, g), (c0, s0) = d["fit"], d["fit1"]
    e = np.quantile(lam, np.linspace(0, 1, 13)); mid, mz = [], []
    for lo, hi in zip(e[:-1], e[1:]):
        k = (lam >= lo) & (lam <= hi); mid.append(np.exp(np.mean(np.log(lam[k])))); mz.append(z2[k].mean())
    xx = np.logspace(np.log10(min(mid)), np.log10(max(mid)), 60)
    a.plot(xx, c0 * xx + s0, color=DARKGREY, lw=0.8, ls=(0, (3, 1.6)), label=r"$\gamma=1$")
    a.plot(xx, c * xx ** g + s, color=TOL["blue"], lw=1.1, label=r"$\hat\gamma=%.2f$" % g)
    a.plot(mid, mz, "o", ms=2.0, color=INK, zorder=3)
    a.set_xscale("log"); a.set_yscale("log"); a.minorticks_off(); a.set_xticks([]); a.set_yticks([])
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): a.spines[sp].set_linewidth(0.5)
    a.set_xlabel(r"$\lambda_i$", fontsize=7, labelpad=1); a.set_ylabel(r"$z_i^2$", fontsize=7, labelpad=1)
    a.set_ylim(min(mz) * 0.8, max(mz) * 2.6)                              # headroom so the key sits clear of the data
    a.legend(loc="upper left", fontsize=6, handlelength=1.4, borderaxespad=0.1, frameon=False)
    ax.text(111.0, yc + 9.4, r"$z_i\sim\mathcal{N}(0,\;c\lambda_i^{\gamma}+s)$", **T)

    arrow(ax, (124.5, yc), (131.0, yc), "LRT", dy=1.4)
    xn = 136.5; ax.add_patch(Circle((xn, yc), 5.0, fc="#EDEDED", ec=INK, lw=0.6, zorder=3))
    ax.text(xn, yc + 0.1, r"$H_0$" "\n" r"$\gamma=1$", ha="center", va="center", fontsize=6.8, linespacing=1.05, zorder=4)

    # ---- (d) outcome: three shrinkage rules
    outs = [(yc + 12.0, r"$\hat\gamma>1$", r"$w_i=\dfrac{\lambda_i^{a}}{\lambda_i^{a}+\delta}$", TOL["green"], "if held-out selection agrees"),
            (yc, r"accept", r"$w_i=\dfrac{\lambda_i}{\lambda_i+\delta}$", INK, "rrBLUP"),
            (yc - 12.0, r"$\hat\gamma<1$", r"$w_i=\dfrac{\lambda_i^{a}}{\lambda_i^{a}+\delta}$", TOL["red"], "if held-out selection agrees")]
    for yo, lab, f, col, note in outs:
        y0 = yc + np.sign(yo - yc) * 2.6
        arrow(ax, (xn + 5.0 if yo == yc else xn + 3.6, y0), (148.0, yo), color=col)
        if yo == yc: ax.text(145.0, yc + 0.9, lab, ha="center", va="bottom", fontsize=6.0, color=col)
        else: ax.text(141.0 + 1.0, (y0 + yo) / 2, lab, ha="right", va="center", fontsize=6.4, color=col)
        ax.text(149.0, yo + 0.4, f, ha="left", va="center", fontsize=7.2, color=col)
        ax.text(149.0, yo - 4.4, note, ha="left", va="top", fontsize=5.7, color=DARKGREY)

    # ---- oracle gap from the fit
    arrow(ax, (111.0, yc - 12.6), (111.0, 6.6), ls=(0, (2.2, 1.4)), color=DARKGREY)
    ax.text(111.0, 5.2, r"$\Delta(\hat\gamma)$: largest in-sample gain of any $w$ over rrBLUP", ha="center", va="top", fontsize=6.4)

    for x, t in ((12.0, "(a) data"), (59.5, "(b) spectral basis"), (111.0, "(c) power-law fit"), (154.0, "(d) test and rule")):
        ax.text(x, H - 1.2, t, ha="center", va="top", fontsize=6.8, color=DARKGREY, style="italic")
    save(fig, "fig1_method_flow")


if __name__ == "__main__":
    main()
