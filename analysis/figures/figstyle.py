"""Shared style for every revision figure (Planning/08 section 5).

Vector PDF saved with dpi=600 (raster content, if any, at 600 dpi), TrueType fonts embedded (pdf.fonttype 42),
a 600-dpi PNG companion, drawn at Springer print size (84 mm single column, 174 mm double column),
Okabe-Ito palette (chosen by the authors) with one fixed colour per method, no top/right spines, no grid lines.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
MM = 1 / 25.4
SINGLE, DOUBLE = 84 * MM, 174 * MM

import colorsys


def sat(hexcol, f=1.15):
    """Same hue and lightness, saturation multiplied by f (authors 2026-10-29: more saturated but professional; 1.3 was toned to 1.15)."""
    r, g, b = (int(hexcol[i:i + 2], 16) / 255 for i in (1, 3, 5))
    h, l, s = colorsys.rgb_to_hls(r, g, b); r, g, b = colorsys.hls_to_rgb(h, l, min(1.0, s * f))
    return "#%02X%02X%02X" % tuple(round(v * 255) for v in (r, g, b))


# Palettes (env FIGPAL selects one; FIGOUT redirects the output folder, used to compare palettes side by side).
# Keys are colour roles: blue/red/green/yellow/cyan/purple/grey for general use; the method and decision colours follow.
PALETTES = {
    "tol": dict(blue=sat("#4477AA"), red=sat("#EE6677"), green=sat("#228833"), yellow=sat("#CCBB44"), cyan=sat("#66CCEE"),
                purple=sat("#AA3377"), grey="#BBBBBB", bayesr=sat("#009988"), brr=sat("#66CCEE"), bayesc=sat("#994455"), reject=sat("#EE7733")),
    "okabe": dict(blue="#0072B2", red="#D55E00", green="#009E73", yellow="#E69F00", cyan="#56B4E9", purple="#CC79A7", grey="#BBBBBB",
                  bayesr="#56B4E9", brr="#009E73", bayesc="#D55E00", reject="#E69F00", epi="#6A3D9A"),
    "npg": dict(blue="#3C5488", red="#E64B35", green="#00A087", yellow="#F39B7F", cyan="#4DBBD5", purple="#8491B4", grey="#B09C85",
                bayesr="#4DBBD5", brr="#91D1C2", bayesc="#7E6148", reject="#F39B7F"),
    "tableau": dict(blue="#4E79A7", red="#E15759", green="#59A14F", yellow="#EDC948", cyan="#76B7B2", purple="#B07AA1", grey="#BAB0AC",
                    bayesr="#76B7B2", brr="#FF9DA7", bayesc="#9C755F", reject="#F28E2B"),
}
PAL = PALETTES[os.environ.get("FIGPAL", "okabe")]                # authors chose Okabe-Ito (2026-10-29)
if os.environ.get("FIGOUT"): OUT = os.environ["FIGOUT"]
TOL = {k: PAL[k] for k in ("blue", "red", "green", "yellow", "cyan", "purple", "grey")}
INK = "#222222"; DARKGREY = "#6E6E6E"

# one colour per method, used in every figure
METHOD = {
    "rrBLUP": DARKGREY, "rrBLUP_CV": DARKGREY,
    "power": TOL["blue"], "power_CV": TOL["blue"],
    "safe": TOL["green"], "safe_CV": TOL["green"],
    "plugin": TOL["yellow"], "gammaBLUP_ML": TOL["yellow"],
    "oracle": "#000000",
    "KRR": TOL["purple"], "KRR_RBF": TOL["purple"],
    "BayesR": PAL["bayesr"], "BRR": PAL["brr"], "BayesC": PAL["bayesc"],
    "epi": PAL.get("epi", TOL["cyan"]),                                       # kernels in purples: KRR light, epistatic dark
}
LABEL = {"rrBLUP_CV": "rrBLUP", "power_CV": "Power filter", "safe_CV": "Safe selector", "gammaBLUP_ML": "Plug-in γ-BLUP",
         "oracle": "Oracle", "KRR_RBF": "KRR (RBF)", "BayesR": "BayesR", "BRR": "BRR", "BayesC": "BayesC", "epi": "Epistatic kernel"}
DECISION = {"reject": PAL["reject"], "keep": TOL["blue"]}
PANEL = {"rice": "Rice (n = 191)", "wheat": "Wheat (n = 599)", "mice": "Mouse HS (n ≤ 1,814)", "rdp1": "Rice RDP1 (n ≤ 413)", "sorghum": "Sorghum (n = 451)"}


def setup():
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "legend.frameon": False, "legend.handlelength": 1.6, "legend.borderaxespad": 0.2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "axes.edgecolor": INK,
        "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.grid": False, "grid.color": "#E6E6E6", "grid.linewidth": 0.5,
        "lines.linewidth": 1.1, "lines.markersize": 3.5, "errorbar.capsize": 0,
        "savefig.dpi": 600, "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "mathtext.default": "regular", "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold", "mathtext.sf": "Arial",
    })


def subtitle(ax, text):
    """Subplot title, used only in multi-panel figures (author rule 2026-10-29)."""
    ax.set_title(text, loc="left", fontsize=7, pad=4, color=INK)


def value_grid(ax, axis="y"):
    """Grid lines removed at the authors' request (2026-10-29); kept as a no-op so the figure scripts stay unchanged."""
    ax.grid(False)


def zero_line(ax, axis="y", value=0.0):
    (ax.axhline if axis == "y" else ax.axvline)(value, color=DARKGREY, lw=0.6, ls=(0, (3, 2)), zorder=1)


def panel_label(ax, letter, x=-0.16, y=1.04):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")


def save(fig, name):
    """Save at exact print width: the figure is widened until its tight bounding box is 174 mm (or 84 mm) wide."""
    os.makedirs(OUT, exist_ok=True)
    target = DOUBLE if fig.get_figwidth() > 120 * MM else SINGLE
    for _ in range(6):
        fig.canvas.draw(); w = fig.get_tightbbox(fig.canvas.get_renderer()).width
        if abs(w - target) < 0.2 * MM: break
        fig.set_size_inches(fig.get_figwidth() + (target - w), fig.get_figheight())
    fig.savefig(os.path.join(OUT, name + ".pdf"), dpi=600)
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=600)
    plt.close(fig)
    print("saved", name)
