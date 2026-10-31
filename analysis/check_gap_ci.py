"""Uncertainty of the oracle gap: for every trait, the gap of check_gap_real.py evaluated at
the two ends of the 95% bootstrap interval of gamma-hat (e7_reml_boot.json), with c and s2 refitted by ML at that
fixed gamma. The larger of the two is a plausible upper value of the gap. writes results/gap_ci.json
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import json, numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
PANELS = ("rice", "wheat", "mice", "rdp1", "sorghum")


def gap_at(lam, z2, g):
    from spectral_filters import ml_fit
    c, s2, _, _ = ml_fit(lam, z2, gamma=float(g)); tau2 = c * lam ** g; w = tau2 / (tau2 + s2)
    f = lambda ld: np.mean((tau2 + s2) * (lam / (lam + np.exp(ld)) - w) ** 2)
    return float(minimize_scalar(f, bounds=(-15, 15), method="bounded", options={"xatol": 1e-10}).fun / np.mean(tau2 + s2))


def run(args):
    from e13_reml_diag import load, reml_basis
    name, trait, lo, hi = args; T = load(name)["traits"][trait]; rows = T["rows"]; G = load(name)["G"][np.ix_(rows, rows)]
    X = np.ones((len(rows), 1)) if T["C"] is None else T["C"]
    lam, A = reml_basis(G, X); z = A @ T["y"]; z2 = (z / z.std()) ** 2
    return dict(dataset=name, trait=trait, ci=[lo, hi], gap_lo=gap_at(lam, z2, lo), gap_hi=gap_at(lam, z2, hi))


if __name__ == "__main__":
    B = {(x["dataset"], x["trait"]): x["ci"] for x in json.load(open(os.path.join(HERE, "results", "e7_reml_boot.json")))}
    tasks = [(d, t, ci[0], ci[1]) for (d, t), ci in B.items()]
    with Pool(int(os.environ.get("GAP_PROCS", 4))) as pool:
        res = pool.map(run, tasks, chunksize=1)
    json.dump(res, open(os.path.join(HERE, "results", "gap_ci.json"), "w"), indent=1)
    up = np.array([max(r["gap_lo"], r["gap_hi"]) for r in res])
    print(f"traits {len(res)}; upper gap: median {np.median(up):.4f}, <0.005 {np.mean(up < 0.005):.0%}, <0.01 {np.mean(up < 0.01):.0%}, max {up.max():.4f}")
    for r in sorted(res, key=lambda r: -max(r['gap_lo'], r['gap_hi']))[:8]:
        print(f"  {r['dataset']:7s} {r['trait'][:28]:28s} ci {r['ci'][0]:.2f}-{r['ci'][1]:.2f} gaps {r['gap_lo']:.4f} {r['gap_hi']:.4f}")
