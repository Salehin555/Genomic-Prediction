"""E15 (R5.4): what happens when the fixed effects (sex, litter, cage density, test date, age) are not removed.

For every mouse trait, with the same five folds (seed 0) and the same estimators as e4_real.py:
  adjusted    covariates estimated by OLS on the training part and removed from both parts (as in e4_real.py)
  unadjusted  intercept only
we record the held-out R2 of rrBLUP_CV and power_CV, and the REML test of gamma = 1 (chi2 p) with X = covariates or
X = intercept only (e13_reml_diag.reml_basis). Replaces the console-only pilot (pilot_mice.py) behind claim K14.
writes results/e15_adjust.json;  usage: python e15_adjust.py [procs]
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import json, numpy as np
from multiprocessing import Pool
from sklearn.model_selection import KFold

HERE = os.path.dirname(os.path.abspath(__file__))


def run(trait):
    from e13_reml_diag import load, reml_basis, reml_test
    from e4_real import adjust
    from spectral_filters import fit_predict
    D = load("mice"); T = D["traits"][trait]; rows = T["rows"]; y = T["y"]; C = T["C"]
    G = D["G"][np.ix_(rows, rows)]; one = np.ones((len(y), 1))
    out = dict(trait=trait, n=int(len(y)))
    for lab, X in (("adjusted", C), ("unadjusted", one)):
        lam, A = reml_basis(G, X); t = reml_test(lam, A, y, 0, 0)
        out[lab] = dict(gamma=t["gamma"], p_chi2=t["p_chi2"], r2={"rrBLUP_CV": [], "power_CV": []})
        for tr, va in KFold(5, shuffle=True, random_state=0).split(y):
            ytr, yva = adjust(X if lab == "adjusted" else None, y, tr, va)
            for m in ("rrBLUP_CV", "power_CV"):
                p, _ = fit_predict(G[np.ix_(tr, tr)], ytr, G[np.ix_(va, tr)], m, seed=0)
                out[lab]["r2"][m].append(float(1 - np.sum((yva - p) ** 2) / np.sum((yva - yva.mean()) ** 2)))
    print(trait, {k: (round(out[k]["gamma"], 2), round(np.mean(out[k]["r2"]["power_CV"]) - np.mean(out[k]["r2"]["rrBLUP_CV"]), 4))
                  for k in ("adjusted", "unadjusted")}, flush=True)
    return out


if __name__ == "__main__":
    from e13_reml_diag import load
    traits = list(load("mice")["traits"].keys())
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as pool:
        res = pool.map(run, traits, chunksize=1)
    json.dump(res, open(os.path.join(HERE, "results", "e15_adjust.json"), "w"), indent=1)
    print("DONE")
