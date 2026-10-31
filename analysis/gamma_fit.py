"""Fit z_i ~ N(0, c*lam_i^gamma + s2) by ML on the GRM eigenbasis; gamma = 1 is the GBLUP (proportional) model.
Likelihood-ratio test of gamma = 1, and comparison with the held-out R2 of PCR / best SAS family vs rrBLUP."""
import json, numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler
from run_fixed import load

def nll(p, lam, z2, gamma=None):
    lc, ls = p[0], p[1]; g = p[2] if gamma is None else gamma
    v = np.exp(lc) * lam ** g + np.exp(ls)
    return 0.5 * np.sum(np.log(v) + z2 / v)

def fit(lam, z2, gamma=None):
    best = None
    for g0 in ([0.5, 1.0, 2.0, 3.0] if gamma is None else [None]):
        for lc in (-2, 0, 2):
            x0 = [lc, -1.0] + ([g0] if gamma is None else [])
            b = [(-20, 10), (-25, 5)] + ([(0.0, 6.0)] if gamma is None else [])
            r = minimize(nll, x0, args=(lam, z2, gamma), method="L-BFGS-B", bounds=b)
            if best is None or r.fun < best.fun: best = r
    return best

for name in ("rice", "wheat"):
    G, Y = load(name); ev, U = np.linalg.eigh(G); ev = np.maximum(ev, 1e-8)
    lam = ev / ev.mean()
    res = json.load(open(f"fixed_{name}.json"))
    print(f"\n{name}: n={len(ev)}")
    for t in Y.columns:
        y = StandardScaler().fit_transform(Y[t].values.reshape(-1, 1)).ravel()
        z2 = (U.T @ y) ** 2
        f1, f0 = fit(lam, z2), fit(lam, z2, gamma=1.0)
        lr = 2 * (f0.fun - f1.fun); p = chi2.sf(max(lr, 0), 1)
        fo = res[t]["folds"]; sc = np.array(fo["scalar"])
        d = lambda m: np.nanmean(np.array(fo[m]) - sc)
        best_sas = max(("grouped", "smooth", "monotone", "spline"), key=d)
        print(f"  {t:14s} gamma={f1.x[2]:.2f}  LRT(gamma=1) p={p:.3g}   dR2: PCR={d('PCR'):+.4f}  "
              f"best SAS({best_sas})={d(best_sas):+.4f}  KRR={d('KRR_RBF'):+.4f}")
