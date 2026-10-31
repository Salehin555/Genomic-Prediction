"""Pilot: decay exponent gamma (ML, LRT of gamma = 1) for the BGLR mice traits, and a 1-seed 5-fold check of the
power filter vs CV-tuned rrBLUP on every trait (individuals with the phenotype observed)."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np, pandas as pd, pyreadr
from multiprocessing import Pool
from scipy.stats import chi2
from sklearn.model_selection import KFold
from spectral_filters import ml_fit, fit_predict

HERE = os.path.dirname(os.path.abspath(__file__))


def grm():
    r = pyreadr.read_r(os.path.join(HERE, "data", "mice.RData"))
    X = r["mice.X"].values.astype(float); f = X.mean(0) / 2; ok = (f > 0.01) & (f < 0.99)
    Z = (X[:, ok] - 2 * f[ok]) / np.sqrt(2 * f[ok] * (1 - f[ok]))
    return Z @ Z.T / ok.sum() + 1e-4 * np.eye(len(Z)), r["mice.pheno"]


def cv(args):
    t, y, G = args; out = {"n": len(y)}
    for m in ("rrBLUP_CV", "power_CV"):
        r2 = []
        for tr, va in KFold(5, shuffle=True, random_state=0).split(y):
            p, _ = fit_predict(G[np.ix_(tr, tr)], y[tr], G[np.ix_(va, tr)], m)
            r2.append(1 - np.sum((y[va] - p) ** 2) / np.sum((y[va] - y[va].mean()) ** 2))
        out[m] = np.mean(r2)
    return t, out


if __name__ == "__main__":
    G0, ph = grm()
    traits = [c for c in ph.columns if c.startswith(("Obesity.BMI", "Obesity.BodyLength", "Obesity.EndNormalBW", "Biochem."))]
    rows, jobs = [], []
    for t in traits:
        y = pd.to_numeric(ph[t], errors="coerce").values; ok = ~np.isnan(y)
        if ok.sum() < 500: continue
        y, G = y[ok], G0[np.ix_(ok, ok)]
        ev, U = np.linalg.eigh(G); lam = np.maximum(ev, 1e-10); lam /= lam.mean()
        ys = (y - y.mean()) / y.std(); z2 = (U.T @ ys) ** 2
        c, s2, g, f1 = ml_fit(lam, z2); f0 = ml_fit(lam, z2, gamma=1.0)[3]
        p = chi2.sf(max(2 * (f0 - f1), 0), 1)
        rows.append((t, g, p, len(y)))
        jobs.append((t, ys, G))
    gam = {}
    for t, g, p, n in sorted(rows, key=lambda r: -r[1]):
        print(f"  {t:28s} n={n:5d} gamma={g:5.2f}  LRT p={p:.2g}"); gam[t] = g
    with Pool(6) as pool:
        for t, o in pool.map(cv, jobs):
            print(f"  CV {t:28s} gamma={gam[t]:.2f} rrBLUP_CV={o['rrBLUP_CV']:.4f}  power_CV={o['power_CV']:.4f}  diff={o['power_CV'] - o['rrBLUP_CV']:+.4f}")
