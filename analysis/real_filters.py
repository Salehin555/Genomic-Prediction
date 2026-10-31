"""Spectral-filter predictors (spectral_filters.py) on rice and wheat, same 5 seeds x 5 folds as run_fixed.py."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, numpy as np, pandas as pd, pyreadr
from multiprocessing import Pool
from sklearn.model_selection import KFold
from spectral_filters import fit_predict, ml_fit, METHODS

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
PHENO = os.path.join(HERE, "..", "data", "Genotype Rice Data", "phenotype.csv")


def load(name):
    if name == "rice":
        G = np.load(os.path.join(DATA, "rice_GRM.npy")) + 1e-4 * np.eye(191)
        ids = np.load(os.path.join(DATA, "rice_ids.npy")).tolist()
        ph = pd.read_csv(PHENO).set_index("Taxa"); ph.index = ph.index.astype(str)
        return G, ph.loc[ids]
    r = pyreadr.read_r(os.path.join(DATA, "wheat.RData"))
    X = r["wheat.X"].values.astype(np.float64) * 2.0
    f = X.mean(0) / 2.0; ok = (f > 0.01) & (f < 0.99); X, f = X[:, ok], f[ok]
    Z = (X - 2 * f) / np.sqrt(2 * f * (1 - f))
    Y = r["wheat.Y"].copy(); Y.columns = ["Env%s" % c for c in Y.columns]
    return Z @ Z.T / Z.shape[1] + 1e-4 * np.eye(len(Z)), Y


def run(args):
    name, t = args
    G, Y = load(name); y = Y[t].values.astype(float)
    ev, U = np.linalg.eigh(G); lam = np.maximum(ev, 1e-10); lam = lam / lam.mean()
    z2 = (U.T @ ((y - y.mean()) / y.std())) ** 2
    g_hat = float(ml_fit(lam, z2)[2])
    r2 = {m: [] for m in METHODS}; info = {m: [] for m in METHODS}
    for seed in range(5):
        for tr, va in KFold(5, shuffle=True, random_state=seed).split(y):
            sst = np.sum((y[va] - y[va].mean()) ** 2)
            for m in METHODS:
                p, inf = fit_predict(G[np.ix_(tr, tr)], y[tr], G[np.ix_(va, tr)], m, seed=seed)
                r2[m].append(float(1 - np.sum((y[va] - p) ** 2) / sst)); info[m].append(inf)
    return f"{name}/{t}", dict(gamma_full=g_hat, r2=r2, info=info)


if __name__ == "__main__":
    jobs = [("rice", t) for t in load("rice")[1].columns] + [("wheat", t) for t in load("wheat")[1].columns]
    with Pool(8) as pool:
        res = dict(pool.map(run, jobs))
    json.dump(res, open(os.path.join(HERE, "results", "real_filters.json"), "w"))
    for k, r in res.items():
        b = np.array(r["r2"]["rrBLUP_ML"])
        s = "  ".join(f"{m}={np.mean(np.array(v) - b):+.4f}" for m, v in r["r2"].items() if m != "rrBLUP_ML")
        a = [d["a"] for d in r["info"]["power_CV"]]
        print(f"{k:32s} gamma={r['gamma_full']:.2f} rrBLUP_ML R2={b.mean():.4f}  {s}  (power a median {np.median(a)})")
