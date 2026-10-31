"""E11: two extra validations requested in the validation register.

usage: python e11_extra.py foldiag DATASET   -> the gamma = 1 diagnostic computed on the TRAINING part of every fold only (chi2 p
                                               on all 25 folds, bootstrap p on seed 0's 5 folds)
                                               (same splits and within-fold covariate adjustment as e4_real.py), so that
                                               the diagnostic-vs-gain association cannot share noise with held-out data
       python e11_extra.py ldak DATASET      -> LDAK-alpha-type baseline: GBLUP on K_alpha = Z diag([f(1-f)]^(1+alpha)) Z'
                                               (alpha = -1 is the standard GRM), alpha chosen by maximum likelihood on
                                               the training part, ridge parameter by inner 5-fold CV (no LD weights)
       python e11_extra.py epi DATASET       -> additive + epistatic kernel K_rho = (1-rho) G + rho (G o G) (each scaled to
                                               mean diagonal 1), rho in {0, .25, .5, .75, 1} chosen by maximum likelihood
                                               on the training part, ridge by inner 5-fold CV (EG-BLUP-type check of the
                                               "gamma < 1 = non-additive signal" interpretation)
DATASET: rice | wheat | mice.  Shared modules are only imported, never modified.
Env: E11_PROCS (workers, default 11); E11_SCHEME=cage holds out whole cages (mice; e4_real.py splits, 3 seeds) and
writes results/e11_{mode}_{dataset}_cage.json.
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import json, time, numpy as np
from multiprocessing import Pool
from sklearn.model_selection import KFold

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
ALPHAS = (-1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25)
RHOS = (0.0, 0.25, 0.5, 0.75, 1.0)
SCHEME = os.environ.get("E11_SCHEME", "kfold")                  # "cage": whole cages held out (mice), 3 seeds
_C = {}


def raw_markers(name):
    if name == "rice": return np.load(os.path.join(DATA, "rice_X.npy")).astype(np.float64)
    import pyreadr
    if name == "wheat": return pyreadr.read_r(os.path.join(DATA, "wheat.RData"))["wheat.X"].values.astype(np.float64) * 2.0
    return pyreadr.read_r(os.path.join(DATA, "mice.RData"))["mice.X"].values.astype(np.float64)


def setup(name):
    if name not in _C:
        import datasets
        D = datasets.rice() if name == "rice" else datasets.LOADERS[name]()
        X = raw_markers(name); f = X.mean(0) / 2.0; ok = (f > 0.01) & (f < 0.99); f = f[ok]
        Z = (X[:, ok] - 2 * f) / np.sqrt(2 * f * (1 - f)); K = {}
        for a in ALPHAS:
            w = (f * (1 - f)) ** (1 + a); Zw = Z * np.sqrt(w)
            K[a] = Zw @ Zw.T / w.sum() + 1e-4 * np.eye(len(Z))
        G = D["G"]; Ga = G / np.mean(np.diag(G)); Ge = G * G; Ge = Ge / np.mean(np.diag(Ge))
        for r in RHOS: K[("epi", r)] = (1 - r) * Ga + r * Ge + 1e-4 * np.eye(len(G))
        _C[name] = (D, K)
    return _C[name]


def adjust(C, y, tr, va):
    if C is None: return y[tr], y[va]
    b = np.linalg.lstsq(C[tr], y[tr], rcond=None)[0]
    return y[tr] - C[tr] @ b, y[va] - C[va] @ b


def run(args):
    mode, name, trait = args
    from spectral_filters import ml_fit, fit_predict
    from scipy.stats import chi2
    D, K = setup(name); T = D["traits"][trait]; rows, y, C = T["rows"], T["y"], T["C"]
    out = dict(dataset=name, trait=trait, mode=mode, folds=[]); t0 = time.time()
    from e4_real import folds as split                          # same splits as e4_real.py (kfold or cage)
    for seed in range(3 if SCHEME == "cage" else 5):
        for fi, (tr, va) in enumerate(split(T, SCHEME, seed)):
            ytr, yva = adjust(C, y, tr, va); rec = dict(seed=seed, fold=fi)
            if mode == "foldiag":
                G = D["G"][np.ix_(rows[tr], rows[tr])]
                ev, U = np.linalg.eigh(G); lam = np.maximum(ev, 1e-10); lam /= lam.mean()
                z2 = (U.T @ ((ytr - ytr.mean()) / ytr.std())) ** 2
                c, s2, g, f1 = ml_fit(lam, z2); c0, s0, _, f0 = ml_fit(lam, z2, gamma=1.0); lr = max(2 * (f0 - f1), 0.0)
                rng = np.random.default_rng(seed * 10 + fi); lrb = []
                for _ in range(100 if seed == 0 else 0):                # bootstrap on seed 0's folds only (cost)
                    zb2 = rng.standard_normal(len(lam)) ** 2 * (c0 * lam + s0)
                    lrb.append(max(2 * (ml_fit(lam, zb2, gamma=1.0)[3] - ml_fit(lam, zb2)[3]), 0.0))
                rec.update(gamma=float(g), p_chi2=float(chi2.sf(lr, 1)), p_boot=(float((1 + np.sum(np.array(lrb) >= lr)) / 101) if lrb else None))
            else:
                best = None
                for a in (ALPHAS if mode == "ldak" else [("epi", r) for r in RHOS]):                                      # alpha by ML on the training part (LDAK-style)
                    Ka = K[a][np.ix_(rows[tr], rows[tr])]
                    ev, U = np.linalg.eigh(Ka); lam = np.maximum(ev, 1e-10); lam /= lam.mean()
                    z2 = (U.T @ ((ytr - ytr.mean()) / ytr.std())) ** 2
                    # -loglik of y ~ N(0, c K_a + s I) in K_a's own eigenbasis; comparable across a because c is free
                    nll = ml_fit(lam, z2, gamma=1.0)[3]
                    if best is None or nll < best[1]: best = (a, nll)
                a = best[0]; Ka = K[a]
                p, _ = fit_predict(Ka[np.ix_(rows[tr], rows[tr])], ytr, Ka[np.ix_(rows[va], rows[tr])], "rrBLUP_CV", seed=seed)
                rec.update(**({"alpha": float(a)} if mode == "ldak" else {"rho": float(a[1])}), r2=float(1 - np.sum((yva - p) ** 2) / np.sum((yva - yva.mean()) ** 2)))
            out["folds"].append(rec)
    out["seconds"] = time.time() - t0
    return out


if __name__ == "__main__":
    mode, name = sys.argv[1], sys.argv[2]
    import datasets                                             # trait list only; the kernels are built in the workers
    D = datasets.rice() if name == "rice" else datasets.LOADERS[name]()
    with Pool(min(int(os.environ.get("E11_PROCS", 11)), len(D["traits"]))) as pool:
        res = pool.map(run, [(mode, name, t) for t in D["traits"]], chunksize=1)
    json.dump(res, open(os.path.join(HERE, "results", f"e11_{mode}_{name}{'_cage' if SCHEME == 'cage' else ''}.json"), "w")); print("DONE", mode, name)
