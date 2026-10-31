"""E13: REML-type gamma = 1 diagnostic (fixed-effect space projected out) -- correction of the size inflation found
on the rice GRM (check_size_centering.py: centring y zeroes the coordinate on the all-ones eigenvector, which the
parametric bootstrap does not reproduce; size 0.13-0.19 at nominal 0.05).

The corrected test works in the (n - p)-dimensional space orthogonal to the fixed effects X = [1, covariates]:
Q = orthonormal basis of col(X)^perp, K* = Q' G Q = U* diag(lam*) U*', z = U*' Q' y. Under y ~ N(X b, c G + s I),
z ~ N(0, c lam* + s I) exactly, so the LRT of gamma = 1 in z_i ~ N(0, c lam*_i^gamma + s) and its parametric
bootstrap are those of the REML likelihood (Patterson & Thompson 1971).

usage: python e13_reml_diag.py size     -> null size: rice (intercept only; chi2 + bootstrap) and mice with the real
                                           covariates of Biochem.Sodium (chi2; old vs REML version)
       python e13_reml_diag.py real     -> REML diagnostic (B = 200) for every trait of rice, wheat, mice, mice_auto
       python e13_reml_diag.py foldiag  -> REML diagnostic (chi2) on the training part of every 5 x 5 k-fold split
                                           (rice, wheat, mice; same splits as e4_real.py)
       python e13_reml_diag.py cagefixed -> mice: REML diagnostic (B = 200) with cage added as a fixed effect, i.e. the
                                           test restricted to within-cage contrasts (is the departure shared environment?)
Shared modules are only imported, never modified.
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import json, time, numpy as np
from multiprocessing import Pool
from scipy.stats import chi2

HERE = os.path.dirname(os.path.abspath(__file__))
_D = {}


def load(name):
    if name not in _D:
        import datasets
        _D[name] = datasets.rice(markers=False) if name == "rice" else datasets.LOADERS[name]()
    return _D[name]


def reml_basis(G, X):
    """lam* (normalised to mean 1) and the map y -> z for the space orthogonal to col(X)."""
    n = len(G); Qf, R = np.linalg.qr(X, mode="complete"); p = int(np.sum(np.abs(np.diag(R)) > 1e-8 * np.abs(R).max()))
    Q = Qf[:, p:]; ev, U = np.linalg.eigh(Q.T @ G @ Q); lam = np.maximum(ev, 1e-10)
    return lam / lam.mean(), (U.T @ Q.T)


def lrt(lam, z2):
    from spectral_filters import ml_fit
    c, s2, g, f1 = ml_fit(lam, z2); c0, s0, _, f0 = ml_fit(lam, z2, gamma=1.0)
    return max(2 * (f0 - f1), 0.0), g, c, s2, c0, s0


def reml_test(lam, A, y, B, seed):
    z = A @ y; z2 = (z / z.std()) ** 2
    lr, g, c, s2, c0, s0 = lrt(lam, z2); rng = np.random.default_rng(seed)
    lrb = [lrt(lam, rng.standard_normal(len(lam)) ** 2 * (c0 * lam + s0))[0] for _ in range(B)]
    return dict(gamma=float(g), lr=float(lr), p_chi2=float(chi2.sf(lr, 1)),
                p_boot=(float((1 + np.sum(np.array(lrb) >= lr)) / (B + 1)) if B else None), h2=float(c / (c + s2)))


def old_test(G_ev, G_U, y, C):
    """The diagnostic as used in e4_real.py (OLS residual, centred, all n eigen-directions), chi2 only."""
    if C is not None:
        y = y - C @ np.linalg.lstsq(C, y, rcond=None)[0]
    z2 = (G_U.T @ ((y - y.mean()) / y.std())) ** 2
    return float(chi2.sf(lrt(G_ev, z2)[0], 1))


def size_task(cfg):
    name, trait, seed, B = cfg["data"], cfg["trait"], cfg["seed"], cfg["B"]
    D = load(name); T = D["traits"][trait]; rows = T["rows"]; G = D["G"][np.ix_(rows, rows)]; n = len(rows)
    C = T["C"]; X = np.ones((n, 1)) if C is None else C
    rng = np.random.default_rng(seed)
    ev, U = np.linalg.eigh(G); lamG = np.maximum(ev, 1e-10); lamG = lamG / lamG.mean()
    g = U @ (np.sqrt(lamG / lamG.sum() * 0.5 * n) * rng.standard_normal(n))     # gamma = 1 signal, h2 = 0.5
    y = X @ rng.standard_normal(X.shape[1]) + g + np.sqrt(0.5) * rng.standard_normal(n)
    lam, A = reml_basis(G, X)
    r = reml_test(lam, A, y, B, seed + 1)
    return dict(cfg, p_old=old_test(lamG, U, y, C), p_reml=r["p_chi2"], p_reml_boot=r["p_boot"])


def fold_task(args):
    """REML diagnostic (chi2) on the training part of every fold of e4_real.py's 5 x 5 k-fold splits."""
    from e4_real import folds as split
    name, trait = args; D = load(name); T = D["traits"][trait]; rows = T["rows"]; out = []
    for seed in range(5):
        for fi, (tr, va) in enumerate(split(T, "kfold", seed)):
            G = D["G"][np.ix_(rows[tr], rows[tr])]; X = np.ones((len(tr), 1)) if T["C"] is None else T["C"][tr]
            lam, A = reml_basis(G, X); r = reml_test(lam, A, T["y"][tr], B=0, seed=0)
            out.append(dict(seed=seed, fold=fi, gamma=r["gamma"], p_chi2=r["p_chi2"]))
    return dict(dataset=name, trait=trait, folds=out)


def real_task(args, cage_fixed=False):
    name, trait = args; t0 = time.time()
    D = load(name); T = D["traits"][trait]; rows = T["rows"]; G = D["G"][np.ix_(rows, rows)]
    X = np.ones((len(rows), 1)) if T["C"] is None else T["C"]
    if cage_fixed:                                              # cage as a fixed effect: test within cages
        grp = np.asarray(T["groups"]).astype(str); cages = np.unique(grp)
        X = np.hstack([X, (grp[:, None] == cages[None, 1:]).astype(float)])
    lam, A = reml_basis(G, X)
    return dict(dataset=name, trait=trait, n=int(len(rows)), p_fixed=int(len(rows) - len(lam)),
                **reml_test(lam, A, T["y"], B=200, seed=0), seconds=time.time() - t0)


def cage_task(args):
    return real_task(args, cage_fixed=True)


if __name__ == "__main__":
    mode = sys.argv[1]; procs = int(os.environ.get("E13_PROCS", 3)); t0 = time.time()
    if mode == "size":
        cf = [dict(data="rice", trait="ht_cm_Yan_et_al_2007", rep=r, B=100) for r in range(200)] + \
             [dict(data="mice", trait="Biochem.Sodium", rep=r, B=0) for r in range(200)]
        for i, c in enumerate(cf): c["seed"] = 70000 + 10 * i
        with Pool(procs) as pool:
            res = pool.map(size_task, cf, chunksize=4)
        for d in ("rice", "mice"):
            R = [r for r in res if r["data"] == d]
            for k in ("p_old", "p_reml", "p_reml_boot"):
                p = np.array([r[k] for r in R if r[k] is not None])
                if len(p): print(f"{d:5s} {k:12s} size@.05 {np.mean(p < .05):.3f}  @.10 {np.mean(p < .10):.3f}  (reps {len(p)})", flush=True)
    else:
        tasks = []
        names = {"real": ("rice", "wheat", "mice", "mice_auto"), "foldiag": ("rice", "wheat", "mice"), "cagefixed": ("mice",)}[mode]
        for name in names:
            tasks += [(name, t) for t in load(name)["traits"]]
        _D.clear()
        with Pool(procs) as pool:
            res = pool.map({"real": real_task, "foldiag": fold_task, "cagefixed": cage_task}[mode], tasks, chunksize=1)
    json.dump(res, open(os.path.join(HERE, "results", f"e13_reml_{mode}.json"), "w")); print("DONE", mode, round(time.time() - t0), "s")
