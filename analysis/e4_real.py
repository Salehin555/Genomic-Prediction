"""E4/E5/E6/E9: real-data comparison under identical splits.

usage: python e4_real.py DATASET SCHEME GROUP N_SEEDS
  DATASET  rice | wheat | mice | mice_auto        (mice_auto: GRM from autosomal SNPs only)
  SCHEME   kfold (5-fold, shuffled by seed) | cage (5 folds of whole cages; mice only)
           | subpop (leave one k-means subpopulation out; rice only, deterministic so use N_SEEDS = 1)
  GROUP    spectral (rrBLUP_ML, rrBLUP_CV, power_CV, safe_CV, gammaBLUP_ML, PCR_CV)
           kernel   (KRR_RBF, DNNGP from sasblup_engine)
           bayes    (BRR, BayesC, BayesR; marker-based Gibbs, bayes_gibbs.py)
Fixed covariates (mice: sex, litter, cage density, test date, age) are estimated by OLS on the training part of each
fold and removed from both parts; R2 is computed on the covariate-adjusted held-out phenotype.
Also records, per trait, the gamma = 1 diagnostic on the full (covariate-adjusted) data: chi2 and bootstrap p-values.
Env: E4_METHODS (comma list overriding the group's methods), E4_TAG (suffix for the output file name),
     GIBBS_ITER / GIBBS_BURN (bayes group), DIAG_B (bootstrap draws of the diagnostic).
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import json, time, numpy as np
from multiprocessing import Pool
from sklearn.model_selection import KFold

HERE = os.path.dirname(os.path.abspath(__file__))
GROUPS = {"spectral": ("rrBLUP_ML", "rrBLUP_CV", "power_CV", "safe_CV", "gammaBLUP_ML", "PCR_CV"),
          "kernel": ("KRR_RBF", "DNNGP"), "bayes": ("BRR", "BayesC", "BayesR")}
_D = {}


def data(name):
    if name not in _D:
        import datasets; _D[name] = datasets.LOADERS[name]()
    return _D[name]


def folds(tr, scheme, seed):
    n = len(tr["y"])
    if scheme == "kfold":
        return list(KFold(5, shuffle=True, random_state=seed).split(np.arange(n)))
    if scheme == "subpop":                                  # leave one subpopulation out (rice, E6)
        return [(np.where(tr["groups"] != k)[0], np.where(tr["groups"] == k)[0]) for k in np.unique(tr["groups"])]
    cages = np.unique(tr["groups"]); rng = np.random.default_rng(seed); rng.shuffle(cages)
    fold_of = {c: i % 5 for i, c in enumerate(cages)}; f = np.array([fold_of[c] for c in tr["groups"]])
    return [(np.where(f != k)[0], np.where(f == k)[0]) for k in range(5)]


def adjust(C, y, tr, va):
    if C is None:
        return y[tr], y[va]
    b = np.linalg.lstsq(C[tr], y[tr], rcond=None)[0]
    return y[tr] - C[tr] @ b, y[va] - C[va] @ b


def diagnostic(G, y, C, B=200, seed=0):
    from spectral_filters import ml_fit
    from scipy.stats import chi2
    if C is not None:
        y = y - C @ np.linalg.lstsq(C, y, rcond=None)[0]
    ev, U = np.linalg.eigh(G); lam = np.maximum(ev, 1e-10); lam = lam / lam.mean()
    z2 = (U.T @ ((y - y.mean()) / y.std())) ** 2
    c, s2, g, f1 = ml_fit(lam, z2); c0, s0, _, f0 = ml_fit(lam, z2, gamma=1.0)
    lr = max(2 * (f0 - f1), 0.0); rng = np.random.default_rng(seed); lrb = []
    for _ in range(B):
        zb2 = rng.standard_normal(len(lam)) ** 2 * (c0 * lam + s0)
        lrb.append(max(2 * (ml_fit(lam, zb2, gamma=1.0)[3] - ml_fit(lam, zb2)[3]), 0.0))
    return dict(gamma=float(g), lr=float(lr), p_chi2=float(chi2.sf(lr, 1)),
                p_boot=float((1 + np.sum(np.array(lrb) >= lr)) / (B + 1)), h2=float(c / (c + s2)),
                eff_rank=float(lam.sum() ** 2 / (lam ** 2).sum()))


def run(args):
    dname, trait, scheme, group, n_seeds = args
    from spectral_filters import fit_predict
    D = data(dname); T = D["traits"][trait]; rows = T["rows"]; y = T["y"]; C = T["C"]
    G = D["G"][np.ix_(rows, rows)]; Z = D["Z"][rows] if (group == "bayes") else None
    methods = tuple(os.environ["E4_METHODS"].split(",")) if os.environ.get("E4_METHODS") else GROUPS[group]
    out = dict(dataset=dname, trait=trait, scheme=scheme, group=group, n=int(len(y)),
               r2={m: [] for m in methods}, r={m: [] for m in methods}, info={m: [] for m in methods}, preds=[])
    if group == "spectral" and scheme == "kfold":
        out["diag"] = diagnostic(G, y, C, B=int(os.environ.get("DIAG_B", 100 if dname == "mice_auto" else 200)))
    t0 = time.time()
    for seed in range(n_seeds):
        for fi, (tr, va) in enumerate(folds(T, scheme, seed)):
            ytr, yva = adjust(C, y, tr, va); rec = dict(seed=seed, fold=fi, va=va.tolist(), y=yva.tolist(), p={})
            for m in methods:
                try:
                    if group == "spectral":
                        p, inf = fit_predict(G[np.ix_(tr, tr)], ytr, G[np.ix_(va, tr)], m, seed=seed)
                    elif group == "kernel":
                        import sasblup_engine as E
                        sc_m, sc_s = ytr.mean(), ytr.std()
                        est = (E.KRR_RBF_Predictor() if m == "KRR_RBF" else E.DNNGPPredictor(seed=seed)).fit(
                            G[np.ix_(tr, tr)], (ytr - sc_m) / sc_s)
                        p = np.asarray(est.predict(G[np.ix_(va, tr)])).ravel() * sc_s + sc_m; inf = {}
                    else:
                        from bayes_gibbs import GibbsRegressor, geweke_z
                        est = GibbsRegressor(m, n_iter=int(os.environ.get("GIBBS_ITER", 3000)), burn=int(os.environ.get("GIBBS_BURN", 1000)), seed=seed).fit(Z[tr], ytr)
                        p = est.predict(Z[va]); inf = dict(geweke=geweke_z(est.trace_, est.burn))
                    r2 = 1 - np.sum((yva - p) ** 2) / np.sum((yva - yva.mean()) ** 2)
                    rr = float(np.corrcoef(yva, p)[0, 1])
                except Exception as ex:
                    print(dname, trait, m, "ERROR", repr(ex), flush=True); r2, rr, p, inf = np.nan, np.nan, np.full(len(va), np.nan), {}
                out["r2"][m].append(float(r2)); out["r"][m].append(rr)
                out["info"][m].append({k: v for k, v in inf.items() if k != "err"}); rec["p"][m] = np.asarray(p).tolist()
            out["preds"].append(rec)
        print(f"{dname}/{trait}/{scheme}/{group}: seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    out["seconds"] = time.time() - t0
    return out


if __name__ == "__main__":
    dname, scheme, group, n_seeds = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    procs = int(sys.argv[5]) if len(sys.argv) > 5 else 11
    fn = os.path.join(HERE, "results", f"e4_{dname}_{scheme}_{group}{os.environ.get('E4_TAG', '')}.json")
    # the same run started separately holds FN.external.lock: wait for it, and skip if it wrote the output
    if os.path.exists(fn + ".external.lock"):
        while os.path.exists(fn + ".external.lock"): time.sleep(60)
        if os.path.exists(fn): print("SKIP, written by the external run:", fn); sys.exit(0)
    traits = list(data(dname)["traits"].keys())
    jobs = [(dname, t, scheme, group, n_seeds) for t in traits]
    with Pool(min(procs, len(jobs))) as pool:
        res = pool.map(run, jobs, chunksize=1)
    json.dump(res, open(fn, "w")); print("DONE", fn)
