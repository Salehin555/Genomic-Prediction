"""E7 (stability), E8 (scaling), E10 (ablation of the two corrections).

usage: python e7_e8_e10.py stability     -> bootstrap CI of gamma-hat per real trait (200 resamples of individuals);
                                            leading-eigenvector vs subpopulation F statistics (rice)
       python e7_e8_e10.py scaling       -> runtime and memory vs n (simulated 0/1/2 genotypes, p = 10,000)
       python e7_e8_e10.py ablation      -> original SAS-BLUP vs bounded leverage only vs inner-CV guard only vs both
"""
import os, sys
if sys.argv[1:2] != ["scaling"]:
    os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import json, time, numpy as np
from multiprocessing import Pool
from sklearn.model_selection import KFold

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


# ---------------------------------------------------------------- E7
def _boot(args):
    dname, trait, B = args
    import datasets
    from spectral_filters import ml_fit
    D = datasets.LOADERS[dname]() if dname != "rice" else datasets.rice(markers=False)
    T = D["traits"][trait]; rows, y, C = T["rows"], T["y"], T["C"]
    if C is not None: y = y - C @ np.linalg.lstsq(C, y, rcond=None)[0]
    G = D["G"][np.ix_(rows, rows)]; rng = np.random.default_rng(0); gs = []
    for b in range(B + 1):
        idx = np.arange(len(y)) if b == 0 else np.unique(rng.integers(0, len(y), len(y)))
        ev, U = np.linalg.eigh(G[np.ix_(idx, idx)]); lam = np.maximum(ev, 1e-10); lam /= lam.mean()
        yy = y[idx]; z2 = (U.T @ ((yy - yy.mean()) / yy.std())) ** 2
        gs.append(float(ml_fit(lam, z2)[2]))
    return dict(dataset=dname, trait=trait, gamma=gs[0], ci=[float(np.percentile(gs[1:], 2.5)), float(np.percentile(gs[1:], 97.5))])


def stability():
    import datasets
    jobs = []
    for d in ("rice", "wheat", "mice"):
        D = datasets.rice(markers=False) if d == "rice" else datasets.LOADERS[d]()
        jobs += [(d, t, 200) for t in D["traits"]]
    with Pool(11) as pool:
        res = pool.map(_boot, jobs, chunksize=1)
    D = datasets.rice(markers=False); G = D["G"]; cl = D["traits"]["ht_cm_Yan_et_al_2007"]["groups"]
    ev, U = np.linalg.eigh(G); U = U[:, ::-1]; F = []
    for i in range(20):                                          # one-way ANOVA F of eigenvector i on subpopulation
        u = U[:, i]; k = np.unique(cl); m = u.mean()
        ssb = sum(np.sum(cl == c) * (u[cl == c].mean() - m) ** 2 for c in k); ssw = sum(np.sum((u[cl == c] - u[cl == c].mean()) ** 2) for c in k)
        F.append(float((ssb / (len(k) - 1)) / (ssw / (len(u) - len(k)))))
    json.dump(dict(gamma_boot=res, rice_F=F), open(os.path.join(RES, "e7_stability.json"), "w")); print("DONE")


# ---------------------------------------------------------------- E8
def scaling():
    import tracemalloc
    from spectral_filters import fit_predict
    rng = np.random.default_rng(0); out = []
    for n in (500, 1000, 2000, 4000, 8000):
        f = rng.uniform(0.05, 0.5, 10000)
        X = rng.binomial(2, f, size=(n + 200, 10000)).astype(np.float64)
        y = X[:, :200] @ rng.standard_normal(200) * 0.05 + rng.standard_normal(n + 200)
        t = time.time(); Z = (X - 2 * f) / np.sqrt(2 * f * (1 - f)); K = Z @ Z.T / Z.shape[1]; t_grm = time.time() - t
        tr, te = np.arange(n), np.arange(n, n + 200)
        t = time.time(); np.linalg.eigh(K[np.ix_(tr, tr)]); t_eig = time.time() - t
        rec = dict(n=n, grm_s=t_grm, eigh_s=t_eig, grm_GB=n * n * 8 / 1e9)
        for m in ("rrBLUP_CV", "power_CV") + (("safe_CV",) if n <= 4000 else ()):
            t = time.time(); fit_predict(K[np.ix_(tr, tr)], y[tr], K[np.ix_(te, tr)], m); rec[m + "_s"] = time.time() - t
        out.append(rec); print(rec, flush=True)
    import os as _o
    json.dump(dict(results=out, cpu_threads=_o.cpu_count(), gpu=False), open(os.path.join(RES, "e8_scaling.json"), "w")); print("DONE")


# ---------------------------------------------------------------- E10
def _abl(args):
    dname, trait, variant, seed = args
    import datasets, sasblup_engine as E, sasblup_fixed as F
    from sklearn.metrics import r2_score
    D = datasets.rice(markers=False) if dname == "rice" else datasets.LOADERS[dname]()
    T = D["traits"][trait]; G = D["G"][np.ix_(T["rows"], T["rows"])]; y = T["y"]
    ev = np.maximum(np.linalg.eigvalsh(G), 1e-8); E.EFF_RANK = int(ev.sum() ** 2 / (ev ** 2).sum())
    out = []
    for fam in ("grouped", "smooth"):
        for fi, (tr, va) in enumerate(KFold(5, shuffle=True, random_state=seed).split(y)):
            m, s = y[tr].mean(), y[tr].std(); yt = (y[tr] - m) / s
            Gtr, Gva = G[np.ix_(tr, tr)], G[np.ix_(va, tr)]
            if variant == "original":
                est = E.SASBLUPEstimator(strategy=fam).fit(Gtr, yt); loo = float(est.loo_)
            elif variant == "bounded_only":
                ev_, U = E.safe_eigh(Gtr); z = U.T @ yt; w = F.family_weights(fam, ev_, U, z, yt)
                loo = float(E.loo_mse(w, U, z, yt)); alpha = U @ ((w / (ev_ + 1e-9)) * z)
                est = type("P", (), {"predict": lambda self, Gv, a=alpha: Gv @ a})()
            elif variant == "guard_only":
                base = E.SASBLUPEstimator(strategy=fam).fit(Gtr, yt); sc = E.SASBLUPEstimator(strategy="scalar").fit(Gtr, yt)
                sa = ss = 0.0
                for itr, iva in KFold(5, shuffle=True, random_state=0).split(yt):
                    Gi, Gv = Gtr[np.ix_(itr, itr)], Gtr[np.ix_(iva, itr)]
                    sa += np.sum((yt[iva] - E.SASBLUPEstimator(strategy=fam).fit(Gi, yt[itr]).predict(Gv)) ** 2)
                    ss += np.sum((yt[iva] - E.SASBLUPEstimator(strategy="scalar").fit(Gi, yt[itr]).predict(Gv)) ** 2)
                est = base if sa < ss else sc; loo = float(base.loo_)
            else:
                est = F.SASBLUPFixed(strategy=fam).fit(Gtr, yt); loo = np.nan
            p = np.asarray(est.predict(Gva)).ravel() * s + m
            out.append(dict(dataset=dname, trait=trait, variant=variant, family=fam, seed=seed, fold=fi,
                            r2=float(r2_score(y[va], p)), loo=loo))
    return out


def ablation():
    jobs = [(d, t, v, s) for d, ts in (("wheat", ("Env1", "Env2", "Env4", "Env5")),
                                      ("rice", ("ht_cm_Yan_et_al_2007", "Brown_Grain_Length_mm", "Brown_Grain_Width_mm", "Brown_Grain_L/Wratio")))
            for t in ts for v in ("original", "bounded_only", "guard_only", "both") for s in (0, 1)]
    with Pool(11) as pool:
        res = [r for rr in pool.map(_abl, jobs, chunksize=1) for r in rr]
    json.dump(res, open(os.path.join(RES, "e10_ablation.json"), "w")); print("DONE")


if __name__ == "__main__":
    {"stability": stability, "scaling": scaling, "ablation": ablation}[sys.argv[1]]()
