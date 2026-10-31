"""Rerun the real-data experiments with the corrected SAS-BLUP and the corrected spectral-gap diagnostic.
usage: python run_fixed.py rice|wheat|tcga [n_seeds]
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("MKL_NUM_THREADS", "2")
import json, time, numpy as np, pandas as pd
from multiprocessing import Pool
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

W = os.path.join(os.path.dirname(os.path.abspath(__file__)), "work")
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
BASELINES = ("BayesB", "KRR_RBF", "PCR", "DNNGP")


def load(name):
    if name == "rice":
        G = np.load(W + r"\rice_GRM.npy") + 1e-4 * np.eye(191)
        ids = np.load(W + r"\rice_ids.npy").tolist()
        ph = pd.read_csv(R + r"\Genotype Rice Data\phenotype.csv").set_index("Taxa"); ph.index = ph.index.astype(str)
        return G, ph.loc[ids]
    if name == "wheat":
        import pyreadr
        r = pyreadr.read_r(W + r"\bglr_dl\x\BGLR\data\wheat.RData")
        X = r["wheat.X"].values.astype(np.float64) * 2.0
        f = X.mean(0) / 2.0; ok = (f > 0.01) & (f < 0.99); X, f = X[:, ok], f[ok]
        Z = (X - 2 * f) / np.sqrt(2 * f * (1 - f))
        Y = r["wheat.Y"].copy(); Y.columns = ["Env%s" % c for c in Y.columns]
        return Z @ Z.T / Z.shape[1] + 1e-4 * np.eye(len(Z)), Y
    if name == "tcga":
        cache = W + r"\tcga_K.npy"
        lab = pd.read_csv(R + r"\Cancer Gene Data\labels (1).csv", index_col=0)
        if not os.path.exists(cache):
            X = pd.read_csv(R + r"\Cancer Gene Data\data (1).csv", index_col=0)
            lab = lab.loc[X.index]
            X = X.loc[:, X.std() > 0].values.astype(np.float64)
            Z = StandardScaler().fit_transform(X)
            np.save(cache, Z @ Z.T / Z.shape[1])
        K = np.load(cache)
        Y = pd.DataFrame({c: (lab["Class"].values == c).astype(float) for c in sorted(lab["Class"].unique())},
                         index=lab.index)
        return K, Y
    raise ValueError(name)


def run_trait(args):
    name, t, n_seeds = args
    import sasblup_engine as E
    import sasblup_fixed as F
    from spectral_gap import diagnose
    import torch; torch.set_num_threads(1)
    G, Y = load(name)
    ev, U = np.linalg.eigh(G); ev = np.maximum(ev, 1e-8)
    E.EFF_RANK = int(ev.sum() ** 2 / (ev ** 2).sum())
    y = Y[t].values.astype(np.float64)
    ys = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()
    gap = diagnose(ev, U, ys, B=1000 if len(y) < 700 else 400)
    methods = list(F.FAMILIES) + list(BASELINES)
    folds = {m: [] for m in methods}; folds_r = {m: [] for m in methods}
    guard = {m: [] for m in F.FAMILIES}; prof = {m: [] for m in F.FAMILIES}; t0 = time.time()
    for seed in range(n_seeds):
        for fi, (tr, va) in enumerate(KFold(5, shuffle=True, random_state=seed).split(y)):
            Gtr, Gva = G[np.ix_(tr, tr)], G[np.ix_(va, tr)]
            sc = StandardScaler(); yt = sc.fit_transform(y[tr].reshape(-1, 1)).ravel()
            for m in methods:
                try:
                    if m in F.FAMILIES:
                        est = F.SASBLUPFixed(strategy=m).fit(Gtr, yt)
                        guard[m].append(bool(est.guard_used_))
                        if seed == 0: prof[m].append(est.weights_.tolist())
                    elif m == "BayesB": est = E.BayesBPredictor().fit(Gtr, yt)
                    elif m == "KRR_RBF": est = E.KRR_RBF_Predictor().fit(Gtr, yt)
                    elif m == "PCR": est = E.PCRPredictor().fit(Gtr, yt)
                    else: est = E.DNNGPPredictor(seed=seed).fit(Gtr, yt)
                    p = sc.inverse_transform(np.asarray(est.predict(Gva)).reshape(-1, 1)).ravel()
                    mt = E.compute_metrics(y[va], p)
                except Exception as ex:
                    print(name, t, m, "ERROR", ex, flush=True)
                    mt = dict(r2=np.nan, r=np.nan)
                folds[m].append(mt["r2"]); folds_r[m].append(mt["r"])
        print(f"{name}/{t}: seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    return t, dict(gap=gap, folds=folds, folds_r=folds_r, guard=guard, profiles=prof, eff_rank=E.EFF_RANK,
                   ev=ev.tolist(), seconds=time.time() - t0)


if __name__ == "__main__":
    name = sys.argv[1]; n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    _, Y = load(name)
    traits = list(Y.columns)
    with Pool(len(traits)) as pool:
        res = dict(pool.map(run_trait, [(name, t, n_seeds) for t in traits]))
    json.dump(res, open(W + rf"\fixed_{name}.json", "w"), default=float)
    for t, r in res.items():
        print(f"\n== {name} / {t}   gap p={r['gap']['p_value']:.3f}  ({r['seconds']:.0f}s)")
        sc = np.array(r["folds"]["scalar"])
        for m, v in sorted(r["folds"].items(), key=lambda kv: -np.nanmean(kv[1])):
            v = np.array(v); g = r["guard"].get(m)
            gs = f" guard-reverted {np.mean(g):.0%}" if g else ""
            print(f"   {m:9s} R2={np.nanmean(v):.4f} ± {np.nanstd(v):.4f}  dR2={np.nanmean(v - sc):+.4f}  "
                  f"min fold dR2={np.nanmin(v - sc):+.4f}{gs}")
    print("DONE")
