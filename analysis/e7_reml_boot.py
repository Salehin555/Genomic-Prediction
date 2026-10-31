"""E7 redone with the REML basis (Error C): bootstrap CI of gamma-hat for every trait of the five panels.

The original E7 (e7_e8_e10.py stability) fitted gamma on the eigenbasis of G after OLS adjustment, i.e. without
projecting out the fixed effects; this version uses reml_basis(G, X) from e13_reml_diag, as the test does.
Resampling: individuals with replacement, duplicates dropped (as in the original E7), 200 resamples, seed 0.
usage: python e7_reml_boot.py            (env E7_PROCS, default 4)
writes results/e7_reml_boot.json
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import json, time, numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
PANELS = ("rice", "wheat", "mice", "rdp1", "sorghum")
B = 200


def boot(args):
    from e13_reml_diag import load, reml_basis
    from spectral_filters import ml_fit
    name, trait = args; t0 = time.time()
    D = load(name); T = D["traits"][trait]; rows, y = T["rows"], T["y"]; G = D["G"][np.ix_(rows, rows)]
    X = np.ones((len(rows), 1)) if T["C"] is None else T["C"]
    rng = np.random.default_rng(0); gs = []
    for b in range(B + 1):
        idx = np.arange(len(y)) if b == 0 else np.unique(rng.integers(0, len(y), len(y)))
        Xb = X[idx]; Xb = Xb[:, np.abs(Xb).sum(0) > 0]                     # drop dummies absent from the resample
        lam, A = reml_basis(G[np.ix_(idx, idx)], Xb); z = A @ y[idx]
        gs.append(float(ml_fit(lam, (z / z.std()) ** 2)[2]))
    return dict(dataset=name, trait=trait, n=int(len(y)), gamma=gs[0], ci=[float(np.percentile(gs[1:], 2.5)), float(np.percentile(gs[1:], 97.5))],
                at_bound=float(np.mean([g <= 1e-3 or g >= 5.99 for g in gs[1:]])), seconds=time.time() - t0)


if __name__ == "__main__":
    from e13_reml_diag import load
    tasks = [(p, t) for p in PANELS for t in load(p)["traits"]]
    tasks.sort(key=lambda a: -len(load(a[0])["traits"][a[1]]["rows"]))           # largest first
    with Pool(int(os.environ.get("E7_PROCS", 4))) as pool:
        res = pool.map(boot, tasks, chunksize=1)
    json.dump(res, open(os.path.join(HERE, "results", "e7_reml_boot.json"), "w"), indent=1)
    for r in res: print(f"{r['dataset']:7s} {r['trait'][:30]:30s} gamma {r['gamma']:.2f} [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}] bound {r['at_bound']:.2f}")
    print("DONE")
