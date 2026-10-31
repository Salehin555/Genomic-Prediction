"""E2 + E3: semi-synthetic phenotypes on the real rice, wheat and mouse GRMs.

usage: python e3_semisynth.py regimes REPS     -> E3: does the regime map hold on real LD and structure?
       python e3_semisynth.py size REPS        -> E2: size of the gamma = 1 test (chi2 and parametric bootstrap)

Phenotypes: g = U diag(sqrt(v)) xi, xi ~ N(0, I), with v_i ~ lam_i^gamma (normalised to total h2), e ~ N(0, 1 - h2);
'structure' puts half of the genetic variance on the three leading eigenvectors and the rest at gamma = 1.
E3 evaluates one 5-fold split per replicate (held-out individuals of the same panel).
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import json, time, numpy as np
from multiprocessing import Pool
from sklearn.model_selection import KFold

HERE = os.path.dirname(os.path.abspath(__file__))
_G = {}


def eig(name):
    if name not in _G:
        import datasets
        G = datasets.LOADERS[name]()["G"] if name != "rice" else datasets.rice(markers=False)["G"]
        ev, U = np.linalg.eigh(G); _G[name] = (G, np.maximum(ev, 1e-10), U)
    return _G[name]


def phenotype(name, scn, h2, rng):
    G, ev, U = eig(name); lam = ev / ev.mean(); n = len(ev)
    if scn == "structure":
        v = lam / lam.sum() * 0.5; v[-3:] += 0.5 / 3                      # eigh is ascending: leading = last
    else:
        v = lam ** float(scn[1:]); v = v / v.sum()
    g = U @ (np.sqrt(v * h2 * n) * rng.standard_normal(n))
    return g + np.sqrt(1 - h2) * rng.standard_normal(n)


def run_regime(cfg):
    from spectral_filters import fit_predict, gamma_test
    G, _, _ = eig(cfg["data"]); rng = np.random.default_rng(cfg["seed"])
    y = phenotype(cfg["data"], cfg["scn"], cfg["h2"], rng)
    out = dict(cfg, r2={}, test=gamma_test(G, y))
    M = ("rrBLUP_CV", "power_CV", "safe_CV", "gammaBLUP_ML")
    sse = {m: 0.0 for m in M}; sst = 0.0
    for tr, va in KFold(5, shuffle=True, random_state=cfg["rep"]).split(y):
        sst += np.sum((y[va] - y[va].mean()) ** 2)
        for m in M:
            p, _ = fit_predict(G[np.ix_(tr, tr)], y[tr], G[np.ix_(va, tr)], m, seed=cfg["rep"])
            sse[m] += np.sum((y[va] - p) ** 2)
    out["r2"] = {m: float(1 - sse[m] / sst) for m in M}
    return out


def run_size(cfg):
    from spectral_filters import ml_fit
    from scipy.stats import chi2
    G, ev, U = eig(cfg["data"]); rng = np.random.default_rng(cfg["seed"]); lam = ev / ev.mean()
    y = phenotype(cfg["data"], cfg["scn"], cfg["h2"], rng)
    z2 = (U.T @ ((y - y.mean()) / y.std())) ** 2
    f1 = ml_fit(lam, z2)[3]; c0, s0, _, f0 = ml_fit(lam, z2, gamma=1.0); lr = max(2 * (f0 - f1), 0.0)
    lrb = []
    for _ in range(cfg["B"]):
        zb2 = rng.standard_normal(len(lam)) ** 2 * (c0 * lam + s0)
        lrb.append(max(2 * (ml_fit(lam, zb2, gamma=1.0)[3] - ml_fit(lam, zb2)[3]), 0.0))
    return dict(cfg, lr=float(lr), p_chi2=float(chi2.sf(lr, 1)), p_boot=float((1 + np.sum(np.array(lrb) >= lr)) / (cfg["B"] + 1)))


if __name__ == "__main__":
    mode, reps = sys.argv[1], int(sys.argv[2]); t0 = time.time()
    if mode == "regimes":
        cf = [dict(data=d, scn=s, h2=h, rep=r) for d in ("mice", "wheat", "rice")
              for s in ("g0.5", "g1", "g2", "g3", "structure") for h in (0.3, 0.6) for r in range(reps)]
        fn = run_regime
    else:
        cf = [dict(data=d, scn=s, h2=0.5, rep=r, B=(100 if s == "g1" else 0)) for d in ("mice", "wheat", "rice")
              for s in ("g1", "g0.75", "g1.25", "g1.5", "g2") for r in range(reps)]
        fn = run_size
    for i, c in enumerate(cf): c["seed"] = 50000 + i
    res = []
    with Pool(11) as pool:
        for i, r in enumerate(pool.imap_unordered(fn, cf, chunksize=2)):
            res.append(r)
            if (i + 1) % 100 == 0: print(f"{i + 1}/{len(cf)} ({time.time() - t0:.0f}s)", flush=True)
    json.dump(res, open(os.path.join(HERE, "results", f"e3_{mode}.json"), "w")); print("DONE", round(time.time() - t0), "s")
