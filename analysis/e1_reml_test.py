"""E1 rejection rates with the REML version of the gamma = 1 test (e13_reml_diag.py).

Re-creates the training data of every E1 task exactly (same seed and generation order as e1_regimes.run) and applies the
REML test (fixed effects = intercept) with the chi2 p-value. Output: results/e1_reml_test.json, one record per task
(scn, n, h2, rep, seed, gamma, p_chi2), to replace the rejection rates of the original test in the E1 tables.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import json, time, numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cfg):
    from e1_regimes import population, P, N_TEST
    from e13_reml_diag import reml_basis, reml_test
    rng = np.random.default_rng(cfg["seed"]); n, h2 = cfg["n"], cfg["h2"]
    mu, v = population(cfg["scn"]); bvar = v / mu
    X = rng.standard_normal((n + N_TEST, P)) * np.sqrt(mu)
    g = np.sqrt(h2) * (X @ (rng.standard_normal(P) * np.sqrt(bvar)))
    y = g + np.sqrt(1 - h2) * rng.standard_normal(n + N_TEST)
    K = X[:n] @ X[:n].T / P
    lam, A = reml_basis(K, np.ones((n, 1))); r = reml_test(lam, A, y[:n], B=0, seed=0)
    return dict(scn=cfg["scn"], n=n, h2=h2, rep=cfg["rep"], seed=cfg["seed"], gamma=r["gamma"], p_chi2=r["p_chi2"])


if __name__ == "__main__":
    cf = json.load(open(os.path.join(HERE, "results", "e1_regimes.json")))
    cf = sorted(({k: c[k] for k in ("scn", "n", "h2", "rep", "seed")} for c in cf), key=lambda c: -c["n"])
    t0 = time.time()
    with Pool(int(os.environ.get("E1_PROCS", 4))) as pool:
        res = pool.map(run, cf, chunksize=4)
    json.dump(res, open(os.path.join(HERE, "results", "e1_reml_test.json"), "w")); print("DONE", round(time.time() - t0), "s")
