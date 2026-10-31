"""E1 (final regime map) with the T1/T6 checks and the in-sample diagnostic of E2.

Population model as in sim_regimes.py (markers x ~ N(0, diag(mu)), mu_j = 1/j, P = 2000; direction j carries genetic
variance ~ mu_j^gamma; 'spiked' = three structure axes carrying half of the genetic variance). For each task we record:
  r2[m]        held-out R2 on N_TEST new individuals, for every method in spectral_filters.ALL_METHODS and the oracle
  mse_g[m]     mean squared error against the true genetic value of the new individuals
  cross[m]     mean of (f - oracle)(oracle - g): T1 says its expectation is 0 because the oracle is the Bayes predictor
  test         the gamma = 1 likelihood-ratio diagnostic computed on the training data only
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import json, sys, time, numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
P, N_TEST = 2000, 1000
SCENARIOS = ("g0.5", "g0.75", "g1", "g1.25", "g1.5", "g2", "g3", "spiked")


def population(scn):
    mu = 1.0 / np.arange(1, P + 1)
    if scn == "spiked":
        mu = mu.copy(); mu[:3] = [6.0, 4.0, 3.0]
        v = mu / mu.sum() * 0.5; v[:3] += 0.5 / 3
    else:
        v = mu ** float(scn[1:]); v = v / v.sum()
    return mu, v


def run(cfg):
    from spectral_filters import fit_predict, gamma_test, ALL_METHODS
    rng = np.random.default_rng(cfg["seed"]); n, h2 = cfg["n"], cfg["h2"]
    mu, v = population(cfg["scn"]); bvar = v / mu
    X = rng.standard_normal((n + N_TEST, P)) * np.sqrt(mu)
    g = np.sqrt(h2) * (X @ (rng.standard_normal(P) * np.sqrt(bvar)))
    y = g + np.sqrt(1 - h2) * rng.standard_normal(n + N_TEST)
    tr, te = np.arange(n), np.arange(n, n + N_TEST)
    K = X @ X.T / P
    C = (X * (bvar * h2)) @ X[tr].T
    preds = {"oracle": C[te] @ np.linalg.solve(C[tr] + (1 - h2) * np.eye(n), y[tr])}
    info = {}
    for m in ALL_METHODS:
        preds[m], info[m] = fit_predict(K[np.ix_(tr, tr)], y[tr], K[np.ix_(te, tr)], m, seed=cfg["rep"])
    sst = np.sum((y[te] - y[te].mean()) ** 2); orc = preds["oracle"]
    out = dict(cfg, r2={}, mse_g={}, cross={}, info={k: {kk: vv for kk, vv in d.items() if kk != "err"} for k, d in info.items()})
    for m, p in preds.items():
        out["r2"][m] = float(1 - np.sum((y[te] - p) ** 2) / sst)
        out["mse_g"][m] = float(np.mean((p - g[te]) ** 2))
        out["cross"][m] = float(np.mean((p - orc) * (orc - g[te])))
    out["test"] = gamma_test(K[np.ix_(tr, tr)], y[tr])
    return out


if __name__ == "__main__":
    # usage: python e1_regimes.py REPS [REPS_N1600]; resumes from results/e1_regimes.partial.json if present
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    reps_big = int(sys.argv[2]) if len(sys.argv) > 2 else reps
    cf = [dict(scn=s, n=n, h2=h2, rep=r) for s in SCENARIOS for n in (100, 200, 400, 800, 1600)
          for h2 in (0.3, 0.5, 0.7) for r in range(reps)]
    for i, c in enumerate(cf): c["seed"] = 20000 + i
    part = os.path.join(HERE, "results", "e1_regimes.partial.json")
    res = json.load(open(part)) if os.path.exists(part) else []
    done = {r["seed"] for r in res}
    cf = [c for c in cf if c["seed"] not in done and (c["n"] < 1600 or c["rep"] < reps_big)]
    cf.sort(key=lambda c: -c["n"])
    t0 = time.time()
    print(len(res), "done,", len(cf), "to run", flush=True)
    with Pool(11) as pool:
        for i, r in enumerate(pool.imap_unordered(run, cf)):
            res.append(r)
            if (i + 1) % 200 == 0 or i + 1 == len(cf):
                json.dump(res, open(part, "w")); print(f"{i + 1}/{len(cf)} ({time.time() - t0:.0f}s)", flush=True)
    json.dump(res, open(os.path.join(HERE, "results", "e1_regimes.json"), "w"))
    print("DONE", round(time.time() - t0), "s")
