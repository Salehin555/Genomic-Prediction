"""Go/no-go simulation: when can a non-uniform spectral filter beat rrBLUP for NEW individuals?

Population model (signal generated from the kernel, so it can transfer to new individuals):
  markers x ~ N(0, diag(mu)), mu_j = 1/j, j = 1..P;   genetic value g = x' beta,
  'gamma' scenarios: Var(beta_j) ~ mu_j^(gamma-1), i.e. direction j carries genetic variance ~ mu_j^gamma.
                     gamma = 1 is the GBLUP/rrBLUP model, under which rrBLUP is the Bayes-optimal predictor.
  'spiked' scenario: three population-structure directions (mu = 6, 4, 3) carry half of the genetic variance,
                     the rest follows gamma = 1.
  y = sqrt(h2) g + sqrt(1 - h2) e;  accuracy is measured on N_TEST new individuals from the same population.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import json, time, numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
P, N_TEST = 2000, 1000
SCENARIOS = ("g0.25", "g0.5", "g1", "g1.5", "g2", "g3", "spiked")


def population(scn):
    mu = 1.0 / np.arange(1, P + 1)
    if scn == "spiked":
        mu = mu.copy(); mu[:3] = [6.0, 4.0, 3.0]
        v = mu / mu.sum() * 0.5; v[:3] += 0.5 / 3
    else:
        v = mu ** float(scn[1:]); v = v / v.sum()
    return mu, v                                                   # v_j: share of genetic variance in direction j


def run(cfg):
    from spectral_filters import fit_predict, METHODS
    scn, n, h2, rep = cfg["scn"], cfg["n"], cfg["h2"], cfg["rep"]
    rng = np.random.default_rng(cfg["seed"])
    mu, v = population(scn)
    X = rng.standard_normal((n + N_TEST, P)) * np.sqrt(mu)
    bvar = v / mu
    g = X @ (rng.standard_normal(P) * np.sqrt(bvar))
    y = np.sqrt(h2) * g + np.sqrt(1 - h2) * rng.standard_normal(n + N_TEST)
    tr, te = np.arange(n), np.arange(n, n + N_TEST)
    K = X @ X.T / P
    out = dict(cfg, r2={}, info={})
    sst = np.sum((y[te] - y[te].mean()) ** 2)
    C = (X * (bvar * h2)) @ X[tr].T                                # oracle: true Bayes predictor under the model
    p_or = C[te] @ np.linalg.solve(C[tr] + (1 - h2) * np.eye(n), y[tr])
    out["r2"]["oracle"] = float(1 - np.sum((y[te] - p_or) ** 2) / sst)
    for m in METHODS:
        p, inf = fit_predict(K[np.ix_(tr, tr)], y[tr], K[np.ix_(te, tr)], m, seed=rep)
        out["r2"][m] = float(1 - np.sum((y[te] - p) ** 2) / sst); out["info"][m] = inf
    return out


if __name__ == "__main__":
    cf = [dict(scn=s, n=n, h2=h2, rep=r) for s in SCENARIOS for n in (100, 200, 400, 800)
          for h2 in (0.3, 0.7) for r in range(20)]
    for i, c in enumerate(cf): c["seed"] = 10000 + i                # reproducible, fixed before sorting
    cf.sort(key=lambda c: -c["n"])                                 # big jobs first for load balance
    t0 = time.time(); res = []
    print(len(cf), "tasks", flush=True)
    with Pool(10) as pool:
        for i, r in enumerate(pool.imap_unordered(run, cf)):
            res.append(r)
            if (i + 1) % 80 == 0:
                print(f"{i + 1}/{len(cf)} ({time.time() - t0:.0f}s)", flush=True)
    json.dump(res, open(os.path.join(HERE, "results", "sim_regimes.json"), "w"))
    print("DONE", round(time.time() - t0), "s")
