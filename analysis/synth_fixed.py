"""Synthetic benchmark exactly as specified in Section 4.2 / Algorithm 3 of the manuscript, with the corrected
SAS-BLUP (bounded leverage + inner-CV collapse guard) and the exact spectral gap (noise sigma^2 per coordinate).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import json, math, time, numpy as np
from multiprocessing import Pool
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

W = os.path.join(os.path.dirname(os.path.abspath(__file__)), "work")
FAMS = ("scalar", "grouped", "smooth", "monotone", "spline")
LAM = np.logspace(-5, 5, 400)


def generate(n, decay, par, align, snr, seed, s=1.0):
    rng = np.random.default_rng(seed)
    Q, Rm = np.linalg.qr(rng.standard_normal((n, n)))
    U = Q * np.sign(np.diag(Rm))                                    # Haar-distributed orthonormal matrix
    i = np.arange(1, n + 1)
    lam = i ** (-par) if decay == "poly" else np.exp(-par * (i - 1) / n)
    lam = lam / lam.sum() * n
    m = math.ceil(n / 10)
    if align == "top": A = np.arange(0, m)
    elif align == "middle":
        A = np.arange(n // 2 - m // 2, n // 2 + math.ceil(m / 2) + 1) - 1   # Eq. (24), 1-based -> 0-based
    elif align == "bottom": A = np.arange(n - m, n)
    else: A = np.arange(n)
    th2 = np.zeros(n); th2[A] = s
    theta = np.sqrt(th2) * rng.choice([-1.0, 1.0], n)
    sigma2 = th2.sum() / (n * snr)
    y = U @ theta + rng.standard_normal(n) * np.sqrt(sigma2)
    K = (U * lam) @ U.T
    return K, y, theta, sigma2, lam


def exact_gap(lam, theta, sigma2):
    th2 = theta ** 2
    w_or = th2 / (th2 + sigma2)
    r_or = np.mean((1 - w_or) ** 2 * th2 + w_or ** 2 * sigma2)
    Wr = lam[None, :] / (lam[None, :] + LAM[:, None])
    rr = np.mean((1 - Wr) ** 2 * th2[None, :] + Wr ** 2 * sigma2, axis=1)
    return float(rr.min() - r_or), float(rr.min()), float(r_or)


def run_config(cfg):
    import sasblup_engine as E, sasblup_fixed as F
    n, decay, par, align, snr, seed = cfg["n"], cfg["decay"], cfg["par"], cfg["align"], cfg["snr"], cfg["seed"]
    K, y, theta, sigma2, lam = generate(n, decay, par, align, snr, seed)
    G = K + 1e-4 * np.eye(n)
    E.EFF_RANK = int(lam.sum() ** 2 / (lam ** 2).sum())
    gap, r_ridge, r_or = exact_gap(lam, theta, sigma2)
    r2 = {f: [] for f in FAMS}; guard = {f: [] for f in FAMS if f != "scalar"}; prof = {f: [] for f in FAMS}
    for tr, va in KFold(5, shuffle=True, random_state=seed).split(y):
        sc = StandardScaler(); yt = sc.fit_transform(y[tr].reshape(-1, 1)).ravel()
        for f in FAMS:
            est = F.SASBLUPFixed(strategy=f).fit(G[np.ix_(tr, tr)], yt)
            p = sc.inverse_transform(est.predict(G[np.ix_(va, tr)]).reshape(-1, 1)).ravel()
            r2[f].append(float(r2_score(y[va], p)))
            if f != "scalar": guard[f].append(bool(est.guard_used_))
            if cfg.get("keep_profiles"): prof[f].append(est.weights_.tolist())
    out = dict(cfg, gap=gap, ridge_risk=r_ridge, oracle_risk=r_or, eff_rank=E.EFF_RANK,
               r2={f: float(np.mean(v)) for f, v in r2.items()},
               r2_folds=r2, guard={f: float(np.mean(v)) for f, v in guard.items()})
    if cfg.get("keep_profiles"):
        th2 = theta ** 2
        out["oracle"] = (th2 / (th2 + sigma2)).tolist(); out["profiles"] = prof; out["lam"] = lam.tolist()
    return out


def configs():
    cf = []
    # main grid (Figure 3, Table 3): 4 alignments x 3 SNR x n in {80,150} x 10 seeds, polynomial decay a = 1
    for align in ("top", "middle", "uniform", "bottom"):
        for snr in (0.25, 1.0, 4.0):
            for n in (80, 150):
                for seed in range(10):
                    cf.append(dict(exp="grid", n=n, decay="poly", par=1.0, align=align, snr=snr, seed=seed,
                                   keep_profiles=(n == 150 and snr == 1.0 and seed == 0)))
    # 50 random configurations (Figure 5)
    rng = np.random.default_rng(7)
    for k in range(50):
        decay = str(rng.choice(["poly", "exp"]))
        par = float(rng.choice([0.5, 1.0, 2.0])) if decay == "poly" else float(rng.choice([2.0, 5.0, 10.0]))
        cf.append(dict(exp="random", n=int(rng.choice([80, 150])), decay=decay, par=par,
                       align=str(rng.choice(["top", "middle", "uniform", "bottom"])),
                       snr=float(rng.choice([0.25, 1.0, 4.0])), seed=1000 + k))
    # sample-size sweep (Figure 6A): n = 80..150, SNR = 1, polynomial a = 1, 5 seeds
    for align in ("top", "middle", "uniform", "bottom"):
        for n in range(80, 151, 10):
            for seed in range(5):
                cf.append(dict(exp="sweep", n=n, decay="poly", par=1.0, align=align, snr=1.0, seed=2000 + seed))
    return cf


def _key(c):
    return "|".join(str(c[k]) for k in ("exp", "n", "decay", "par", "align", "snr", "seed"))


if __name__ == "__main__":
    cf = configs(); t0 = time.time()
    part = os.path.join(W, "fixed_synth.partial.json")
    res = json.load(open(part)) if os.path.exists(part) else []
    done = {_key(r) for r in res}
    todo = [c for c in cf if _key(c) not in done]
    print(len(cf), "configurations,", len(todo), "to run", flush=True)
    with Pool(10) as pool:
        for i, r in enumerate(pool.imap_unordered(run_config, todo, chunksize=2)):
            res.append(r)
            if (i + 1) % 25 == 0 or i + 1 == len(todo):
                json.dump(res, open(part, "w"))
                print(f"{len(res)}/{len(cf)} done ({time.time() - t0:.0f}s)", flush=True)
    json.dump(res, open(os.path.join(W, "fixed_synth.json"), "w"))
    print("DONE", round(time.time() - t0), "s")
