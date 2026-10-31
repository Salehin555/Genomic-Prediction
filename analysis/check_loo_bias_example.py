"""Remark 10: a non-degenerate example in which the minimizers of the expected leave-one-out risk R-bar and of the
in-sample risk R differ for the power family (n = 300 eigenvalues uniform on [0.3, 3], rescaled to mean one, random
eigenvectors, sigma^2 = 1, theta_i^2 = h * lambda_i^gamma). Writes results/loo_bias_example.json.
usage: python check_loo_bias_example.py"""
import json, os
import numpy as np
from scipy import linalg, optimize
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
rng = np.random.default_rng(7)


def make(n, gam, h):
    lam = np.sort(rng.uniform(0.3, 3, n))[::-1]; lam /= lam.mean()
    U = linalg.qr(rng.normal(size=(n, n)))[0]
    th2 = h * lam ** gam; m = U @ (np.sqrt(th2) * rng.choice([-1, 1], n)); return lam, U, m


def crit(par, lam, U, m, s2):
    a, ld = par; d = np.exp(ld); w = lam ** a / (lam ** a + d); S = (U * w) @ U.T; M = np.eye(len(lam)) - S; si = np.diag(S)
    Q = (M * (1 / (1 - si) ** 2)) @ M; n = len(lam)
    return (s2 * np.trace(Q) + m @ Q @ m) / n, (np.sum((M @ m) ** 2) + s2 * np.sum(w ** 2)) / n


out = []
for n, gam, h in ((300, 1.0, 3.0), (300, 2.0, 3.0), (300, 0.5, 5.0), (600, 2.0, 5.0)):
    lam, U, m = make(n, gam, h); s2 = 1.0
    rb = optimize.minimize(lambda p: crit(p, lam, U, m, s2)[0], x0=[1.0, 0.0], bounds=[(0.1, 6), (-6, 6)])
    r = optimize.minimize(lambda p: crit(p, lam, U, m, s2)[1], x0=[1.0, 0.0], bounds=[(0.1, 6), (-6, 6)])
    ex = crit(rb.x, lam, U, m, s2)[1] - crit(r.x, lam, U, m, s2)[1]
    out.append(dict(n=n, gamma=gam, h=h, a_rbar=rb.x[0], delta_rbar=float(np.exp(rb.x[1])), a_r=r.x[0],
                    delta_r=float(np.exp(r.x[1])), excess=float(ex)))
    print(out[-1])
json.dump(out, open(os.path.join(RES, "loo_bias_example.json"), "w"), indent=1)
