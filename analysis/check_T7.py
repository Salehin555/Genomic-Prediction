"""Numerical checks of T7: (a) CV(w) equals the exact leave-one-out error of kernel ridge regression with the fixed
kernel K_w = U diag(w/(1-w)) U' and unit penalty; (b) max_i (S_w)_ii <= max_k w_k."""
import numpy as np
rng = np.random.default_rng(0); n = 60
A = rng.standard_normal((n, 200)); G = A @ A.T / 200
lam, U = np.linalg.eigh(G); y = rng.standard_normal(n)
for trial in range(3):
    w = rng.uniform(0, 0.97, n)
    S = U @ np.diag(w) @ U.T
    short = (y - S @ y) / (1 - np.diag(S))
    Kw = U @ np.diag(w / (1 - w)) @ U.T
    exact = np.empty(n)
    for i in range(n):
        o = np.delete(np.arange(n), i)
        a = np.linalg.solve(Kw[np.ix_(o, o)] + np.eye(n - 1), y[o])
        exact[i] = y[i] - Kw[i, o] @ a
    print(f"trial {trial}: max |shortcut - refit LOO residual| = {np.max(np.abs(short - exact)):.2e};"
          f"  max S_ii = {np.diag(S).max():.4f} <= max w = {w.max():.4f}")
