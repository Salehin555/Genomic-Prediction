"""Numerical check of the local gap formula (theory plan, T2):
Delta(gamma) = min_delta (1/n) sum (tau_i^2 + s2) (lam_i/(lam_i+delta) - w_i(gamma))^2,  tau_i^2 = c lam_i^gamma,
predicted   ~ (gamma-1)^2 * (1/n) * min_b sum omega_i (log lam_i - b)^2,  omega_i = (tau_i^2+s2) w_i^2 (1-w_i)^2 at gamma=1."""
import numpy as np
from scipy.optimize import minimize_scalar
rng = np.random.default_rng(0); n = 300
lam = np.sort(rng.gamma(0.7, 1.0, n))[::-1]; lam /= lam.mean(); c, s2 = 1.0, 0.8

def gap(g):
    tau2 = c * lam ** g; w = tau2 / (tau2 + s2)
    f = lambda ld: np.mean((tau2 + s2) * (lam / (lam + np.exp(ld)) - w) ** 2)
    return minimize_scalar(f, bounds=(-15, 15), method="bounded", options={"xatol": 1e-12}).fun

w1 = c * lam / (c * lam + s2); om = (c * lam + s2) * w1 ** 2 * (1 - w1) ** 2; L = np.log(lam)
b = np.sum(om * L) / np.sum(om); A = np.mean(om * (L - b) ** 2)
print("A (curvature) =", A, "  gap(1) =", gap(1.0))
for e in (0.4, 0.2, 0.1, 0.05, 0.02):
    for g in (1 - e, 1 + e):
        print(f"gamma={g:5.2f}  exact={gap(g):.3e}  predicted={A*(g-1)**2:.3e}  ratio={gap(g)/(A*(g-1)**2):.4f}")
