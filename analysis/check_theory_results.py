"""Numerical checks of the results of Section 3:
Lemma 1 (oracle weights minimise R), Proposition 1 (exact excess-risk identity), Theorem 1 (i) nonnegativity,
(ii) two-sided dispersion bound for the constant family, (iii) zero rrBLUP gap iff theta^2 proportional to lambda,
and the worked instance of Remark 3 (n = 2: best constant 5/7, Delta_u = 9/140 = 0.064; unweighted mean 0.65 gives 0.07875)."""
import numpy as np
from scipy.optimize import minimize_scalar
rng = np.random.default_rng(0)
R = lambda w, th2, s2: np.mean((1 - w) ** 2 * th2 + w ** 2 * s2)
worst = dict(lemma=0.0, identity=0.0, disp_lo=np.inf, disp_hi=np.inf)
for t in range(2000):
    n = rng.integers(2, 60); lam = np.sort(rng.gamma(0.8, 1, n))[::-1] + 1e-3
    th2 = rng.gamma(0.5, 2, n); s2 = rng.uniform(0.1, 3)
    ws = th2 / (th2 + s2); w = rng.uniform(0, 1, n)
    worst["lemma"] = max(worst["lemma"], R(ws, th2, s2) - R(w, th2, s2))                      # must be <= 0
    worst["identity"] = max(worst["identity"], abs(R(w, th2, s2) - R(ws, th2, s2) - np.mean((th2 + s2) * (w - ws) ** 2)))
    du = minimize_scalar(lambda c: R(np.full(n, c), th2, s2), bounds=(0, 1), method="bounded").fun - R(ws, th2, s2)
    worst["disp_lo"] = min(worst["disp_lo"], du - s2 * ws.var())                              # must be >= 0
    worst["disp_hi"] = min(worst["disp_hi"], (th2.max() + s2) * ws.var() - du)               # must be >= 0
print("Lemma 1: max R(w*) - R(w) over random w =", worst["lemma"], "(<= 0 required)")
print("Prop 1: max |identity error| =", worst["identity"])
print("Thm 1(ii): min lower-bound slack =", worst["disp_lo"], "; min upper-bound slack =", worst["disp_hi"], "(>= 0 required, up to optimiser tolerance)")
lam = np.sort(rng.gamma(0.8, 1, 40))[::-1]; s2 = 0.7
for kappa in (0.5, 2.0):
    th2 = kappa * lam; f = lambda ld: R(lam / (lam + np.exp(ld)), th2, s2) - R(th2 / (th2 + s2), th2, s2)
    r = minimize_scalar(f, bounds=(-10, 10), method="bounded", options={"xatol": 1e-12})
    print(f"Thm 1(iii): theta^2 = {kappa} lambda -> min rrBLUP gap = {r.fun:.2e}, optimal lambda = {np.exp(r.x):.4f} (sigma^2/kappa = {s2/kappa:.4f})")
th2 = lam * np.exp(rng.normal(0, 0.5, 40)); f = lambda ld: R(lam / (lam + np.exp(ld)), th2, s2) - R(th2 / (th2 + s2), th2, s2)
print("Thm 1(iii): non-proportional theta^2 -> gap =", round(minimize_scalar(f, bounds=(-10, 10), method="bounded").fun, 5), "(> 0 required)")
th2 = np.array([4.0, 1.0]); ws = th2 / (th2 + 1)
cb = minimize_scalar(lambda c: R(np.full(2, c), th2, 1.0), bounds=(0, 1), method="bounded", options={"xatol": 1e-12}).x
print("Remark 3 (worked instance): w* =", ws, " Var =", ws.var(), " best constant =", round(cb, 6), "(5/7 =", round(5 / 7, 6), ")")
print("  Delta_u at the best constant =", round(R(np.full(2, cb), th2, 1.0) - R(ws, th2, 1.0), 6), "(paper: 9/140 =", round(9 / 140, 6), ")")
print("  Delta_u at the unweighted mean 0.65 =", round(R(np.full(2, ws.mean()), th2, 1.0) - R(ws, th2, 1.0), 6), "(paper: 0.07875)")
