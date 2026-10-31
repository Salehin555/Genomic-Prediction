"""Size check of the gamma = 1 test on the rice GRM under an exact gamma = 1 null.
Compares (a) the test as implemented (y centred, all eigen-directions kept) with (b) the same test after dropping the
eigen-direction aligned with the all-ones vector (centring sets its coordinate to exactly 0, which the parametric
bootstrap does not reproduce)."""
import sys, numpy as np, datasets
from scipy.stats import chi2
from spectral_filters import ml_fit

def lrt(lam, z2):
    f1 = ml_fit(lam, z2)[3]; c0, s0, _, f0 = ml_fit(lam, z2, gamma=1.0); return max(2 * (f0 - f1), 0.0), c0, s0

def test(lam, U, y, rng, B, keep):
    z2 = (U.T @ ((y - y.mean()) / y.std())) ** 2; lam, z2 = lam[keep], z2[keep]
    lr, c0, s0 = lrt(lam, z2); lrb = []
    for _ in range(B):
        zb2 = rng.standard_normal(len(lam)) ** 2 * (c0 * lam + s0); lrb.append(lrt(lam, zb2)[0])
    return chi2.sf(lr, 1), (1 + np.sum(np.array(lrb) >= lr)) / (B + 1)

reps, B = int(sys.argv[1]), int(sys.argv[2])
G = datasets.rice(markers=False)["G"]; ev, U = np.linalg.eigh(G); ev = np.maximum(ev, 1e-10); lam = ev / ev.mean(); n = len(ev)
one = np.abs(U.T @ np.ones(n)) / np.sqrt(n); k1 = int(np.argmax(one)); print("ones-direction", k1, "alignment", one[k1].round(4), "lam", lam[k1])
keep_all = np.ones(n, bool); keep_drop = keep_all.copy(); keep_drop[k1] = False
rng = np.random.default_rng(7); P = {"all": [], "drop": []}
for r in range(reps):
    v = lam / lam.sum(); y = U @ (np.sqrt(v * 0.5 * n) * rng.standard_normal(n)) + np.sqrt(0.5) * rng.standard_normal(n)
    P["all"].append(test(lam, U, y, rng, B, keep_all)); P["drop"].append(test(lam, U, y, rng, B, keep_drop))
for k, v in P.items():
    v = np.array(v); print(k, "size chi2 @.05", (v[:, 0] < .05).mean().round(3), " boot @.05", (v[:, 1] < .05).mean().round(3), "@.10", (v[:, 1] < .10).mean().round(3))
