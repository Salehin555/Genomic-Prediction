"""Marker-based Bayesian regression baselines, single-site Gibbs samplers (numba).

  BRR    : beta_j ~ N(0, s2b)                                   (Bayesian ridge regression)
  BayesC : beta_j ~ (1-pi) delta_0 + pi N(0, s2b), pi ~ Beta(1,1)   (Habier et al. 2011)
  BayesR : beta_j ~ sum_k pi_k N(0, g_k s2g), g = (0, 1e-4, 1e-3, 1e-2), pi ~ Dirichlet(1,1,1,1)  (Erbe et al. 2012)

Priors follow common BGLR-style defaults: scaled-inverse-chi2 with nu = 5 for every variance, scales set so that the
prior mode corresponds to h2 = 0.5. Markers are standardised; y is centred inside fit(). The posterior mean of beta
over post-burn-in iterations is used for prediction.
"""
import numpy as np
from numba import njit

NU = 5.0
GAMMA_R = np.array([0.0, 1e-4, 1e-3, 1e-2])


@njit(cache=True)
def _dot(X, j, e):
    s = 0.0
    for i in range(X.shape[0]): s += X[i, j] * e[i]
    return s


@njit(cache=True)
def _axpy(X, j, a, e):
    for i in range(X.shape[0]): e[i] -= a * X[i, j]


@njit(cache=True)
def _scaled_inv_chi2(ss, df):
    return ss / np.random.chisquare(df)


@njit(cache=True)
def _gibbs(X, y, model, n_iter, burn, seed, h2p):
    np.random.seed(seed)
    n, p = X.shape
    xx = np.empty(p)
    for j in range(p):
        s = 0.0
        for i in range(n): s += X[i, j] * X[i, j]
        xx[j] = s
    vy = np.var(y); sumvar = xx.sum() / n
    e = y.copy(); beta = np.zeros(p); post = np.zeros(p)
    s2e = vy * (1 - h2p); Se = s2e * (NU - 2) / NU
    pi1 = 0.5 if model == 1 else 1.0
    s2b = vy * h2p / (sumvar * pi1); Sb = s2b * (NU - 2) / NU          # BRR / BayesC
    s2g = vy * h2p; Sg = s2g * (NU - 2) / NU                          # BayesR genetic variance (std. markers)
    piR = np.array([0.95, 0.03, 0.015, 0.005]); cls = np.zeros(p, np.int64)
    trace = np.zeros(n_iter)
    for it in range(n_iter):
        nz = 0; ssb = 0.0; cnt = np.zeros(4)
        for j in range(p):
            bj = beta[j]
            rhs = _dot(X, j, e) + xx[j] * bj
            if model == 0:                                             # BRR
                lhs = xx[j] + s2e / s2b
                nb = rhs / lhs + np.random.normal() * np.sqrt(s2e / lhs)
                nz += 1; ssb += nb * nb
            elif model == 1:                                           # BayesC
                v0 = xx[j] * s2e; v1 = xx[j] * xx[j] * s2b + v0
                lo = np.log(pi1) - np.log(1 - pi1) + 0.5 * (np.log(v0) - np.log(v1)) + 0.5 * rhs * rhs * (1 / v0 - 1 / v1)
                if np.random.random() < 1.0 / (1.0 + np.exp(-lo)):
                    lhs = xx[j] + s2e / s2b
                    nb = rhs / lhs + np.random.normal() * np.sqrt(s2e / lhs)
                    nz += 1; ssb += nb * nb
                else:
                    nb = 0.0
            else:                                                      # BayesR
                lp = np.empty(4)
                for k in range(4):
                    v = xx[j] * s2e + (xx[j] * xx[j] * GAMMA_R[k] * s2g if k > 0 else 0.0)
                    lp[k] = np.log(piR[k]) - 0.5 * np.log(v) - 0.5 * rhs * rhs / v
                m = lp.max(); pr = np.exp(lp - m); pr /= pr.sum()
                u = np.random.random(); k = 0; c = pr[0]
                while u > c and k < 3:
                    k += 1; c += pr[k]
                cls[j] = k; cnt[k] += 1
                if k > 0:
                    lhs = xx[j] + s2e / (GAMMA_R[k] * s2g)
                    nb = rhs / lhs + np.random.normal() * np.sqrt(s2e / lhs)
                    nz += 1; ssb += nb * nb / GAMMA_R[k]
                else:
                    nb = 0.0
            if nb != bj:
                _axpy(X, j, nb - bj, e); beta[j] = nb
        ee = 0.0
        for i in range(n): ee += e[i] * e[i]
        s2e = _scaled_inv_chi2(ee + NU * Se, n + NU)
        if model == 2:
            s2g = _scaled_inv_chi2(ssb + NU * Sg, nz + NU)
            g = np.empty(4)
            for k in range(4): g[k] = np.random.gamma(cnt[k] + 1.0, 1.0)
            piR = g / g.sum()
        else:
            s2b = _scaled_inv_chi2(ssb + NU * Sb, nz + NU)
            if model == 1:
                pi1 = np.random.beta(nz + 1.0, p - nz + 1.0)
        trace[it] = s2e
        if it >= burn:
            post += beta
    return post / (n_iter - burn), trace


MODELS = {"BRR": 0, "BayesC": 1, "BayesR": 2}


class GibbsRegressor:
    def __init__(self, model="BayesC", n_iter=3000, burn=1000, seed=0, h2_prior=0.5):
        self.model, self.n_iter, self.burn, self.seed, self.h2p = model, n_iter, burn, seed, h2_prior

    def fit(self, X, y):
        self.mu_ = float(y.mean())
        Xf = np.asfortranarray(X, dtype=np.float64)
        self.beta_, self.trace_ = _gibbs(Xf, y - self.mu_, MODELS[self.model], self.n_iter, self.burn, self.seed, self.h2p)
        return self

    def predict(self, X):
        return X @ self.beta_ + self.mu_


def _bm_var(x, nb=10):
    """Variance of the mean by batch means (accounts for autocorrelation)."""
    m = len(x) // nb; bm = x[: m * nb].reshape(nb, m).mean(1)
    return bm.var(ddof=1) / nb


def geweke_z(trace, burn, a=0.1, b=0.5):
    """Geweke diagnostic on the post-burn-in trace, with batch-means variances."""
    t = trace[burn:]; n = len(t); A, B = t[: int(a * n)], t[int((1 - b) * n):]
    return float((A.mean() - B.mean()) / np.sqrt(_bm_var(A) + _bm_var(B) + 1e-300))
