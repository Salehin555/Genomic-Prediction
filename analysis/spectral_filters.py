"""Spectral-filter predictors on a kernel K = U diag(ev) U' fitted to training data.

All predictors have the form  y_hat(new) = mean + K(new, train) U diag(w / ev) U' (y - mean)  with weights w:
  rrBLUP_ML    w = c*lam / (c*lam + s2),          (c, s2) by maximum likelihood      (standard GBLUP/REML-type)
  gammaBLUP_ML w = c*lam^g / (c*lam^g + s2),      (c, s2, g) by maximum likelihood   (Bayes rule if Var(theta_i) ~ lam_i^g)
  rrBLUP_CV    w = lam / (lam + d),               d by inner 5-fold CV
  power_CV     w = lam^a / (lam^a + d),           (a, d) by inner 5-fold CV          (nests rrBLUP at a = 1)
  PCR_CV       w = 1 for the top k eigenvectors,  k by inner 5-fold CV
  safe_CV      rrBLUP_CV or power_CV, whichever has the lower nested 5-fold CV error (refitted per inner split)
where lam = ev / mean(ev).
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import KFold

DELTAS = np.logspace(-4, 4, 81)
POWERS = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0])
METHODS = ("rrBLUP_ML", "gammaBLUP_ML", "rrBLUP_CV", "power_CV", "PCR_CV")
ALL_METHODS = METHODS + ("safe_CV",)


def _eig(K):
    ev, U = np.linalg.eigh(K)
    return np.maximum(ev, 1e-10), U


def _nll(p, lam, z2, g=None):
    gg = p[2] if g is None else g
    v = np.exp(p[0]) * lam ** gg + np.exp(p[1])
    return 0.5 * np.sum(np.log(v) + z2 / v)


def ml_fit(lam, z2, gamma=None):
    """ML fit of z_i ~ N(0, c lam_i^g + s2); gamma=None estimates g in [0, 6]."""
    best = None
    for g0 in ([0.5, 1.0, 2.0, 3.0] if gamma is None else [None]):
        for lc in (-2.0, 0.0, 2.0):
            x0 = [lc, -1.0] + ([g0] if gamma is None else [])
            bd = [(-20, 10), (-25, 5)] + ([(0.0, 6.0)] if gamma is None else [])
            r = minimize(_nll, x0, args=(lam, z2, gamma), method="L-BFGS-B", bounds=bd)
            if best is None or r.fun < best.fun:
                best = r
    c, s2 = np.exp(best.x[0]), np.exp(best.x[1])
    return c, s2, (best.x[2] if gamma is None else gamma), best.fun


def _power_grid(ev):
    lam = ev / ev.mean()
    la = lam[None, None, :] ** POWERS[:, None, None]
    return (la / (la + DELTAS[None, :, None])).reshape(-1, len(ev))      # (len(POWERS)*len(DELTAS)) x n


def _ridge_grid(ev):
    lam = ev / ev.mean()
    return lam[None, :] / (lam[None, :] + DELTAS[:, None])


def _pcr_ks(n):
    return np.unique(np.round(np.logspace(0, np.log10(max(n - 1, 2)), 25)).astype(int))


def _pcr_grid(ev, ks):
    n = len(ev); rank = np.arange(n)                                      # eigh is ascending: top k = last k
    return np.array([(rank >= n - k).astype(float) for k in ks])


def _cv_select(K, y, grid_fn, folds=5, seed=0):
    err = None
    for itr, iva in KFold(folds, shuffle=True, random_state=seed).split(y):
        yi = y[itr]; m = yi.mean()
        ev, U = _eig(K[np.ix_(itr, itr)])
        z = U.T @ (yi - m)
        A = K[np.ix_(iva, itr)] @ U
        P = (grid_fn(ev) / ev * z) @ A.T + m
        e = ((P - y[iva]) ** 2).sum(1)
        err = e if err is None else err + e
    return int(np.argmin(err))


def fit_predict(K_tr, y_tr, K_te_tr, method, seed=0):
    """Return (predictions for the test rows, info dict)."""
    ev, U = _eig(K_tr); m = y_tr.mean(); z = U.T @ (y_tr - m); lam = ev / ev.mean(); info = {}
    if method == "rrBLUP_ML":
        c, s2, _, _ = ml_fit(lam, z ** 2, gamma=1.0); w = c * lam / (c * lam + s2)
    elif method == "gammaBLUP_ML":
        c, s2, g, _ = ml_fit(lam, z ** 2); w = c * lam ** g / (c * lam ** g + s2); info["gamma"] = float(g)
    elif method == "rrBLUP_CV":
        w = _ridge_grid(ev)[_cv_select(K_tr, y_tr, _ridge_grid, seed=seed)]
    elif method == "power_CV":
        i = _cv_select(K_tr, y_tr, _power_grid, seed=seed)
        w = _power_grid(ev)[i]; info["a"] = float(POWERS[i // len(DELTAS)])
    elif method == "PCR_CV":
        ks = _pcr_ks(int(0.8 * len(y_tr)))
        i = _cv_select(K_tr, y_tr, lambda e: _pcr_grid(e, ks), seed=seed)
        k = min(int(ks[i]), len(ev)); w = _pcr_grid(ev, [k])[0]; info["k"] = k
    elif method == "safe_CV":
        # held-out selection between rrBLUP_CV and power_CV, each refitted on every inner split (T6)
        err = {"rrBLUP_CV": 0.0, "power_CV": 0.0}
        for itr, iva in KFold(5, shuffle=True, random_state=seed + 101).split(y_tr):
            for mm in err:
                p, _ = fit_predict(K_tr[np.ix_(itr, itr)], y_tr[itr], K_tr[np.ix_(iva, itr)], mm, seed=seed)
                err[mm] += float(np.sum((y_tr[iva] - p) ** 2))
        pick = "power_CV" if err["power_CV"] < err["rrBLUP_CV"] else "rrBLUP_CV"
        p, inf = fit_predict(K_tr, y_tr, K_te_tr, pick, seed=seed)
        return p, dict(inf, pick=pick, err=err)
    else:
        raise ValueError(method)
    return K_te_tr @ (U @ (w / ev * z)) + m, info


def gamma_test(K_tr, y_tr):
    """Diagnostic (T8): ML fit of z_i ~ N(0, c lam_i^g + s2) on the training data; LRT of g = 1 (chi2_1 p-value)."""
    from scipy.stats import chi2
    ev, U = _eig(K_tr); lam = ev / ev.mean(); z2 = (U.T @ ((y_tr - y_tr.mean()) / y_tr.std())) ** 2
    c, s2, g, f1 = ml_fit(lam, z2); f0 = ml_fit(lam, z2, gamma=1.0)[3]
    lr = max(2 * (f0 - f1), 0.0)
    return dict(gamma=float(g), lr=float(lr), p=float(chi2.sf(lr, 1)))
