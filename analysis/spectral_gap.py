"""Corrected spectral-gap diagnostic (Section 3.8 of the manuscript).

Rotated phenotype z = U^T y (y standardised). Noise variance sigma^2 is estimated by maximum likelihood under
the variance-component model y = g + e, g ~ N(0, s_g K), e ~ N(0, s_e I), i.e. z_i ~ N(0, s_g*lambda_i + s_e).
Risk (Eq. 6): R(w) = (1/n) sum_i [(1-w_i)^2 theta_i^2 + w_i^2 sigma^2], with the plug-in theta_i^2 = z_i^2.
Gap (Eq. 22): Delta = min_lambda R(ridge(lambda)) - R(oracle).
Calibration: parametric bootstrap under the proportional (rrBLUP/GBLUP) model, z* ~ N(0, s_g lambda + s_e),
re-estimating sigma^2 on every draw; p = (1 + #{Delta* >= Delta}) / (B + 1).
"""
import numpy as np
from scipy.optimize import minimize

LAM_GRID = np.logspace(-5, 5, 200)


def ml_varcomp(ev, z):
    def nll(p):
        sg, se = np.exp(p)
        v = sg * ev + se
        return 0.5 * np.sum(np.log(v) + z ** 2 / v)
    best = None
    for x0 in ([0.0, 0.0], [1.0, -2.0], [-2.0, 1.0]):
        r = minimize(nll, np.array(x0), method="L-BFGS-B", bounds=[(-25, 10), (-25, 10)])
        if best is None or r.fun < best.fun:
            best = r
    return np.exp(best.x)  # s_g, s_e


def gap_stat(ev, z, s2):
    th2 = z ** 2
    w_or = th2 / (th2 + s2)
    r_or = np.mean((1 - w_or) ** 2 * th2 + w_or ** 2 * s2)
    W = ev[None, :] / (ev[None, :] + LAM_GRID[:, None])
    rr = np.mean((1 - W) ** 2 * th2[None, :] + W ** 2 * s2, axis=1)
    i = int(np.argmin(rr))
    return float(rr[i] - r_or), float(rr[i]), float(r_or), float(LAM_GRID[i])


def diagnose(ev, U, y_std, B=1000, seed=0):
    z = U.T @ y_std
    sg, se = ml_varcomp(ev, z)
    d, rr, ro, lam = gap_stat(ev, z, se)
    rng = np.random.default_rng(seed)
    sd = np.sqrt(sg * ev + se)
    null = np.empty(B)
    for b in range(B):
        zb = rng.standard_normal(len(ev)) * sd
        sgb, seb = ml_varcomp(ev, zb)
        null[b] = gap_stat(ev, zb, seb)[0]
    p = (1 + np.sum(null >= d)) / (B + 1)
    h2 = sg * ev.mean() / (sg * ev.mean() + se)
    return dict(delta=d, rel_gap=d / rr if rr > 0 else np.nan, ridge_risk=rr, oracle_risk=ro, lam=lam,
                sigma_g2=sg, sigma_e2=se, h2=h2, null_mean=float(null.mean()),
                null_q95=float(np.quantile(null, 0.95)), p_value=float(p),
                excess=d - float(null.mean()))


def notebook_delta(ev, U, y_std):
    """The original notebook computation, for reference (sigma^2 floored at 1e-8, scaled by 1/n)."""
    n = len(ev); z = U.T @ y_std
    sig2 = max(float(np.var(y_std - U @ (U.T @ y_std))), 1e-8)
    w_or = z ** 2 / (z ** 2 + max(sig2 / n, 1e-12))
    or_risk = float(np.sum(((1 - w_or) * z) ** 2) + np.sum(w_or ** 2) * sig2 / n)
    best = min(float(np.sum(((1 - ev / (ev + l)) * z) ** 2) + np.sum((ev / (ev + l)) ** 2) * sig2 / n)
               for l in np.logspace(-5, 5, 120))
    return max(best - or_risk, 0.0)
