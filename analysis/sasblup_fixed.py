"""SAS-BLUP with the two corrections agreed for the revision.

(1) Bounded leverage (Condition 1 of Appendix B): adaptive weights are restricted to [0, w_bar], where w_bar is the
    largest weight of the LOO-tuned rrBLUP profile on the same training set. Because sum_k U_ik^2 = 1, every leverage
    then satisfies (S_w)_ii <= w_bar < 1, which rules out the near-interpolating profiles that drive the LOO shortcut
    to 0/0. The rrBLUP profile itself remains feasible, so the families still nest rrBLUP.
(2) Collapse guard on held-out data: the adaptive profile is retained only if, in a 5-fold inner cross-validation
    within the training set (refitting both estimators on each inner split), its held-out MSE is lower than rrBLUP's.
All other steps (warm starts, restarts, penalty and knot grids, isotonic projection, spline basis) follow the
authors' notebook.
"""
import numpy as np
from scipy.interpolate import BSpline
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
import sasblup_engine as E

FAMILIES = ("scalar", "grouped", "smooth", "monotone", "spline")


def _csig(x, wb):
    return wb * E._sig(x)


def _clogit(w, wb):
    return E._logit(np.clip(w / wb, 1e-6, 1 - 1e-6))


def _warm(ev, U, z, y):
    w_sc, _, lam, loo_sc = E._scalar_warmstart(ev, U, z, y)
    wb = float(w_sc.max())
    return w_sc, _clogit(w_sc, wb), wb, loo_sc


def _grad_logit(wr, wb, U, U2, z, y, gamma=0.0):
    s = E._sig(wr); w = wb * s
    val, gw = E._loo_val_grad_w(w, U, U2, z, y)
    dw = np.diff(w); pen = gamma * float(np.sum(dw ** 2))
    gp = np.zeros_like(w); gp[:-1] -= 2 * gamma * dw; gp[1:] += 2 * gamma * dw
    g = (gw + gp) * wb * s * (1 - s)
    return val + pen, np.where(np.isfinite(g), g, 0.)


def learn_grouped(ev, U, z, y, n_groups=None):
    n = len(ev)
    if n_groups is None:
        n_groups = max(3, min(max(E.EFF_RANK + 2, 5), 12))
    groups = np.array_split(np.arange(n), n_groups)
    w_sc, lg_sc, wb, loo_sc = _warm(ev, U, z, y)
    g_warm = np.array([lg_sc[g].mean() for g in groups])
    U2 = U ** 2

    def expand(gr):
        w = np.zeros(n); gv = _csig(gr, wb)
        for k, grp in enumerate(groups): w[grp] = gv[k]
        return w

    def obj(gr):
        s = E._sig(gr); w = expand(gr)
        val, gw = E._loo_val_grad_w(w, U, U2, z, y)
        g = np.array([np.sum(gw[grp]) * wb * s[k] * (1 - s[k]) for k, grp in enumerate(groups)])
        return val, np.where(np.isfinite(g), g, 0.)
    rng = np.random.RandomState(0)
    inits = [g_warm] + [g_warm + rng.randn(n_groups) * 0.5 for _ in range(2)]
    best, _ = E._multi_restart_jac(obj, inits, maxiter=300)
    w = np.clip(expand(best), 0, wb)
    return (w_sc if E.loo_mse(w, U, z, y) > loo_sc + 0.005 else w)


def learn_smooth(ev, U, z, y, gamma_grid=None):
    n = len(ev)
    if gamma_grid is None: gamma_grid = np.logspace(-3, 3, 6)
    w_sc, lg_sc, wb, loo_sc = _warm(ev, U, z, y)
    try:
        w_gr = learn_grouped(ev, U, z, y, n_groups=max(3, min(E.EFF_RANK + 2, 10)))
        lg_gr = _clogit(w_gr, wb); has_gr = True
    except Exception:
        has_gr = False
    rng = np.random.RandomState(1); U2 = U ** 2
    best_val, best_w = np.inf, w_sc.copy()
    for gm in gamma_grid:
        f = lambda wr, gm=gm: _grad_logit(wr, wb, U, U2, z, y, gamma=gm)
        inits = [lg_sc.copy()] + ([lg_gr.copy()] if has_gr else []) + [lg_sc + rng.randn(n) * 0.3]
        br, _ = E._multi_restart_jac(f, inits, maxiter=100)
        wc = np.clip(_csig(br, wb), 0, wb); v = E.loo_mse(wc, U, z, y)
        if v < best_val: best_val, best_w = v, wc.copy()
    return (w_sc if best_val > loo_sc + 0.005 else best_w)


def learn_monotone(ev, U, z, y):
    n = len(ev)
    w_sc, lg_sc, wb, loo_sc = _warm(ev, U, z, y)
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip"); U2 = U ** 2
    idx = np.arange(n, dtype=float)

    def obj(wr):
        s = E._sig(wr); wf = wb * s
        wm = np.clip(iso.fit_transform(idx, wf), 0, wb)
        val, gw = E._loo_val_grad_w(wm, U, U2, z, y)
        g = gw * wb * s * (1 - s)
        return (val if np.isfinite(val) else 1e9), np.where(np.isfinite(g), g, 0.)
    rng = np.random.RandomState(2); decay = np.linspace(1., 0., n)
    inits = [lg_sc.copy()] + [lg_sc + rng.randn(n) * 0.4 * decay for _ in range(2)]
    inits.append(np.where(np.arange(n) < max(E.EFF_RANK, 5), 2., -2.))
    br, _ = E._multi_restart_jac(obj, inits, maxiter=150)
    wm = np.clip(iso.fit_transform(idx, _csig(br, wb)), 0, wb)
    return (w_sc if E.loo_mse(wm, U, z, y) > loo_sc + 0.005 else wm)


def learn_spline(ev, U, z, y, n_knots=5):
    n = len(ev)
    n_knots = max(2, min(n_knots, max(2, E.EFF_RANK // 2)))
    t = np.linspace(0, 1, n); deg = 3
    knots = np.linspace(0, 1, n_knots + 2)[1:-1]
    ta = np.r_[[0] * deg, knots, [1] * deg]
    cols = []
    for i in range(len(ta) - deg - 1):
        ts = ta[i:i + deg + 2]
        if len(ts) < deg + 2: continue
        cols.append(np.nan_to_num(BSpline.basis_element(ts, extrapolate=False)(t)))
    B = np.column_stack(cols) if cols else np.ones((n, 1))
    w_sc, lg_sc, wb, loo_sc = _warm(ev, U, z, y)
    try: c0 = np.linalg.lstsq(B, lg_sc, rcond=None)[0]
    except Exception: c0 = np.zeros(B.shape[1])
    U2 = U ** 2

    def obj(c):
        s = E._sig(B @ c); w = wb * s
        val, gw = E._loo_val_grad_w(w, U, U2, z, y)
        g = (wb * s * (1 - s))[:, None] * B
        gc = g.T @ gw
        return val, np.where(np.isfinite(gc), gc, 0.)
    rng = np.random.RandomState(3)
    inits = [c0.copy()] + [c0 + rng.randn(B.shape[1]) * 0.5 for _ in range(2)] + [np.zeros(B.shape[1])]
    bc, _ = E._multi_restart_jac(obj, inits, maxiter=150)
    w = np.clip(_csig(B @ bc, wb), 0, wb)
    return (w_sc if E.loo_mse(w, U, z, y) > loo_sc + 0.005 else w)


def family_weights(strategy, ev, U, z, y):
    if strategy == "scalar": return E._learn_scalar(ev, U, z, y)[0]
    if strategy == "grouped": return learn_grouped(ev, U, z, y)
    if strategy == "smooth": return learn_smooth(ev, U, z, y)
    if strategy == "monotone": return learn_monotone(ev, U, z, y)
    if strategy == "spline": return learn_spline(ev, U, z, y)
    raise ValueError(strategy)


def _fit_alpha(G_tr, y_tr, strategy):
    ev, U = E.safe_eigh(G_tr); z = U.T @ y_tr
    w = family_weights(strategy, ev, U, z, y_tr)
    return U @ ((w / (ev + 1e-9)) * z), w, ev


class SASBLUPFixed:
    """SAS-BLUP with bounded leverage and an inner-CV collapse guard."""

    def __init__(self, strategy="grouped", guard_folds=5, guard_seed=0):
        self.strategy = strategy; self.guard_folds = guard_folds; self.guard_seed = guard_seed

    def fit(self, G_tr, y_tr):
        alpha, w, ev = _fit_alpha(G_tr, y_tr, self.strategy)
        self.guard_used_ = False
        if self.strategy != "scalar":
            a_sc, w_sc, _ = _fit_alpha(G_tr, y_tr, "scalar")
            sse_a = sse_s = 0.0
            for itr, iva in KFold(self.guard_folds, shuffle=True, random_state=self.guard_seed).split(y_tr):
                Gi = G_tr[np.ix_(itr, itr)]; Gv = G_tr[np.ix_(iva, itr)]
                aa = _fit_alpha(Gi, y_tr[itr], self.strategy)[0]
                as_ = _fit_alpha(Gi, y_tr[itr], "scalar")[0]
                sse_a += float(np.sum((y_tr[iva] - Gv @ aa) ** 2))
                sse_s += float(np.sum((y_tr[iva] - Gv @ as_) ** 2))
            if sse_a >= sse_s:
                alpha, w = a_sc, w_sc; self.guard_used_ = True
        self.alpha_, self.weights_, self.ev_ = alpha, w, ev
        return self

    def predict(self, G_val_tr):
        return G_val_tr @ self.alpha_
