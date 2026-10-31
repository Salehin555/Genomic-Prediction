# Auto-extracted from "source code.ipynb" (cell 2): definitions only, unchanged.
import numpy as np, pandas as pd, torch, torch.nn as nn
from scipy.linalg import eigh
from scipy.optimize import minimize
from scipy.interpolate import BSpline
from scipy.stats import pearsonr, ttest_rel
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
USE_GPU = False
DEVICE = torch.device("cpu")
EFF_RANK = 9          # set by the caller from the full kernel, as in the notebook

# ---- reconstructed: analytical gradient of the LOO criterion (Eq. 18), missing from the saved notebook ----
def _loo_val_grad_w(w, U, U2, z, y):
    """LOO MSE and its gradient with respect to the weights w (same clipping as loo_mse)."""
    w = np.clip(np.asarray(w, dtype=np.float64), 0., 1. - 1e-7)
    n = len(y)
    yh = U @ (w * z)
    H = np.clip(U2 @ w, 0., 1. - 1e-7)
    d = 1. - H
    r = (y - yh) / d
    val = float(np.mean(r ** 2))
    g = (2. / n) * (-z * (U.T @ (r / d)) + U2.T @ (r ** 2 / d))
    if not np.isfinite(val):
        return 1e9, np.zeros_like(w)
    return val, np.where(np.isfinite(g), g, 0.)

def _loo_val_grad(wr, U, U2, z, y, gamma=0.0):
    """LOO MSE plus the roughness penalty gamma*sum (w_{i+1}-w_i)^2 (Eq. 16), gradient w.r.t. logits wr."""
    w = _sig(wr)
    val, gw = _loo_val_grad_w(w, U, U2, z, y)
    dw = np.diff(w)
    pen = gamma * float(np.sum(dw ** 2))
    gp = np.zeros_like(w)
    gp[:-1] -= 2. * gamma * dw
    gp[1:] += 2. * gamma * dw
    g = (gw + gp) * w * (1. - w)
    return val + pen, np.where(np.isfinite(g), g, 0.)


def _to_gpu(arr):
    return torch.tensor(np.asarray(arr, dtype=np.float32), device=DEVICE)

def _to_cpu(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy().astype(np.float64)
    return np.asarray(t, dtype=np.float64)

def _gpu_matmul(A, B):
    if USE_GPU:
        r = torch.mm(_to_gpu(A), _to_gpu(B))
        torch.cuda.synchronize(); return _to_cpu(r)
    return A @ B

def _gpu_eigh(K):
    if USE_GPU:
        vals_t, vecs_t = torch.linalg.eigh(_to_gpu(K))
        return _to_cpu(vals_t)[::-1].copy(), _to_cpu(vecs_t)[:,::-1].copy()
    vals, vecs = eigh(K)
    return vals[::-1].copy(), vecs[:,::-1].copy()

def clear_gpu():
    if USE_GPU: torch.cuda.empty_cache(); gc.collect()

STRATEGIES = ('scalar', 'grouped', 'smooth', 'monotone', 'spline')

def safe_eigh(K, jitter=1e-8):
    Kjit = K + jitter*np.eye(K.shape[0])
    vals, vecs = _gpu_eigh(Kjit)
    return np.maximum(vals, 0.), vecs

def scalar_shrinkage(ev, lam):
    return ev / (ev + lam)

def oracle_shrinkage(theta, sigma2, n):
    return theta**2 / (theta**2 + max(sigma2/n, 1e-12))

def _sig(x):
    xc = np.clip(x, -500, 500)
    return np.where(xc>=0, 1./(1.+np.exp(-xc)), np.exp(xc)/(1.+np.exp(xc)))

def _logit(w):
    return np.log(np.clip(w,1e-6,1-1e-6) / (1-np.clip(w,1e-6,1-1e-6)))

_GPU_LOO_CACHE = {}

def loo_mse(w, U, z, y):
    w = np.clip(np.asarray(w, dtype=np.float32), 0., 1.-1e-7)
    if USE_GPU:
        key = (id(U), U.shape)
        if key not in _GPU_LOO_CACHE:
            _GPU_LOO_CACHE[key] = {
                'U' :torch.tensor(U,   dtype=torch.float32,device=DEVICE),
                'U2':torch.tensor(U**2,dtype=torch.float32,device=DEVICE),
                'z' :torch.tensor(z,   dtype=torch.float32,device=DEVICE),
                'y' :torch.tensor(y,   dtype=torch.float32,device=DEVICE),
            }
        c  = _GPU_LOO_CACHE[key]
        wt = torch.tensor(w, dtype=torch.float32, device=DEVICE)
        yh = c['U'] @ (wt * c['z'])
        Hd = torch.clamp(c['U2'] @ wt, 0., 1.-1e-7)
        r  = (c['y'] - yh) / (1. - Hd)
        v  = float(torch.mean(r**2).item())
        return v if np.isfinite(v) else 1e9
    w   = np.asarray(w, dtype=np.float64)
    yh  = U @ (w * z)
    Hd  = np.clip((U**2) @ w, 0., 1.-1e-7)
    r   = (y - yh) / (1. - Hd)
    v   = float(np.mean(r**2))
    return v if np.isfinite(v) else 1e9

def _clear_loo_cache():
    _GPU_LOO_CACHE.clear(); clear_gpu()

def _scalar_warmstart(ev, U, z, y, lam_grid=None):
    """Find best scalar λ → return (w_sc, logit_sc, best_lam, loo_sc)."""
    if lam_grid is None: lam_grid = np.logspace(-5, 5, 40)
    best_lam, best_mse = lam_grid[0], np.inf
    for lam in lam_grid:
        mse = loo_mse(scalar_shrinkage(ev, lam), U, z, y)
        if mse < best_mse: best_mse, best_lam = mse, lam
    w_sc = np.clip(scalar_shrinkage(ev, best_lam), 1e-6, 1-1e-6)
    return w_sc, _logit(w_sc), best_lam, best_mse

def _multi_restart(obj_fn, inits, maxiter=500, ftol=1e-12, gtol=1e-8):
    """Run L-BFGS-B from each init; return best (x, val)."""
    best_val, best_x = np.inf, inits[0]
    for x0 in inits:
        try:
            res = minimize(obj_fn, x0, method='L-BFGS-B',
                           options={'maxiter':maxiter,'ftol':ftol,'gtol':gtol})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val, best_x = res.fun, res.x.copy()
        except Exception:
            pass
    return best_x, best_val

def _multi_restart_jac(obj_jac, inits, maxiter=150, ftol=1e-11):
    """Multi-start L-BFGS-B with analytical gradient (jac=True)."""
    best_val, best_x = np.inf, inits[0]
    for x0 in inits:
        try:
            res = minimize(obj_jac, x0, method='L-BFGS-B', jac=True,
                           options={'maxiter': maxiter, 'ftol': ftol, 'gtol': 1e-8})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val, best_x = res.fun, res.x.copy()
        except Exception:
            pass
    return best_x, best_val

def _collapse_guard(w_cand, loo_cand, w_sc, loo_sc, thr=0.005):
    """If candidate is worse than scalar + thr, return scalar."""
    if loo_cand > loo_sc + thr:
        return w_sc.copy(), True
    return w_cand, False

def _learn_scalar(ev, U, z, y, lam_grid=None):
    w_sc, _, best_lam, _ = _scalar_warmstart(ev, U, z, y, lam_grid)
    return w_sc, best_lam

def _learn_grouped(ev, U, z, y, n_groups):
    n = len(ev)
    if n_groups is None:
        n_groups = max(3, min(max(EFF_RANK+2, 5), 12))
    groups  = np.array_split(np.arange(n), n_groups)
    w_sc, logit_sc, _, loo_sc = _scalar_warmstart(ev, U, z, y)
    g_warm  = np.array([logit_sc[g].mean() for g in groups])

    def expand(g_raw):
        g=_sig(g_raw); w=np.zeros(n)
        for k,grp in enumerate(groups): w[grp]=g[k]
        return w

    rng   = np.random.RandomState(0)
    inits = [g_warm] + [g_warm + rng.randn(n_groups)*0.5 for _ in range(2)]  # 3 restarts
    U2=U**2
    def obj_grouped(g_raw):
        g_sig=_sig(g_raw); sw_g=g_sig*(1.-g_sig)
        w=expand(g_raw); val,gw=_loo_val_grad_w(w,U,U2,z,y)
        grad_g=np.array([np.sum(gw[grp])*sw_g[k] for k,grp in enumerate(groups)])
        return val,np.where(np.isfinite(grad_g),grad_g,0.)
    best_g,best_val=_multi_restart_jac(obj_grouped,inits,maxiter=300)
    w_opt = np.clip(expand(best_g), 0, 1)
    w_fin, collapsed = _collapse_guard(w_opt, loo_mse(w_opt,U,z,y), w_sc, loo_sc)
    return (w_sc if collapsed else w_fin), n_groups

def _learn_smooth(ev, U, z, y, gamma_grid=None):
    n = len(ev)
    if gamma_grid is None: gamma_grid = np.logspace(-3, 3, 6)   # 6 (was 14)
    w_sc, logit_sc, _, loo_sc = _scalar_warmstart(ev, U, z, y)

    # Grouped as secondary warm-start
    try:
        w_gr,_ = _learn_grouped(ev,U,z,y,n_groups=max(3,min(EFF_RANK+2,10)))
        logit_gr = _logit(np.clip(w_gr,1e-6,1-1e-6)); has_gr=True
    except Exception: has_gr=False

    rng = np.random.RandomState(1)
    best_val, best_w = np.inf, w_sc.copy()

    U2=U**2  # precompute once
    for gamma in gamma_grid:
        # Analytical gradient with TV penalty (jac=True → 1 eval/step vs 150)
        def make_obj(gm):
            def f(wr): return _loo_val_grad(wr,U,U2,z,y,gamma=gm)
            return f
        obj=make_obj(gamma)
        inits=[logit_sc.copy()]
        if has_gr: inits.append(logit_gr.copy())
        inits+=[logit_sc+rng.randn(n)*0.3 for _ in range(1)]
        best_wr,_=_multi_restart_jac(obj,inits,maxiter=100)
        w_cand=np.clip(_sig(best_wr),0,1); val_loo=loo_mse(w_cand,U,z,y)
        if val_loo<best_val: best_val,best_w=val_loo,w_cand.copy()

    w_fin, collapsed = _collapse_guard(best_w, best_val, w_sc, loo_sc)
    return (w_sc if collapsed else w_fin), (None if collapsed else best_val)

def _learn_monotone(ev, U, z, y):
    n = len(ev)
    w_sc, logit_sc, _, loo_sc = _scalar_warmstart(ev, U, z, y)
    iso = IsotonicRegression(increasing=False, out_of_bounds='clip')

    U2=U**2
    def obj_proj(wr):
        # Straight-through gradient: grad flows through sigmoid, not isotonic projection
        w_free=_sig(wr); sw=w_free*(1.-w_free)
        w_mono=np.clip(iso.fit_transform(np.arange(n,dtype=float),w_free),0,1)
        val,gw=_loo_val_grad_w(w_mono,U,U2,z,y)
        gwr=gw*sw  # chain rule (straight-through for isotonic)
        return (val if np.isfinite(val) else 1e9), np.where(np.isfinite(gwr),gwr,0.)

    rng=np.random.RandomState(2); decay=np.linspace(1.,0.,n)
    inits=[logit_sc.copy()]
    for _ in range(2): inits.append(logit_sc+rng.randn(n)*0.4*decay)
    inits.append(np.where(np.arange(n)<max(EFF_RANK,5),2.,-2.))
    best_wr,_=_multi_restart_jac(obj_proj,inits,maxiter=150)
    w_free     = _sig(best_wr)
    w_mono     = np.clip(iso.fit_transform(np.arange(n,dtype=float), w_free),0,1)
    loo_mono   = loo_mse(w_mono, U, z, y)

    w_fin, collapsed = _collapse_guard(w_mono, loo_mono, w_sc, loo_sc)
    return (w_sc if collapsed else w_fin), 'isotonic'

def _learn_spline(ev, U, z, y, n_knots):
    n       = len(ev)
    n_knots = max(2, min(n_knots, max(2, EFF_RANK//2)))
    t_idx   = np.linspace(0,1,n); degree=3
    knots   = np.linspace(0,1,n_knots+2)[1:-1]
    t_aug   = np.r_[[0]*degree, knots, [1]*degree]
    B_cols  = []
    for i in range(len(t_aug)-degree-1):
        t_sub = t_aug[i:i+degree+2]
        if len(t_sub)<degree+2: continue
        B_cols.append(np.nan_to_num(BSpline.basis_element(t_sub,extrapolate=False)(t_idx)))
    B      = np.column_stack(B_cols) if B_cols else np.ones((n,1))
    n_coef = B.shape[1]

    w_sc, logit_sc, _, loo_sc = _scalar_warmstart(ev, U, z, y)
    try: c_warm,_,_,_ = np.linalg.lstsq(B, logit_sc, rcond=None)
    except Exception: c_warm = np.zeros(n_coef)

    U2=U**2
    def obj_spline(c):
        # Analytical gradient through spline basis
        w=_sig(B@c); sw_B=(w*(1-w))[:,None]*B   # Jacobian ∂w/∂c
        val,gw=_loo_val_grad_w(w,U,U2,z,y)
        gc=sw_B.T@gw  # chain rule: ∂LOO/∂c = (∂w/∂c)ᵀ ∂LOO/∂w
        return val, np.where(np.isfinite(gc),gc,0.)

    rng=np.random.RandomState(3)
    inits=[c_warm.copy()]+[c_warm+rng.randn(n_coef)*0.5 for _ in range(2)]+[np.zeros(n_coef)]
    best_c,_=_multi_restart_jac(obj_spline,inits,maxiter=150)
    w_sp      = np.clip(_sig(B@best_c),0,1)
    loo_sp    = loo_mse(w_sp,U,z,y)

    w_fin, collapsed = _collapse_guard(w_sp, loo_sp, w_sc, loo_sc)
    return (w_sc if collapsed else w_fin), n_knots

class SASBLUPEstimator(BaseEstimator, RegressorMixin):
    """SAS-BLUP: per-eigenvector weights wᵢ∈[0,1]; hardened against collapse."""

    def __init__(self, strategy='grouped', n_groups=None, n_knots=5, verbose=False):
        self.strategy=strategy; self.n_groups=n_groups
        self.n_knots=n_knots;   self.verbose=verbose

    def fit(self, G_tr, y_tr):
        self.ev_, self.U_ = safe_eigh(G_tr)
        self.z_ = self.U_.T @ y_tr; self.y_tr_ = y_tr.copy()
        ev,U,z,y = self.ev_,self.U_,self.z_,y_tr
        if   self.strategy=='scalar'  : w,hp=_learn_scalar(ev,U,z,y)
        elif self.strategy=='grouped' : w,hp=_learn_grouped(ev,U,z,y,self.n_groups)
        elif self.strategy=='smooth'  : w,hp=_learn_smooth(ev,U,z,y)
        elif self.strategy=='monotone': w,hp=_learn_monotone(ev,U,z,y)
        elif self.strategy=='spline'  : w,hp=_learn_spline(ev,U,z,y,self.n_knots)
        else: raise ValueError(f"Unknown strategy '{self.strategy}'")
        self.weights_=w; self.hp_=hp
        self.loo_=loo_mse(w,U,z,y)
        self.alpha_=U@((w/(ev+1e-9))*z)
        if self.verbose:
            print(f"  [{self.strategy:>8}]  LOO={self.loo_:.5f}  w∈[{w.min():.3f},{w.max():.3f}]")
        return self

    def predict(self, G_val_tr):
        return _gpu_matmul(G_val_tr, self.alpha_.reshape(-1,1)).ravel()

    def shrinkage_profile(self):
        return self.ev_.copy(), self.weights_.copy()

def compute_metrics(y_true, y_pred):
    if np.std(y_pred)<1e-10: return dict(r2=-999.,r=0.,rmse=np.inf,mae=np.inf)
    r2=float(r2_score(y_true,y_pred)); r,_=pearsonr(y_true,y_pred)
    return dict(r2=r2,r=float(r),rmse=float(np.sqrt(np.mean((y_true-y_pred)**2))),
                mae=float(mean_absolute_error(y_true,y_pred)))

def summarise_cv(results, ref='scalar'):
    ref_r2=np.array([m['r2'] for m in results[ref]]); rows=[]
    for method,folds in results.items():
        r2s=np.array([m['r2'] for m in folds]); rs=np.array([m['r'] for m in folds])
        rms=np.array([m['rmse'] for m in folds]); ok=~np.isnan(r2s)
        p=float(ttest_rel(r2s[ok],ref_r2[ok]).pvalue) if (method!=ref and ok.sum()>=2) else np.nan
        rows.append(dict(method=method,r2_mean=float(np.nanmean(r2s)),r2_std=float(np.nanstd(r2s)),
                          r_mean=float(np.nanmean(rs)),r_std=float(np.nanstd(rs)),
                          rmse=float(np.nanmean(rms)),
                          delta_r2=float(np.nanmean(r2s)-np.nanmean(ref_r2)),p_ttest=p))
    return pd.DataFrame(rows).sort_values('r2_mean',ascending=False).reset_index(drop=True)

class BayesBPredictor:
    def __init__(self,n_components=None,pi=0.05,n_iter=30,seed=0):
        self.n_components=n_components; self.pi=pi; self.n_iter=n_iter; self.seed=seed
    def fit(self,G_tr,y_tr):
        ev,U=safe_eigh(G_tr); n=len(y_tr); k=self.n_components or min(n-1,n)
        X=U[:,:k]*np.sqrt(np.maximum(ev[:k],0)); self._ev=ev; self._U=U; self._k=k
        beta=np.zeros(k,dtype=np.float32); sigma2_e=float(np.var(y_tr))
        sigma2_b=sigma2_e/(k*self.pi+1e-8); gamma=np.ones(k,dtype=np.float32)*self.pi
        if USE_GPU:
            Xt=_to_gpu(X); yt=_to_gpu(y_tr.astype(np.float32)); bt=torch.zeros(k,dtype=torch.float32,device=DEVICE)
            for _ in range(self.n_iter):
                for j in range(k):
                    xj=Xt[:,j]; Xb=torch.mv(Xt,bt); rj=yt-Xb+bt[j]*xj
                    xjxj=float(torch.dot(xj,xj).item()); bjh=float(torch.dot(xj,rj).item())/(xjxj+sigma2_e/sigma2_b+1e-8)
                    lo=np.log(self.pi+1e-8)-np.log(1-self.pi+1e-8)+0.5*bjh**2*xjxj/sigma2_e
                    gj=1./(1.+np.exp(-lo)); gamma[j]=gj; bt[j]=gj*bjh
                resid=yt-torch.mv(Xt,bt); sigma2_e=float(torch.mean(resid**2).item())+1e-8
                b_np=bt.cpu().numpy(); active=gamma>0.05
                if active.any(): sigma2_b=float(np.mean(b_np[active]**2)/(self.pi+1e-8))+1e-8
            beta=bt.cpu().numpy().astype(np.float64); del Xt,yt,bt; clear_gpu()
        else:
            for _ in range(self.n_iter):
                for j in range(k):
                    xj=X[:,j]; rj=y_tr-X@beta+beta[j]*xj
                    bjh=(xj@rj)/(xj@xj+sigma2_e/sigma2_b+1e-8)
                    lo=np.log(self.pi+1e-8)-np.log(1-self.pi+1e-8)+0.5*bjh**2*(xj@xj)/sigma2_e
                    gamma[j]=1./(1.+np.exp(-lo)); beta[j]=gamma[j]*bjh
                resid=y_tr-X@beta; sigma2_e=float(np.mean(resid**2))+1e-8
                active=gamma>0.05
                if active.any(): sigma2_b=float(np.mean(beta[active]**2)/(self.pi+1e-8))+1e-8
        self._beta=beta; return self
    def predict(self,G_val_tr):
        ev,U,k=self._ev,self._U,self._k
        alpha=U[:,:k]@(self._beta/(np.sqrt(np.maximum(ev[:k],0))+1e-9))
        return G_val_tr@alpha

class KRR_RBF_Predictor:
    def __init__(self,gamma_grid=None,lam_grid=None):
        self.gamma_grid=gamma_grid or np.logspace(-3,1,8)   # 8 (was 15)
        self.lam_grid=lam_grid or np.logspace(-4,4,12)   # 12 (was 20)
    def _rbf(self,G_tr,G_val_tr=None,gamma=1.0):
        if USE_GPU:
            Gt=_to_gpu(G_tr); d=torch.diag(Gt); sq=d[:,None]+d[None,:]-2*Gt
            K_tr=torch.exp(-gamma*torch.clamp(sq,min=0.)).cpu().numpy().astype(np.float64)
            if G_val_tr is None: del Gt; clear_gpu(); return K_tr
            Gvt=_to_gpu(G_val_tr)
            dv=torch.diag(Gvt) if G_val_tr.shape[0]==G_val_tr.shape[1] else torch.full((G_val_tr.shape[0],),float(d.mean().item()),device=DEVICE)
            sq_v=dv[:,None]+d[None,:]-2*Gvt
            K_val=torch.exp(-gamma*torch.clamp(sq_v,min=0.)).cpu().numpy().astype(np.float64)
            del Gt,Gvt; clear_gpu(); return K_tr,K_val
        diag=np.diag(G_tr); sq=diag[:,None]+diag[None,:]-2*G_tr; K_tr=np.exp(-gamma*np.maximum(sq,0))
        if G_val_tr is None: return K_tr
        n_val=G_val_tr.shape[0]
        dv=np.array([G_val_tr[i,i] if G_val_tr.shape[0]==G_val_tr.shape[1] else diag.mean() for i in range(n_val)])
        return K_tr,np.exp(-gamma*np.maximum(dv[:,None]+diag[None,:]-2*G_val_tr,0))
    def fit(self,G_tr,y_tr):
        best_g,best_l,best_mse=self.gamma_grid[0],self.lam_grid[0],np.inf
        for gamma in self.gamma_grid:
            K_tr=self._rbf(G_tr,gamma=gamma); ev_k,U_k=safe_eigh(K_tr); z=U_k.T@y_tr
            for lam in self.lam_grid:
                mse=loo_mse(scalar_shrinkage(ev_k,lam),U_k,z,y_tr)
                if mse<best_mse: best_mse,best_g,best_l=mse,gamma,lam
        self.gamma_=best_g; self.lam_=best_l; self._G_tr=G_tr
        K_tr=self._rbf(G_tr,gamma=best_g); m=RidgeCV(alphas=[best_l],fit_intercept=False); m.fit(K_tr,y_tr); self._m=m
        return self
    def predict(self,G_val_tr):
        K_tr,K_val=self._rbf(self._G_tr,G_val_tr,self.gamma_); return self._m.predict(K_val)

class PCRPredictor:
    def fit(self,G_tr,y_tr):
        ev,U=safe_eigh(G_tr); z=U.T@y_tr; n=len(ev); best_k,best=1,np.inf
        for k in range(1,n):
            w=np.zeros(n); w[:k]=1.; mse=loo_mse(w,U,z,y_tr)
            if mse<best: best,best_k=mse,k
        self.alpha_=U[:,:best_k]@(z[:best_k]/(ev[:best_k]+1e-9)); return self
    def predict(self,G_val_tr): return G_val_tr@self.alpha_

class DNNGPPredictor:
    """
    Simplified DNNGP: DR → BatchNorm → ReLU → Dropout → Linear
    Training features: top-k PC scores from G_tr eigendecomposition
    Validation features: G_val_tr @ U[:,k] projected onto training PCs
    """
    def __init__(self, k_pcs=30, hidden=(64, 32), dropout=0.4,
                 n_epochs=300, lr=1e-3, seed=0):
        self.k_pcs    = k_pcs
        self.hidden   = hidden
        self.dropout  = dropout
        self.n_epochs = n_epochs
        self.lr       = lr
        self.seed     = seed

    def fit(self, G_tr, y_tr):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        ev, U = safe_eigh(G_tr)
        n  = len(y_tr)
        k  = min(n - 2, self.k_pcs)

        # Training PC scores: shape (n_tr, k)
        X_tr = U[:, :k] * np.sqrt(np.maximum(ev[:k], 0))
        self._ev = ev; self._U = U; self._k = k

        X_t = _to_gpu(X_tr.astype(np.float32))
        y_t = torch.tensor(y_tr.astype(np.float32),
                           dtype=torch.float32, device=DEVICE)

        # Build network: k → hidden[0] → hidden[1] → 1
        layers = []
        in_dim = k
        for h in self.hidden:
            layers += [nn.Linear(in_dim, h),
                       nn.BatchNorm1d(h),
                       nn.ReLU(),
                       nn.Dropout(self.dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self._model = nn.Sequential(*layers).to(DEVICE)

        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.n_epochs)
        loss_fn = nn.MSELoss()

        best_loss, best_state = np.inf, None
        for epoch in range(self.n_epochs):
            self._model.train()
            optimizer.zero_grad()
            pred = self._model(X_t).squeeze()
            loss = loss_fn(pred, y_t)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {kk: vv.clone()
                              for kk, vv in self._model.state_dict().items()}

        if best_state:
            self._model.load_state_dict(best_state)
        return self

    def predict(self, G_val_tr):
        ev, U, k = self._ev, self._U, self._k
        # Validation projection onto training PCs
        X_val = G_val_tr @ U[:, :k] / (np.sqrt(np.maximum(ev[:k], 0)) + 1e-9)
        X_val_t = _to_gpu(X_val.astype(np.float32))
        self._model.eval()
        with torch.no_grad():
            pred = self._model(X_val_t).squeeze().cpu().numpy()
        return np.asarray(pred, dtype=np.float64)

ALL_METHODS=list(STRATEGIES)+['BayesB','KRR_RBF','PCR','DNNGP']

def nested_cv_baselines(G,y,methods=None,n_outer=5,seed=0,verbose=False):
    if methods is None: methods=ALL_METHODS
    n=len(y); kf=KFold(n_splits=n_outer,shuffle=True,random_state=seed); res={m:[] for m in methods}
    for fi,(tr,val) in enumerate(kf.split(np.arange(n))):
        G_tr=G[np.ix_(tr,tr)]; G_val=G[np.ix_(val,tr)]; sc=StandardScaler()
        y_s=sc.fit_transform(y[tr].reshape(-1,1)).ravel()
        if verbose: print(f"  fold {fi+1}/{n_outer}",end='  ')
        for method in methods:
            try:
                if method in STRATEGIES: est=SASBLUPEstimator(strategy=method); est.fit(G_tr,y_s); p_s=est.predict(G_val)
                elif method=='BayesB': est=BayesBPredictor(); est.fit(G_tr,y_s); p_s=est.predict(G_val)
                elif method=='KRR_RBF': est=KRR_RBF_Predictor(); est.fit(G_tr,y_s); p_s=est.predict(G_val)
                elif method=='PCR': est=PCRPredictor(); est.fit(G_tr,y_s); p_s=est.predict(G_val)
                elif method=='DNNGP': est=DNNGPPredictor(seed=seed); est.fit(G_tr,y_s); p_s=est.predict(G_val)
                pred=sc.inverse_transform(p_s.reshape(-1,1)).ravel(); m=compute_metrics(y[val],pred)
            except Exception as e:
                if verbose: print(f"\n  [{method}] ERR: {e}")
                m=dict(r2=np.nan,r=np.nan,rmse=np.nan,mae=np.nan)
            res[method].append(m)
            if verbose: print(f"{method}={m['r2']:.3f}",end='  ')
        if verbose: print()
        _clear_loo_cache()
    return res

def apply_fdr_correction(all_base, trait_names, ref='scalar'):
    """
    Collect all p-values across traits × methods,
    apply FDR (BH) correction, return corrected p-value table.
    """
    records = []
    for trait in trait_names:
        df = summarise_cv(all_base[trait], ref=ref)
        for _, row in df.iterrows():
            if row['method'] != ref and not np.isnan(row.get('p_ttest', np.nan)):
                records.append({
                    'trait'   : trait,
                    'method'  : row['method'],
                    'r2_mean' : row['r2_mean'],
                    'delta_r2': row['delta_r2'],
                    'p_raw'   : row['p_ttest']
                })

    if not records:
        print("  No p-values to correct."); return pd.DataFrame()

    df_p = pd.DataFrame(records)
    reject, p_fdr, _, _ = multipletests(df_p['p_raw'].values, method='fdr_bh')
    df_p['p_fdr']    = p_fdr
    df_p['rejected'] = reject   # True = significant after FDR

    df_p.to_csv(TAB_DIR / 'table_fdr_corrected.csv', index=False)
    print("\n  FDR-corrected p-values:")
    print(f"  {'Trait':<35} {'Method':<12} {'p_raw':>8} "
          f"{'p_FDR':>8} {'Sig?':>6}")
    print(f"  {'-'*72}")
    for _, row in df_p.iterrows():
        sig = '✓' if row['rejected'] else '✗'
        short = row['trait'].replace('Brown_','').replace('_mm','')\
                            .replace('ht_cm_Yan_et_al_2007','Height')
        print(f"  {short:<35} {row['method']:<12} "
              f"{row['p_raw']:>8.4f} {row['p_fdr']:>8.4f} {sig:>6}")
    return df_p

def cohens_d(a, b):
    """Pooled Cohen's d between arrays a and b."""
    a, b = np.asarray(a), np.asarray(b)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2: return np.nan
    s_pool = np.sqrt(((na-1)*np.var(a, ddof=1) +
                      (nb-1)*np.var(b, ddof=1)) / (na + nb - 2))
    return float((np.mean(a) - np.mean(b)) / (s_pool + 1e-12))

def compute_effect_sizes(all_base, trait_names, ref='scalar'):
    """Cohen's d: SAS-BLUP strategy R² vs scalar R² per trait."""
    rows = []
    sas_strats = [s for s in STRATEGIES if s != ref]
    for trait in trait_names:
        ref_r2s = np.array([m['r2'] for m in all_base[trait][ref]
                            if not np.isnan(m['r2'])])
        short = trait.replace('Brown_','').replace('_mm','')\
                     .replace('ht_cm_Yan_et_al_2007','Height')
        for strat in sas_strats:
            cand_r2s = np.array([m['r2'] for m in all_base[trait].get(strat,[])
                                 if not np.isnan(m['r2'])])
            d = cohens_d(cand_r2s, ref_r2s)
            rows.append(dict(trait=short, method=strat,
                             mean_ref=float(np.mean(ref_r2s)),
                             mean_cand=float(np.mean(cand_r2s)) if len(cand_r2s) else np.nan,
                             delta=float(np.mean(cand_r2s)-np.mean(ref_r2s)) if len(cand_r2s) else np.nan,
                             cohens_d=d,
                             magnitude=('negligible' if abs(d)<0.2 else
                                        'small'  if abs(d)<0.5 else
                                        'medium' if abs(d)<0.8 else 'large')))
    df_es = pd.DataFrame(rows)
    df_es.to_csv(TAB_DIR / 'table_effect_sizes.csv', index=False)
    print("\n  Effect sizes (Cohen's d):")
    print(f"  {'Trait':<20} {'Method':<12} {'Δ R²':>8} {'d':>8} {'Magnitude'}")
    print(f"  {'-'*60}")
    for _, row in df_es.iterrows():
        print(f"  {row['trait']:<20} {row['method']:<12} "
              f"{row['delta']:>+8.4f} {row['cohens_d']:>8.3f}  {row['magnitude']}")
    return df_es

def bootstrap_r2_ci(all_base, trait_names,
                    n_boot=1000, ci=0.95, seed=42):
    """Bootstrap confidence intervals for mean R² per method per trait."""
    rng  = np.random.RandomState(seed)
    rows = []
    alpha = 1 - ci
    for trait in trait_names:
        short = trait.replace('Brown_','').replace('_mm','')\
                     .replace('ht_cm_Yan_et_al_2007','Height')
        for method in ALL_METHODS:
            r2s = np.array([m['r2'] for m in all_base[trait].get(method, [])
                            if not np.isnan(m['r2'])])
            if len(r2s) < 3:
                rows.append(dict(trait=short, method=method,
                                 mean=np.nan, ci_lo=np.nan, ci_hi=np.nan))
                continue
            boot = [np.mean(rng.choice(r2s, len(r2s), replace=True))
                    for _ in range(n_boot)]
            rows.append(dict(trait=short, method=method,
                             mean=float(np.mean(r2s)),
                             ci_lo=float(np.percentile(boot, alpha/2*100)),
                             ci_hi=float(np.percentile(boot, (1-alpha/2)*100))))
    df_ci = pd.DataFrame(rows)
    df_ci.to_csv(TAB_DIR / 'table_bootstrap_ci.csv', index=False)
    print("\n  Bootstrap 95% CI for R² (first 10 rows):")
    print(df_ci.head(10).to_string(index=False))
    return df_ci
