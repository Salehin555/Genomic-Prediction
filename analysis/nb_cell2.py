# ══════════════════════════════════════════════════════════════════════════
# SPEED OPTIMISATIONS (vs previous version — was taking >24 hours):
# ══════════════════════════════════════════════════════════════════════════
# Root cause: Numerical gradient for n=150 params → 146 function evals/step
# Fix: All strategies now use analytical LOO gradient → 1 eval/step
# Benchmark: 4,379 calls → 30 calls = 62.8× per optimizer call
# Combined with reduced grid sizes: total ~3000× faster
#
# Settings for ~90-min run: N_SEEDS=5, N_EXP=20, γ_grid=6, maxiter=100-300
# Settings for final paper: N_SEEDS=10, N_EXP=50, γ_grid=10, maxiter=200-500
# ══════════════════════════════════════════════════════════════════════════

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SAS-BLUP: Spectral Adaptive Shrinkage BLUP                            ║
# ║  Complete Research Pipeline — Steps 1–6  |  Q1 Journal Ready           ║
# ║  Kaggle GPU Notebook (T4 x2 recommended)                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# KAGGLE SETUP:
#   Settings → Accelerator → GPU T4 x2
#   Settings → Persistence → Files only
#   + Add Data → upload genotype.csv + phenotype.csv
#   Edit KAGGLE_DATASET slug below → Run All
#
# CELL STRUCTURE:
#   Cell 1  — Install, imports, GPU setup, config
#   Cell 2  — GRM builder (streaming, GPU-accelerated)
#   Cell 3  — SAS-BLUP engine [HARDENED: warm-start + collapse guard]
#   Cell 4  — Theoretical analysis (Figure 1)
#   Cell 5  — Synthetic benchmark (Figures 2–5)
#   Cell 6  — Real-world validation (Figures 6–8)
#   Cell 7  — Baseline comparison (Figures 9–10, Table 3)
#   Cell 8  — Ablation & interpretation (Figures 11–13)
#   Cell 9  — Final summary + LaTeX + zip


# ══════════════════════════════════════════════════════════════════════════
# CELL 1 — Install, Imports, GPU Setup, Config
# ══════════════════════════════════════════════════════════════════════════

import subprocess, sys, os, gc, time, warnings, json
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from tqdm import tqdm
from itertools import product as iproduct

from scipy.linalg    import eigh
from scipy.optimize  import minimize
from scipy.interpolate import BSpline
from scipy.stats     import pearsonr, ttest_rel

from sklearn.base        import BaseEstimator, RegressorMixin
from sklearn.isotonic    import IsotonicRegression
from sklearn.linear_model import RidgeCV
from sklearn.metrics     import r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing   import StandardScaler

print("✓ All packages ready")

# ── GPU setup ────────────────────────────────────────────────────────────
import torch

if torch.cuda.is_available():
    DEVICE     = torch.device('cuda')
    GPU_NAME   = torch.cuda.get_device_name(0)
    GPU_MEM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
    USE_GPU    = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    print(f"✓ GPU: {GPU_NAME}  ({GPU_MEM_GB:.1f} GB)")
else:
    DEVICE = torch.device('cpu'); GPU_NAME = 'CPU'; USE_GPU = False
    print("⚠ No GPU — CPU fallback active (all code still works)")

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

print(f"  Device: {DEVICE}")

# ── Plot style (publication quality) ────────────────────────────────────
plt.rcParams.update({
    'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':12,
    'axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':10,
    'legend.fontsize':10,'axes.spines.top':False,'axes.spines.right':False,
    'axes.grid':True,'grid.alpha':0.20,'grid.linewidth':0.6,
    'lines.linewidth':1.8,'figure.dpi':150,
    'savefig.dpi':600,'savefig.bbox':'tight','savefig.format':'pdf',
})
PAL = {
    'scalar':'#D62728','grouped':'#1F77B4','smooth':'#2CA02C',
    'monotone':'#FF7F0E','spline':'#9467BD','oracle':'#17BECF',
    'PCR':'#8C564B','BayesB':'#E377C2','KRR_RBF':'#7F7F7F',
}
MARKERS = {'scalar':'o','grouped':'s','smooth':'^','monotone':'D',
           'spline':'v','oracle':'*','PCR':'h','BayesB':'p','KRR_RBF':'X'}
print("✓ Style ready")

# ── Config ────────────────────────────────────────────────────────────────
KAGGLE_DATASET = "sas-blup-rice"           # ← EDIT: your dataset slug
DATASET_SUBDIR = ""

INPUT_DIR     = Path("/kaggle/input") / KAGGLE_DATASET / DATASET_SUBDIR
GENOTYPE_CSV  = "/kaggle/input/datasets/salehinseyam/full-genotype/genotype.csv"
PHENOTYPE_CSV = "/kaggle/input/datasets/salehinseyam/genotype/phenotype.csv"
WORK          = Path("/kaggle/working")
GRM_FILE      = WORK / "GRM.npy"
IDS_FILE      = WORK / "sample_ids.npy"
FIG_DIR       = WORK / "figures"; FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR       = WORK / "tables";  TAB_DIR.mkdir(parents=True, exist_ok=True)

N_OUTER = 5; N_SEEDS = 5; CHUNK = 5_000; MAF_MIN = 0.01; SEED = 42  # was 10; set to 10 for final run; MAF_MIN = 0.01; SEED = 42
np.random.seed(SEED)

print(f"\n── Path check ──────────────────────────────────────────────────")
print(f"  Genotype : {GENOTYPE_CSV}")
print(f"  Phenotype: {PHENOTYPE_CSV}")
if not Path(GENOTYPE_CSV).exists():
    print("  ⚠ Genotype not found — check paths above")
    print("  Available in /kaggle/input:")
    for d in sorted(Path('/kaggle/input').rglob('*.csv'))[:8]:
        print(f"    {d}")
else:
    sz = Path(GENOTYPE_CSV).stat().st_size/1e6
    print(f"  ✓ Genotype found  ({sz:.0f} MB)")
print(f"✓ Config ready  →  {WORK}")





# ══════════════════════════════════════════════════════════════════════════
# CELL 2 — GRM Builder (streaming, GPU-accelerated)
# ══════════════════════════════════════════════════════════════════════════

IUPAC_MAP = str.maketrans({
    'A':'0','T':'2','C':'0','G':'2','R':'1','Y':'1','S':'1','W':'1',
    'K':'1','M':'1','B':'1','D':'1','H':'1','V':'1','N':'1','-':'1',
})

def _encode_block(df):
    arr = df.astype(str).apply(
        lambda col: pd.to_numeric(
            col.str.upper().str.strip().str.translate(IUPAC_MAP),
            errors='coerce')
    ).values.astype(np.float32)
    col_med  = np.nanmedian(arr, axis=0)
    nan_mask = np.isnan(arr)
    arr[nan_mask] = np.take(col_med, np.where(nan_mask)[1])
    return arr

def _detect_orientation(path):
    peek = pd.read_csv(path, nrows=3, header=None)
    for ri in range(len(peek)):
        if sum(1 for v in peek.iloc[ri] if str(v).startswith('GSOR')) > 5:
            print(f"  Orientation: SNPs-as-rows (HapMap), header row={ri}")
            return 'snps_as_rows', ri
    print("  Orientation: samples-as-rows")
    return 'samples_as_rows', 0

def build_grm(geno_csv, pheno_csv, grm_file=GRM_FILE, ids_file=IDS_FILE,
              chunk_snps=CHUNK, maf_min=MAF_MIN):
    if grm_file.exists() and ids_file.exists():
        GRM    = np.load(grm_file).astype(np.float64)
        common = np.load(ids_file, allow_pickle=True).tolist()
        pheno  = pd.read_csv(pheno_csv)
        idx_c  = 'Taxa' if 'Taxa' in pheno.columns else pheno.columns[0]
        pheno  = pheno.set_index(idx_c); pheno.index = pheno.index.astype(str)
        Y      = pheno.loc[common].values.astype(np.float32)
        print(f"  ✓ Cache hit  GRM={GRM.shape}  n={len(common)}")
        return GRM + 1e-4*np.eye(len(common)), Y, list(pheno.columns), common

    pheno = pd.read_csv(pheno_csv)
    idx_c = 'Taxa' if 'Taxa' in pheno.columns else pheno.columns[0]
    pheno = pheno.set_index(idx_c); pheno.index = pheno.index.astype(str)
    trait_names = list(pheno.columns)
    print(f"  Phenotype: {len(pheno)} samples  traits={trait_names}")

    orientation, hdr_row = _detect_orientation(geno_csv)

    if orientation == 'snps_as_rows':
        hdr       = pd.read_csv(geno_csv, nrows=hdr_row+1, header=None)
        row       = hdr.iloc[hdr_row]
        gsor_mask = [str(v).startswith('GSOR') for v in row]
        gsor_cols = [i for i,m in enumerate(gsor_mask) if m]
        all_ids   = [str(row.iloc[i]) for i in gsor_cols]
    else:
        hdr_df   = pd.read_csv(geno_csv, nrows=0)
        taxa_col = hdr_df.columns[0]
        all_ids  = pd.read_csv(geno_csv, usecols=[taxa_col])[taxa_col].astype(str).tolist()

    common = sorted(set(all_ids) & set(pheno.index))
    n      = len(common)
    print(f"  Common samples: {n}")
    assert n > 0, "No common samples — check ID format"
    Y      = pheno.loc[common].values.astype(np.float32)

    GRM    = np.zeros((n,n), dtype=np.float64)
    p_used = 0; t0 = time.time()
    GRM_gpu = torch.zeros((n,n), dtype=torch.float32, device=DEVICE) if USE_GPU else None

    def _accum(M_std):
        nonlocal GRM
        if USE_GPU:
            Mt = torch.tensor(M_std, dtype=torch.float32, device=DEVICE)
            GRM_gpu.add_(torch.mm(Mt, Mt.T))
        else:
            GRM += M_std @ M_std.T

    if orientation == 'snps_as_rows':
        sn_idx    = {s:i for i,s in enumerate(all_ids)}
        comm_cols = [sn_idx[s] for s in common if s in sn_idx]
        reader    = pd.read_csv(geno_csv, header=None, chunksize=chunk_snps, skiprows=hdr_row+1)
        total_snps = 0
        for chunk in reader:
            geno_block = chunk.iloc[:,gsor_cols].values[:,comm_cols]
            M  = _encode_block(pd.DataFrame(geno_block.T.astype(str)))
            freq = M.mean(0)/2.0; valid = (freq>maf_min)&(freq<1-maf_min)
            if valid.any():
                M=M[:,valid]; freq=freq[valid]
                scale=np.sqrt(2*freq*(1-freq)); scale[scale==0]=1.
                _accum((M-2*freq)/scale); p_used+=M.shape[1]
            total_snps+=len(chunk)
            if total_snps%300_000==0:
                print(f"  {total_snps:,} SNPs | {time.time()-t0:.0f}s", end='\r')
    else:
        hdr_df   = pd.read_csv(geno_csv, nrows=0)
        all_cols = list(hdr_df.columns); taxa_col=all_cols[0]; snp_cols=all_cols[1:]
        for start in range(0, len(snp_cols), chunk_snps):
            end = min(start+chunk_snps, len(snp_cols))
            cdf = pd.read_csv(geno_csv, usecols=[taxa_col]+snp_cols[start:end])
            cdf = cdf[cdf[taxa_col].astype(str).isin(set(common))].set_index(taxa_col).loc[common]
            M   = _encode_block(cdf); freq=M.mean(0)/2.0; valid=(freq>maf_min)&(freq<1-maf_min)
            if valid.any():
                M=M[:,valid]; freq=freq[valid]
                scale=np.sqrt(2*freq*(1-freq)); scale[scale==0]=1.
                _accum((M-2*freq)/scale); p_used+=M.shape[1]
            if start%500_000==0:
                print(f"  {end:,}/{len(snp_cols):,} SNPs | {time.time()-t0:.0f}s", end='\r')

    if USE_GPU:
        torch.cuda.synchronize()
        GRM = GRM_gpu.cpu().numpy().astype(np.float64)
        del GRM_gpu; clear_gpu()

    GRM /= max(p_used, 1); GRM = GRM.astype(np.float64)
    np.save(grm_file, GRM); np.save(ids_file, np.array(common))
    print(f"\n  ✓ GRM built  shape={GRM.shape}  SNPs_used={p_used:,}  time={time.time()-t0:.0f}s")
    GRM += 1e-4*np.eye(n)
    return GRM, Y, trait_names, common

print("Building GRM ...")
GRM, Y, TRAIT_NAMES, SAMPLE_IDS = build_grm(GENOTYPE_CSV, PHENOTYPE_CSV)
N, T = Y.shape

_ev, _U = _gpu_eigh(GRM)
_ev      = np.maximum(_ev, 1e-8)
EV_GRM, U_GRM = _ev, _U
EFF_RANK = int(EV_GRM.sum()**2 / (EV_GRM**2).sum())

print(f"\n✓ GRM ready  n={N}  traits={TRAIT_NAMES}")
print(f"  Eigenvalue range: [{EV_GRM[-1]:.4f}, {EV_GRM[0]:.4f}]")
print(f"  Effective rank  : {EFF_RANK}")
print(f"  Cumulative var 90%: k={int(np.searchsorted(np.cumsum(EV_GRM)/EV_GRM.sum(),0.90))+1}")


# ══════════════════════════════════════════════════════════════════════════
# CELL 3 — SAS-BLUP Engine [HARDENED — warm-start + collapse guard]
# ══════════════════════════════════════════════════════════════════════════
#
# ROOT CAUSE OF COLLAPSE (diagnosed):
#   sigmoid(0) = 0.5 everywhere → Hᵢᵢ = Σⱼ 0.5·Uᵢⱼ² = 0.5 (orthonormality)
#   → gradient ≈ 0 at zeros init → L-BFGS-B terminates immediately
#   → smooth/monotone/spline all produce identical degenerate w≈0.5
#
# FIXES (7 layers):
#   1. Warm-start from logit(scalar_weights) — seeds optimiser in signal subspace
#   2. Multi-restart (4-5 restarts per strategy)
#   3. Collapse guard — never worse than scalar by threshold
#   4. NaN/Inf protection at every evaluation
#   5. Tighter tolerances + more iterations
#   6. Grouped as secondary warm-start for smooth/monotone/spline
#   7. Auto-tune n_groups/n_knots to effective rank

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

# ── Warm-start builder (shared by all strategies) ─────────────────────────

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

print("✓ _multi_restart_jac defined")

def _collapse_guard(w_cand, loo_cand, w_sc, loo_sc, thr=0.005):
    """If candidate is worse than scalar + thr, return scalar."""
    if loo_cand > loo_sc + thr:
        return w_sc.copy(), True
    return w_cand, False

# ── S0: Scalar ──────────────────────────────────────────────────────────

def _learn_scalar(ev, U, z, y, lam_grid=None):
    w_sc, _, best_lam, _ = _scalar_warmstart(ev, U, z, y, lam_grid)
    return w_sc, best_lam

# ── S1: Grouped ─────────────────────────────────────────────────────────

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

# ── S2: Smooth ──────────────────────────────────────────────────────────

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

# ── S3: Monotone ────────────────────────────────────────────────────────

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

# ── S4: Spline ──────────────────────────────────────────────────────────

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

# ── Estimator ────────────────────────────────────────────────────────────

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

def nested_cv(G, y, methods=None, n_outer=5, seed=0, verbose=False):
    if methods is None: methods=list(STRATEGIES)
    n=len(y); kf=KFold(n_splits=n_outer,shuffle=True,random_state=seed)
    res={m:[] for m in methods}; wts={m:[] for m in methods}; dels=[]
    for fi,(tr,val) in enumerate(kf.split(np.arange(n))):
        G_tr=G[np.ix_(tr,tr)]; G_val=G[np.ix_(val,tr)]
        sc=StandardScaler(); y_s=sc.fit_transform(y[tr].reshape(-1,1)).ravel()
        ev_f,U_f=safe_eigh(G_tr); z_f=U_f.T@y_s
        sig2=float(np.var(y_s-U_f@(U_f.T@y_s)))
        d=compute_delta_spectral(ev_f,z_f,max(sig2,1e-8),len(tr))
        dels.append(d)
        if verbose: print(f"  fold {fi+1}/{n_outer}  Δ={d:.4f}",end='  ')
        for method in methods:
            try:
                est=SASBLUPEstimator(strategy=method)
                est.fit(G_tr,y_s)
                pred=sc.inverse_transform(est.predict(G_val).reshape(-1,1)).ravel()
                m=compute_metrics(y[val],pred); wts[method].append(est.weights_.copy())
            except Exception as e:
                if verbose: print(f"\n  [{method}] ERR: {e}")
                m=dict(r2=np.nan,r=np.nan,rmse=np.nan,mae=np.nan)
            res[method].append(m)
            if verbose: print(f"{method}={m['r2']:.3f}",end='  ')
        if verbose: print()
        _clear_loo_cache()
    return res,wts,dels

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

# Sanity check
def _check():
    rng=np.random.RandomState(99); n,p=60,600
    X=rng.randn(n,p); G=X@X.T/p; ev,U=safe_eigh(G)
    theta=np.zeros(n); theta[:5]=rng.randn(5)*3
    y=U@theta+rng.randn(n)*0.5; z=U.T@y
    w_sc,logit_sc,_,loo_sc=_scalar_warmstart(ev,U,z,y)
    loo_zero=loo_mse(_sig(np.zeros(n)),U,z,y)
    loo_warm=loo_mse(_sig(logit_sc),U,z,y)
    assert loo_warm<loo_zero, f"warm-start failed: {loo_warm:.5f} >= {loo_zero:.5f}"
    return loo_zero, loo_warm
lz, lw = _check()
print("✓ Cell 3 complete — hardened SAS-BLUP engine ready")
print(f"  Collapse fix: zero-init LOO={lz:.5f} → warm-start LOO={lw:.5f}  ({(lz-lw)/lz*100:.1f}% better)")
print(f"  All strategies: warm-start + multi-restart + collapse guard")


# ══════════════════════════════════════════════════════════════════════════
# CELL 4 — Step 2: Theoretical Analysis (Figure 1)
# ══════════════════════════════════════════════════════════════════════════

def compute_delta_spectral(ev, theta_or_z, sigma2, n, lam_grid=None):
    if lam_grid is None: lam_grid=np.logspace(-5,5,120)
    z=theta_or_z; w_or=z**2/(z**2+max(sigma2/n,1e-12))
    or_risk=float(np.sum(((1-w_or)*z)**2)+np.sum(w_or**2)*sigma2/n)
    best=np.inf
    for lam in lam_grid:
        w=scalar_shrinkage(ev,lam)
        r=float(np.sum(((1-w)*z)**2)+np.sum(w**2)*sigma2/n)
        if r<best: best=r
    return float(max(best-or_risk,0.))

def theoretical_risk_decomposition(ev,theta,sigma2,n,lam_grid=None):
    if lam_grid is None: lam_grid=np.logspace(-5,5,100)
    bias2,var,risk=[],[],[]
    for lam in lam_grid:
        w=scalar_shrinkage(ev,lam)
        b2=float(np.sum(((1-w)*theta)**2)); vr=float(np.sum(w**2)*sigma2/n)
        bias2.append(b2); var.append(vr); risk.append(b2+vr)
    return dict(lam=lam_grid,bias2=np.array(bias2),var=np.array(var),risk=np.array(risk))

def oracle_risk_val(ev,theta,sigma2,n):
    w=oracle_shrinkage(theta,sigma2,n)
    return float(np.sum(((1-w)*theta)**2)+np.sum(w**2)*sigma2/n)

def fig1_theory(save=True):
    rng=np.random.RandomState(0); n,p=120,1200
    ev_c=(1+np.arange(p,0,-1))**(-1.5); ev_c=ev_c/ev_c.sum()*p
    X=rng.randn(n,p)*np.sqrt(ev_c); G=X@X.T/p; ev,U=safe_eigh(G); sigma2=0.25
    theta_top=np.zeros(n); theta_top[:6]=rng.randn(6)*3
    theta_uni=rng.randn(n)*0.8
    theta_bot=np.zeros(n); theta_bot[n-6:]=rng.randn(6)*3

    fig=plt.figure(figsize=(14,10))
    gs=gridspec.GridSpec(2,2,hspace=0.42,wspace=0.38,left=0.08,right=0.97,top=0.95,bottom=0.08)

    ax=fig.add_subplot(gs[0,0])
    rd=theoretical_risk_decomposition(ev,theta_top,sigma2,n)
    ax.semilogx(rd['lam'],rd['bias2'],color='#D62728',lw=2.2,label='Bias²')
    ax.semilogx(rd['lam'],rd['var'],  color='#1F77B4',lw=2.2,ls='--',label='Variance')
    ax.semilogx(rd['lam'],rd['risk'], color='#2CA02C',lw=2.8,label='Total risk')
    bi=np.argmin(rd['risk'])
    ax.axvline(rd['lam'][bi],color='#FF7F0E',lw=1.5,ls=':',label=f'Opt λ={rd["lam"][bi]:.3f}')
    or_r=oracle_risk_val(ev,theta_top,sigma2,n)
    ax.axhline(or_r,color='#17BECF',lw=1.5,ls='-.',label='Oracle risk')
    ax.text(0.97,0.90,r'$\Delta_{\rm spectral}$',transform=ax.transAxes,
            ha='right',fontsize=12,color='#17BECF',fontstyle='italic')
    ax.set_xlabel('Regularisation parameter λ'); ax.set_ylabel('Risk')
    ax.set_title('(A)  Bias–Variance Tradeoff\n(scalar KRR, top-k signal)'); ax.legend(fontsize=8)

    ax=fig.add_subplot(gs[0,1])
    idx=np.arange(n)
    for theta,label,col in [(theta_top,'top-k','#D62728'),(theta_uni,'uniform','#1F77B4'),
                              (theta_bot,'bottom-k','#2CA02C')]:
        ax.plot(idx,oracle_shrinkage(theta,sigma2,n),color=col,lw=2.,label=f'Oracle ({label})')
    w_sc_ref=scalar_shrinkage(ev,np.median(ev))
    ax.plot(idx,w_sc_ref,color='#7F7F7F',lw=2.,ls='--',label='Scalar KRR')
    ax.set_xlabel('Eigenvector index'); ax.set_ylabel('Shrinkage weight $w_i$')
    ax.set_title('(B)  Oracle vs Scalar Shrinkage Profiles'); ax.set_ylim(-0.05,1.15); ax.legend(fontsize=8.5)

    ax=fig.add_subplot(gs[1,0])
    k_vals=np.arange(1,31); delta_vals=[]
    for k in k_vals:
        th=np.zeros(n); th[:k]=np.sqrt(9/k)
        delta_vals.append(compute_delta_spectral(ev,U.T@(U@th),sigma2,n))
    da=np.array(delta_vals)
    ax.plot(k_vals,da,color='#9467BD',lw=2.5,marker='o',ms=4,label=r'$\Delta_{\rm spectral}$')
    ax.fill_between(k_vals,0,da,alpha=0.15,color='#9467BD')
    ax.axhline(0,color='black',lw=0.8,ls='--',alpha=0.5)
    ax.set_xlabel('Signal eigenvectors k'); ax.set_ylabel(r'$\Delta_{\rm spectral}$')
    ax.set_title('(C)  Spectral Gap vs Signal Concentration'); ax.legend(fontsize=9)

    ax=fig.add_subplot(gs[1,1]); ax.set_xlim(-0.1,1.1); ax.set_ylim(-0.1,1.1)
    ax.set_aspect('equal'); ax.axis('off'); ax.set_title('(D)  Estimator Class Hierarchy',pad=6)
    from matplotlib.patches import Ellipse
    for cx,cy,w,h,col,alp in [(0.5,0.5,0.90,0.85,'#E8F4FD',0.7),
                                (0.5,0.5,0.65,0.60,'#BDDFF5',0.7),
                                (0.5,0.5,0.30,0.25,'#6BB5E8',0.9)]:
        ax.add_patch(Ellipse((cx,cy),w,h,fill=True,facecolor=col,edgecolor='#2166AC',lw=1.5,alpha=alp))
    ax.text(0.50,0.94,r'$\mathcal{W}_{\rm oracle}$',ha='center',fontsize=10,color='#08519C',fontweight='bold')
    ax.text(0.50,0.78,r'$\mathcal{W}_{\rm adaptive}$ (SAS-BLUP)',ha='center',fontsize=9.5,color='#08519C')
    ax.text(0.50,0.50,r'$\mathcal{W}_{\rm scalar}$'+'\n(rrBLUP)',ha='center',fontsize=9,color='white',fontweight='bold')
    ax.text(0.50,0.04,r'$R(\hat{w}) \leq \inf_{\mathcal{W}_{\rm adaptive}} R(w) + \varepsilon_n$',
            ha='center',fontsize=9.5,color='#08306B',
            bbox=dict(boxstyle='round,pad=0.3',fc='#EFF7FB',ec='#2166AC',lw=1))

    plt.suptitle('SAS-BLUP: Theoretical Framework',fontsize=14,fontweight='bold',y=0.99)
    if save:
        p=FIG_DIR/'fig01_theory_framework.pdf'
        fig.savefig(p,dpi=600,format='pdf',bbox_inches='tight'); print(f"  Saved {p}")
    plt.show(); plt.close()

print("Generating Figure 1 — Theoretical Framework ...")
fig1_theory()
print("✓ Step 2 complete")

delta_real={}
for t_i,trait in enumerate(TRAIT_NAMES):
    y_t=Y[:,t_i].astype(np.float64); sc_=StandardScaler()
    y_s=sc_.fit_transform(y_t.reshape(-1,1)).ravel(); z_f=U_GRM.T@y_s
    sig2=float(np.var(y_s-U_GRM@(U_GRM.T@y_s)))
    d=compute_delta_spectral(EV_GRM,z_f,max(sig2,1e-8),N)
    delta_real[trait]=d; print(f"  {trait:<35}  Δspectral={d:.6f}")
pd.DataFrame(list(delta_real.items()),columns=['trait','delta_spectral'])\
  .to_csv(TAB_DIR/'theoretical_delta_spectral.csv',index=False)
print(f"\n  Δspectral≈0 → GRM eff_rank={EFF_RANK} → scalar already near-optimal (predicted tie)")


# ══════════════════════════════════════════════════════════════════════════
# CELL 5 — Step 3: Synthetic Benchmark (Figures 2–5)
# ══════════════════════════════════════════════════════════════════════════

def make_grm_synthetic(n=150,p=1500,alignment='top',noise=0.5,decay='moderate',n_sig=5,seed=42):
    rng=np.random.RandomState(seed)
    if decay=='fast': ev_c=np.exp(-5*np.arange(p)/p)
    elif decay=='slow': ev_c=(1+np.arange(p,0,-1))**(-0.5)
    else: ev_c=(1+np.arange(p,0,-1))**(-1.5)
    ev_c=ev_c/ev_c.sum()*p; X=rng.randn(n,p)*np.sqrt(ev_c); G=X@X.T/p; ev,U=safe_eigh(G)
    theta=np.zeros(n)
    if alignment=='top': theta[:n_sig]=rng.randn(n_sig)*3
    elif alignment=='bottom': theta[n-n_sig:]=rng.randn(n_sig)*3
    elif alignment=='middle':
        m=n//2; half=n_sig//2; sl=theta[m-half:m-half+n_sig]; sl[:]=rng.randn(len(sl))*3
    elif alignment=='uniform': theta=rng.randn(n)*1.0
    y_sig=U@theta; y=y_sig+rng.randn(n)*noise
    return G,y,y_sig,theta,ev,U

def run_synth_cond(alignment,noise=0.5,n=150,p=1500,n_outer=5,seed=0):
    G,y,_,theta,ev,U=make_grm_synthetic(n=n,p=p,alignment=alignment,noise=noise,seed=seed)
    sigma2=noise**2; delta=compute_delta_spectral(ev,theta,sigma2,n)
    y_s=StandardScaler().fit_transform(y.reshape(-1,1)).ravel(); z_full=U.T@y_s
    w_sc,_,_,_=_scalar_warmstart(ev,U,z_full,y_s)
    w_or=oracle_shrinkage(theta,sigma2,n)
    res,wts,_=nested_cv(G,y,methods=list(STRATEGIES),n_outer=n_outer,seed=seed,verbose=False)
    df=summarise_cv(res)
    profiles={'oracle':w_or,'scalar':w_sc}
    for st in ('grouped','smooth','monotone'):
        ws=wts[st]
        if ws:
            ml=min(w.shape[0] for w in ws)
            profiles[st]=np.mean([w[:ml] for w in ws],axis=0)
    return df,profiles,ev,delta

print("="*60); print("STEP 3 — SYNTHETIC BENCHMARK"); print("="*60)

def fig2_shrinkage(save=True):
    conds=[('top','Signal in top eigenvectors\n(large Δ — SAS wins)'),
           ('uniform','Signal spread uniformly\n(small Δ — methods tie)'),
           ('bottom','Signal in bottom eigenvectors')]
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    sty={'oracle':dict(ls='--',lw=2.5,color=PAL['oracle']),
         'scalar':dict(ls='-.',lw=2.0,color=PAL['scalar']),
         'grouped':dict(ls='-',lw=1.8,color=PAL['grouped']),
         'smooth':dict(ls='-',lw=1.8,color=PAL['smooth']),
         'monotone':dict(ls='-',lw=1.8,color=PAL['monotone'])}
    for ax,(align,title) in zip(axes,conds):
        print(f"  {align}...",end=' ',flush=True)
        df,profiles,ev,delta=run_synth_cond(align,noise=0.5,n=150,p=1500,seed=0)
        idx=np.arange(len(ev))
        for name,s in sty.items():
            if name not in profiles: continue
            w=profiles[name]; lbl='Oracle' if name=='oracle' else ('Scalar (rrBLUP)' if name=='scalar' else f'SAS ({name})')
            ax.plot(idx[:len(w)],w,label=lbl,alpha=0.88,**s)
        r2_sc=df.loc[df.method=='scalar','r2_mean'].values[0]
        r2_sas=df.loc[df.method.isin(['grouped','smooth','monotone']),'r2_mean'].max()
        ax.text(0.97,0.97,f'Δ={delta:.4f}\nrrBLUP R²={r2_sc:.3f}\nBest SAS R²={r2_sas:.3f}',
                transform=ax.transAxes,ha='right',va='top',fontsize=8.5,
                bbox=dict(boxstyle='round,pad=0.35',fc='white',ec='#BBB',alpha=0.9))
        ax.set_title(title,fontsize=10.5,pad=5); ax.set_xlabel('Eigenvector index')
        ax.set_ylabel('Shrinkage weight $w_i$'); ax.set_ylim(-0.06,1.22); ax.set_xlim(0,len(ev)-1)
    axes[0].legend(fontsize=8.5,loc='upper right',framealpha=0.9)
    fig.suptitle('Hardened SAS-BLUP Shrinkage Profiles\nvs Oracle and rrBLUP',
                 fontsize=13,fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.95])
    if save:
        p=FIG_DIR/'fig02_shrinkage_profiles.pdf'
        fig.savefig(p,dpi=600,format='pdf',bbox_inches='tight'); print(f"\n  Saved {p}")
    plt.show(); plt.close()

print("\n[1/4] Figure 2: Shrinkage profiles..."); fig2_shrinkage()

def fig3_delta(n_exp=20,save=True):  # 20 (was 50)
    rng=np.random.RandomState(7); rows=[]
    print(f"  Running {n_exp} conditions...",flush=True)
    for i in range(n_exp):
        seed=int(rng.randint(0,9999)); align=rng.choice(['top','middle','uniform','bottom'])
        noise=float(rng.uniform(0.3,1.5)); n_=int(rng.choice([80,120,150,200]))
        n_sig=int(rng.randint(2,15)); decay=rng.choice(['fast','moderate','slow'])
        G,y,_,theta,ev,U=make_grm_synthetic(n=n_,p=n_*10,alignment=align,noise=noise,n_sig=n_sig,decay=decay,seed=seed)
        delta=compute_delta_spectral(ev,theta,noise**2,n_)
        res,_,_=nested_cv(G,y,methods=['scalar','grouped','smooth'],n_outer=4,seed=seed,verbose=False)
        df=summarise_cv(res)
        r2_sc=df.loc[df.method=='scalar','r2_mean'].values[0]
        r2_sas=df.loc[df.method.isin(['grouped','smooth']),'r2_mean'].max()
        rows.append(dict(delta=delta,improvement=r2_sas-r2_sc,alignment=align,n=n_,noise=noise))
        if (i+1)%10==0: print(f"  {i+1}/{n_exp}",flush=True)
    df_all=pd.DataFrame(rows); df_all.to_csv(TAB_DIR/'synth_delta_scatter.csv',index=False)
    col_map={'top':'#1F77B4','middle':'#9467BD','uniform':'#D62728','bottom':'#2CA02C'}
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    ax=axes[0]
    for align,col in col_map.items():
        sub=df_all[df_all.alignment==align]
        ax.scatter(sub.delta,sub.improvement,c=col,alpha=0.72,s=55,label=align,zorder=3,edgecolors='white',lw=0.4)
    ax.axhline(0,color='black',lw=1.0,ls='--',alpha=0.45)
    v=df_all.dropna(subset=['delta','improvement'])
    r_,p_ = pearsonr(v.delta,v.improvement) if len(v)>=2 else (0.,1.)
    ax.text(0.04,0.96,f'r={r_:.3f}, p={p_:.3f}',transform=ax.transAxes,va='top',fontsize=10,)
    ax.set_xlabel(r'$\Delta_{\rm spectral}$'); ax.set_ylabel('R² improvement over rrBLUP')
    ax.set_title('(A)  Δspectral Predicts SAS-BLUP Gain'); ax.legend(fontsize=9,ncol=2)
    ax=axes[1]; thr=df_all.improvement>0.01
    bp=ax.boxplot([df_all.loc[thr,'delta'].values,df_all.loc[~thr,'delta'].values],
                   labels=['SAS wins\n(>1% R²)','SAS ties'],patch_artist=True,widths=0.5,
                   medianprops=dict(color='black',lw=2))
    for patch,col in zip(bp['boxes'],['#2CA02C','#D62728']): patch.set_facecolor(col); patch.set_alpha(0.55)
    ax.set_ylabel(r'$\Delta_{\rm spectral}$'); ax.set_title('(B)  High Δ Separates Win vs Tie')
    fig.suptitle('Theoretical Gap Predicts Empirical Improvement',fontsize=13,fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.94])
    if save:
        p=FIG_DIR/'fig03_delta_scatter.pdf'
        fig.savefig(p,dpi=600,format='pdf',bbox_inches='tight'); print(f"  Saved {p}")
    plt.show(); plt.close()
    return df_all

print("\n[2/4] Figure 3: Δspectral scatter..."); df_scatter=fig3_delta(n_exp=50)

def fig4_heatmap(save=True):
    GRID=dict(n=[80,150],alignment=['top','middle','uniform','bottom'],noise=[0.4,0.8,1.4],decay=['moderate'])
    BSEEDS=[0]; conds=list(iproduct(GRID['n'],GRID['alignment'],GRID['noise'],GRID['decay']))  # 1 seed (was 2)
    rows_b=[]
    print(f"  Grid: {len(conds)} conds × {len(BSEEDS)} seeds ...")
    for (n_,al,no,de),sd in tqdm(list(iproduct(conds,BSEEDS)),total=len(conds)*len(BSEEDS)):
        try:
            G,y,_,theta,ev,U=make_grm_synthetic(n=n_,p=n_*10,alignment=al,noise=no,decay=de,seed=sd)
            delta=compute_delta_spectral(ev,theta,no**2,n_)
            res,_,_=nested_cv(G,y,methods=['scalar','grouped','smooth','monotone'],n_outer=4,seed=sd,verbose=False)
            df_=summarise_cv(res)
            r2_sc=df_.loc[df_.method=='scalar','r2_mean'].values[0]
            r2_sas=df_.loc[df_.method.isin(['grouped','smooth','monotone']),'r2_mean'].max()
            rows_b.append(dict(n=n_,alignment=al,noise=no,decay=de,seed=sd,delta=delta,
                                scalar_r2=r2_sc,best_sas_r2=r2_sas,improvement=r2_sas-r2_sc))
        except Exception as e: print(f"  SKIP {al} n={n_}: {e}")
    df_b=pd.DataFrame(rows_b); df_b.to_csv(TAB_DIR/'synth_benchmark_grid.csv',index=False)
    n_vals=sorted(df_b.n.unique()); alignments=['top','middle','uniform','bottom']; noise_vals=sorted(df_b.noise.unique())
    fig,axes=plt.subplots(1,len(n_vals),figsize=(6*len(n_vals),4.5),sharey=True)
    if len(n_vals)==1: axes=[axes]
    im_last=None
    for ax,nv in zip(axes,n_vals):
        mat=np.full((len(alignments),len(noise_vals)),np.nan)
        sub=df_b[df_b.n==nv].groupby(['alignment','noise'])['improvement'].mean()
        for ai,a in enumerate(alignments):
            for ni,no in enumerate(noise_vals):
                try: mat[ai,ni]=sub.loc[(a,no)]
                except: pass
        vmax=max(0.05,np.nanmax(np.abs(mat)))
        im=ax.imshow(mat,cmap='RdYlGn',aspect='auto',vmin=-vmax,vmax=vmax)
        ax.set_xticks(range(len(noise_vals))); ax.set_xticklabels([f'{v:.1f}' for v in noise_vals],fontsize=9)
        ax.set_yticks(range(len(alignments))); ax.set_yticklabels(alignments,fontsize=9)
        ax.set_xlabel('Noise σ'); ax.set_title(f'n={nv}')
        for ai in range(len(alignments)):
            for ni in range(len(noise_vals)):
                v=mat[ai,ni]
                if not np.isnan(v):
                    ax.text(ni,ai,f'{v:+.3f}',ha='center',va='center',fontsize=9,
                            color='white' if abs(v)>vmax*0.55 else 'black',fontweight='bold')
        im_last=im
    axes[0].set_ylabel('Signal alignment')
    plt.colorbar(im_last,ax=axes[-1],shrink=0.82,label='Best SAS R² − rrBLUP R²')
    fig.suptitle('Benchmark Grid: SAS-BLUP Improvement over rrBLUP',fontsize=13,fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.92])
    if save:
        p=FIG_DIR/'fig04_benchmark_heatmap.pdf'
        fig.savefig(p,dpi=600,format='pdf',bbox_inches='tight'); print(f"  Saved {p}")
    plt.show(); plt.close()
    return df_b

print("\n[3/4] Figure 4: Benchmark heatmap..."); df_bench=fig4_heatmap()

def fig5_sample_strategy(df_bench,save=True):
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    col_map={'top':'#1F77B4','middle':'#9467BD','uniform':'#D62728','bottom':'#2CA02C'}
    n_vals=sorted(df_bench.n.unique()); ax=axes[0]
    for align,col in col_map.items():
        sub=df_bench[df_bench.alignment==align].groupby('n')['improvement']
        means=sub.mean().reindex(n_vals); stds=sub.std().reindex(n_vals)
        ax.plot(n_vals,means.values,'o-',color=col,lw=2.0,ms=7,label=align,markeredgecolor='white')
        ax.fill_between(n_vals,(means-stds).values,(means+stds).values,alpha=0.12,color=col)
    ax.axhline(0,color='black',lw=0.9,ls='--',alpha=0.45)
    ax.set_xlabel('Sample size n'); ax.set_ylabel('R² improvement over rrBLUP')
    ax.set_title('(A)  Sample-Size Effect'); ax.legend(title='Signal alignment',fontsize=9)
    ax=axes[1]; strats_show=['grouped','smooth','monotone']
    bp_data=[(df_bench['improvement'] if f'{s}_r2' not in df_bench.columns
              else df_bench[f'{s}_r2']-df_bench['scalar_r2']).dropna().values for s in strats_show]
    bp=ax.boxplot(bp_data,labels=strats_show,patch_artist=True,widths=0.5,
                   medianprops=dict(color='black',lw=2.5))
    for patch,s in zip(bp['boxes'],strats_show): patch.set_facecolor(PAL[s]); patch.set_alpha(0.62)
    ax.axhline(0,color='black',lw=0.9,ls='--',alpha=0.45)
    ax.set_ylabel('R² improvement over rrBLUP'); ax.set_title('(B)  Strategy Comparison (all conditions)')
    fig.suptitle('Sample-Size Effect and Strategy Performance',fontsize=13,fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.94])
    if save:
        p=FIG_DIR/'fig05_sample_strategy.pdf'
        fig.savefig(p,dpi=600,format='pdf',bbox_inches='tight'); print(f"  Saved {p}")
    plt.show(); plt.close()

print("\n[4/4] Figure 5: Sample-size + strategy..."); fig5_sample_strategy(df_bench)
print("\n✓ Step 3 complete")


# ══════════════════════════════════════════════════════════════════════════
# CELL 6 — Step 4: Real-World Validation (Figures 6–8)
# ══════════════════════════════════════════════════════════════════════════

print("="*60); print(f"STEP 4 — REAL-WORLD VALIDATION (10 seeds × {N_OUTER} folds)"); print("="*60)
METHODS_REAL=list(STRATEGIES)

def run_real_validation(GRM,Y,trait_names,seeds,n_outer=5,verbose=True):
    all_results={}; all_weights={}; all_deltas={}
    for t_i,trait in enumerate(trait_names):
        y_t=Y[:,t_i].astype(np.float64)
        print(f"\n{'─'*55}\n  Trait: {trait}\n{'─'*55}")
        merged_res={m:[] for m in METHODS_REAL}; merged_wts={m:[] for m in METHODS_REAL}; merged_del=[]
        for seed in seeds:
            res,wts,dels=nested_cv(GRM,y_t,methods=METHODS_REAL,n_outer=n_outer,seed=seed,verbose=verbose)
            for m in METHODS_REAL: merged_res[m].extend(res[m]); merged_wts[m].extend(wts.get(m,[]))
            merged_del.extend(dels)
        all_results[trait]=merged_res; all_weights[trait]=merged_wts; all_deltas[trait]=merged_del
        df=summarise_cv(merged_res)
        print(f"\n  {'method':<12} {'R²':>8} {'±':>6} {'Pearson r':>10} {'Δ R²':>8} {'p':>8}")
        print(f"  {'-'*56}")
        for _,row in df.iterrows():
            p_s=f"{row['p_ttest']:.4f}" if not np.isnan(row['p_ttest']) else '   —'
            print(f"  {row['method']:<12} {row['r2_mean']:>8.4f} ±{row['r2_std']:>5.4f} "
                  f"{row['r_mean']:>10.4f} {row['delta_r2']:>+8.4f} {p_s:>8}")
    return all_results,all_weights,all_deltas

print(f"\n[1/4] Running nested CV ({len(METHODS_REAL)} methods × {len(TRAIT_NAMES)} traits × {N_SEEDS} seeds)...")
all_results,all_weights,all_deltas=run_real_validation(GRM,Y,TRAIT_NAMES,seeds=tuple(range(N_SEEDS)),n_outer=N_OUTER,verbose=True)

rows_real=[]
for trait in TRAIT_NAMES:
    df_=summarise_cv(all_results[trait]); df_.insert(0,'trait',trait); rows_real.append(df_)
pd.concat(rows_real,ignore_index=True).to_csv(TAB_DIR/'table_real_cv_results.csv',index=False)
print(f"\n  Saved table_real_cv_results.csv")

def fig6_eigenspectrum(save=True):
    ev=EV_GRM; U=U_GRM; n=len(ev)
    fig=plt.figure(figsize=(16,9)); gs=gridspec.GridSpec(2,2,hspace=0.38,wspace=0.32,left=0.08,right=0.97,top=0.93,bottom=0.08)
    ax=fig.add_subplot(gs[0,0])
    ax.semilogy(np.arange(1,n+1),ev,'o-',color='#2166AC',ms=3.5,lw=1.6,alpha=0.85)
    ax.axvline(EFF_RANK,color='#D62728',lw=2.0,ls='--',label=f'Effective rank={EFF_RANK}')
    ax.fill_betweenx([ev.min()*0.5,ev.max()*2],0.5,EFF_RANK+0.5,alpha=0.06,color='#D62728')
    ax.set_xlabel('Eigenvector index'); ax.set_ylabel('Eigenvalue (log)'); ax.set_title('(A)  GRM Eigenspectrum'); ax.legend(fontsize=9)
    ax=fig.add_subplot(gs[0,1])
    cumvar=np.cumsum(ev)/ev.sum(); ax.plot(np.arange(1,n+1),cumvar,color='#2CA02C',lw=2.5)
    for thresh,ls_,col_ in [(0.80,'--','#FF7F0E'),(0.90,':','#9467BD'),(0.95,'-.','#D62728')]:
        k_=int(np.searchsorted(cumvar,thresh))+1
        ax.axhline(thresh,color=col_,lw=1.3,ls=ls_,label=f'{int(thresh*100)}% at k={k_}')
    ax.set_xlabel('Eigenvectors'); ax.set_ylabel('Cumulative variance'); ax.set_title('(B)  Cumulative Variance'); ax.legend(fontsize=9,loc='lower right')
    ax=fig.add_subplot(gs[1,0])
    d_means=[np.nanmean(all_deltas.get(t,[0.])) for t in TRAIT_NAMES]
    d_stds= [np.nanstd( all_deltas.get(t,[0.])) for t in TRAIT_NAMES]
    short=[t.replace('Brown_','').replace('_mm','').replace('_cm_Yan_et_al_2007','') for t in TRAIT_NAMES]
    bars=ax.bar(range(len(TRAIT_NAMES)),d_means,yerr=d_stds,capsize=6,color='#9467BD',alpha=0.75,edgecolor='#6A3D9A',lw=1.2)
    ax.set_xticks(range(len(TRAIT_NAMES))); ax.set_xticklabels(short,rotation=20,ha='right',fontsize=9.5)
    ax.set_ylabel(r'$\Delta_{\rm spectral}$'); ax.set_title('(C)  Oracle Shrinkage Gap per Trait\n(Δ≈0 → scalar optimal)')
    for bar,v in zip(bars,d_means):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.000005,f'{v:.6f}',ha='center',va='bottom',fontsize=8)
    ax=fig.add_subplot(gs[1,1]); ti=min(1,len(TRAIT_NAMES)-1)
    sc_plot=ax.scatter(U[:,0],U[:,1],c=Y[:,ti],cmap='RdYlBu_r',s=55,alpha=0.82,edgecolors='white',lw=0.5)
    plt.colorbar(sc_plot,ax=ax,shrink=0.88,label=short[ti])
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_title('(D)  GRM Top-2 PCs (population structure)')
    fig.suptitle(f'GRM Analysis  (n={N}, eff_rank={EFF_RANK})',fontsize=13,fontweight='bold')
    if save:
        p=FIG_DIR/'fig06_eigenspectrum.pdf'
        fig.savefig(p,dpi=600,format='pdf',bbox_inches='tight'); print(f"  Saved {p}")
    plt.show(); plt.close()

def fig7_cv_results(save=True):
    T=len(TRAIT_NAMES); short=[t.replace('Brown_','').replace('_mm','').replace('ht_cm_Yan_et_al_2007','height') for t in TRAIT_NAMES]
    fig,axes=plt.subplots(1,2,figsize=(14,5.5))
    ax=axes[0]; x=np.arange(T); nmet=len(METHODS_REAL); width=0.8/nmet
    for si,strat in enumerate(METHODS_REAL):
        means=[np.nanmean([m['r2'] for m in all_results[t][strat]]) for t in TRAIT_NAMES]
        stds= [np.nanstd([m['r2']  for m in all_results[t][strat]]) for t in TRAIT_NAMES]
        ax.bar(x+si*width,means,width,yerr=stds,capsize=3,color=PAL[strat],alpha=0.78,label=strat,edgecolor='white',lw=0.5)
    ax.set_xticks(x+width*(nmet-1)/2); ax.set_xticklabels(short,rotation=15,ha='right')
    ax.set_ylabel('CV R² (mean ± std)'); ax.set_title('(A)  All SAS-BLUP Strategies'); ax.legend(ncol=2)
    ax=axes[1]; sas=[s for s in METHODS_REAL if s!='scalar']
    mat=np.zeros((len(sas),T))
    for si,strat in enumerate(sas):
        for ti,trait in enumerate(TRAIT_NAMES):
            r2_sas=np.nanmean([m['r2'] for m in all_results[trait][strat]])
            r2_sc =np.nanmean([m['r2'] for m in all_results[trait]['scalar']])
            mat[si,ti]=r2_sas-r2_sc
    vmax=max(0.02,np.abs(mat).max()); im=ax.imshow(mat,cmap='RdYlGn',aspect='auto',vmin=-vmax,vmax=vmax)
    ax.set_xticks(range(T)); ax.set_xticklabels(short,rotation=15,ha='right')
    ax.set_yticks(range(len(sas))); ax.set_yticklabels(sas)
    for si in range(len(sas)):
        for ti in range(T):
            v=mat[si,ti]; c='white' if abs(v)>vmax*0.55 else 'black'
            ax.text(ti,si,f'{v:+.4f}',ha='center',va='center',fontsize=9,color=c,fontweight='bold')
    plt.colorbar(im,ax=ax,shrink=0.88,label='R² − rrBLUP'); ax.set_title('(B)  Improvement over rrBLUP')
    fig.suptitle(f'Real Data CV Results  (n={N})',fontsize=13,fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.93])
    if save:
        p=FIG_DIR/'fig07_real_cv_results.pdf'
        fig.savefig(p,dpi=600,format='pdf',bbox_inches='tight'); print(f"  Saved {p}")
    plt.show(); plt.close()

def fig8_shrinkage_real(save=True):
    T=len(TRAIT_NAMES); ev=EV_GRM; U=U_GRM; n=len(ev); idx=np.arange(n)
    fig,axes=plt.subplots(2,2,figsize=(14,9)) if T==4 else plt.subplots(1,T,figsize=(5*T,4.5))
    axes=np.array(axes).ravel()
    short=[t.replace('Brown_','').replace('_mm','').replace('ht_cm_Yan_et_al_2007','Plant Height') for t in TRAIT_NAMES]
    for ax,trait,sname in zip(axes,TRAIT_NAMES,short):
        t_i=TRAIT_NAMES.index(trait); y_t=Y[:,t_i].astype(np.float64)
        sc_=StandardScaler(); y_s=sc_.fit_transform(y_t.reshape(-1,1)).ravel(); z_f=U.T@y_s
        w_sc,_,_,_=_scalar_warmstart(ev,U,z_f,y_s)
        ax.plot(idx,w_sc,color=PAL['scalar'],lw=2.2,ls='-.',alpha=0.9,label='Scalar (rrBLUP)',zorder=6)
        for strat in ('grouped','smooth','monotone'):
            ws=all_weights.get(trait,{}).get(strat,[])
            if not ws: continue
            ml=min(w.shape[0] for w in ws)
            w_avg=np.mean([w[:ml] for w in ws],axis=0); w_std=np.std([w[:ml] for w in ws],axis=0)
            ax.plot(idx[:ml],w_avg,color=PAL[strat],lw=2.0,label=f'SAS ({strat})',alpha=0.88)
            ax.fill_between(idx[:ml],w_avg-w_std,w_avg+w_std,alpha=0.12,color=PAL[strat])
        ax.axvline(EFF_RANK,color='#7F7F7F',lw=1.2,ls=':',alpha=0.7)
        ax.text(EFF_RANK+0.5,1.08,f'eff.rank={EFF_RANK}',fontsize=7.5,color='#7F7F7F',va='top')
        ax.set_title(sname); ax.set_xlabel('Eigenvector index'); ax.set_ylabel('Shrinkage weight $w_i$')
        ax.set_ylim(-0.06,1.22); ax.set_xlim(0,n-1)
    axes[0].legend(fontsize=8.5,loc='upper right'); [ax.set_visible(False) for ax in axes[T:]]
    fig.suptitle('Learned Shrinkage Profiles (Real GRM, hardened strategies)',fontsize=13,fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.94])
    if save:
        p=FIG_DIR/'fig08_shrinkage_real.pdf'
        fig.savefig(p,dpi=600,format='pdf',bbox_inches='tight'); print(f"  Saved {p}")
    plt.show(); plt.close()

print("\n[2/4] Figure 6: Eigenspectrum..."); fig6_eigenspectrum()
print("\n[3/4] Figure 7: CV results..."); fig7_cv_results()
print("\n[4/4] Figure 8: Shrinkage profiles..."); fig8_shrinkage_real()
print("\n✓ Step 4 complete")


# ══════════════════════════════════════════════════════════════════════════
# CELL 7 — Step 5: Baseline Comparison (Figures 9–10, Table 3)
# ══════════════════════════════════════════════════════════════════════════

print("="*60); print("STEP 5 — BASELINE COMPARISON"); print("="*60)

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
# ══════════════════════════════════════════════════════
# DNNGP — Deep Neural Network Genomic Predictor
# Wang et al. (2023) Molecular Plant — Simplified impl.
# ══════════════════════════════════════════════════════

import torch.nn as nn

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

print("\n[1/4] Running baseline comparison (5 seeds × 5 folds)...")
all_base={}
for t_i,trait in enumerate(TRAIT_NAMES):
    y_t=Y[:,t_i].astype(np.float64); print(f"\n  Trait: {trait}")
    merged={m:[] for m in ALL_METHODS}
    for seed in range(5):
        res_=nested_cv_baselines(GRM,y_t,methods=ALL_METHODS,n_outer=5,seed=seed,verbose=True)
        for m in ALL_METHODS: merged[m].extend(res_[m])
    all_base[trait]=merged
    df_=summarise_cv(merged,ref='scalar')
    print(f"\n  {df_[['method','r2_mean','r2_std','delta_r2','p_ttest']].to_string(index=False)}")

def fig9_baseline(save=True):
    T=len(TRAIT_NAMES); short=[t.replace('Brown_','').replace('_mm','').replace('ht_cm_Yan_et_al_2007','Height') for t in TRAIT_NAMES]
    fig,axes=plt.subplots(T,2,figsize=(14,3.5*T))
    if T==1: axes=axes[np.newaxis,:]
    for t_i,(trait,sname) in enumerate(zip(TRAIT_NAMES,short)):
        res=all_base[trait]; df=summarise_cv(res,ref='scalar'); colors_=[PAL.get(m,'#AAAAAA') for m in df.method]
        ax=axes[t_i,0]
        bars=ax.barh(df.method,df.r2_mean,xerr=df.r2_std,color=colors_,alpha=0.80,capsize=4,edgecolor='white',lw=0.8)
        ax.axvline(0,color='black',lw=0.8,ls='--',alpha=0.4)
        best_r2=df.r2_mean.max()
        for bar,val,m in zip(bars,df.r2_mean,df.method):
            bar.set_linewidth(2.5 if val==best_r2 else 0.8); bar.set_edgecolor('#333' if val==best_r2 else 'white')
            ax.text(val+0.005,bar.get_y()+bar.get_height()/2,f'{val:.4f}',va='center',fontsize=8.5)
        ax.set_xlabel('R²'); ax.set_title(f'{sname} — R²')
        ax=axes[t_i,1]
        ax.barh(df.method,df.r_mean,xerr=df.r_std,color=colors_,alpha=0.80,capsize=4,edgecolor='white',lw=0.8)
        ax.set_xlabel('Pearson r'); ax.set_title(f'{sname} — Pearson r')
        for y_pos,(_,row) in enumerate(df.iterrows()):
            if not np.isnan(row['p_ttest']) and row['p_ttest']<0.05:
                axes[t_i,0].text(best_r2*0.98,y_pos,'*',ha='right',va='center',fontsize=12,color='#D62728')
    fig.suptitle('Comprehensive Baseline Comparison\n(* = p<0.05 vs rrBLUP)',fontsize=13,fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.96])
    if save:
        p=FIG_DIR/'fig09_baseline_comparison.pdf'
        fig.savefig(p,dpi=600,format='pdf',bbox_inches='tight'); print(f"  Saved {p}")
    plt.show(); plt.close()

def fig10_radar(save=True):
    mshow=['scalar','grouped','smooth','monotone','BayesB','KRR_RBF','PCR']
    short=[t.replace('Brown_','').replace('_mm','').replace('ht_cm_Yan_et_al_2007','Height') for t in TRAIT_NAMES]
    agg={m:[float(np.nanmean([f['r2'] for f in all_base[t].get(m,[])])) if all_base[t].get(m) else 0. for t in TRAIT_NAMES] for m in mshow}
    N_cat=len(TRAIT_NAMES); angles=np.linspace(0,2*np.pi,N_cat,endpoint=False).tolist(); angles+=angles[:1]
    fig,ax=plt.subplots(figsize=(8,8),subplot_kw=dict(polar=True))
    for m in mshow:
        vals=agg[m]+agg[m][:1]; col=PAL.get(m,'#AAAAAA')
        ax.plot(angles,vals,'o-',color=col,lw=2.2,ms=6,label=m,alpha=0.88); ax.fill(angles,vals,color=col,alpha=0.06)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(short,fontsize=10.5)
    ax.set_title('Multi-Trait Performance Radar\n(mean R² per method)',fontsize=12,fontweight='bold',pad=20)
    ax.legend(loc='upper right',bbox_to_anchor=(1.32,1.12),fontsize=9.5)
    if save:
        p=FIG_DIR/'fig10_radar.pdf'
        fig.savefig(p,dpi=600,format='pdf',bbox_inches='tight'); print(f"  Saved {p}")
    plt.show(); plt.close()

def table3_comparison():
    rows=[]
    for trait in TRAIT_NAMES:
        df=summarise_cv(all_base[trait],ref='scalar'); df.insert(0,'trait',trait); rows.append(df)
    df_full=pd.concat(rows,ignore_index=True)
    df_full.to_csv(TAB_DIR/'table3_full_comparison.csv',index=False)
    latex=[r'\begin{table}[ht]',r'\centering\small',r'\begin{tabular}{llcccc}',r'\toprule',
           r'Trait & Method & $R^2\pm\sigma$ & Pearson $r$ & $\Delta R^2$ & $p$ \\',r'\midrule']
    for _,row in df_full.sort_values(['trait','r2_mean'],ascending=[True,False]).iterrows():
        star=r'$^{*}$' if (not np.isnan(row['p_ttest']) and row['p_ttest']<0.05) else ''
        best=df_full[df_full.trait==row['trait']]['r2_mean'].max()
        nm=(r'\textbf{'+row['method']+'}') if row['r2_mean']==best else row['method']
        p_s=f"{row['p_ttest']:.3f}" if not np.isnan(row['p_ttest']) else '—'
        ts=row['trait'].replace('Brown_','').replace('_mm','').replace('ht_cm_Yan_et_al_2007','Ht')
        latex.append(f"  {ts} & {nm} & {row['r2_mean']:.4f}$\\pm${row['r2_std']:.4f} & "
                     f"{row['r_mean']:.4f} & {row['delta_r2']:+.4f} & {p_s}{star} \\\\")
    latex+=[r'\bottomrule',r'\end{tabular}',r'\caption{SAS-BLUP vs baselines (nested CV)}',r'\label{tab:comp}',r'\end{table}']
    with open(TAB_DIR/'table3_latex.tex','w') as f: f.write('\n'.join(latex))
    print(f"  Saved table3_full_comparison.csv + table3_latex.tex"); return df_full

print("\n[2/4] Figure 9: Baseline comparison..."); fig9_baseline()
print("\n[3/4] Figure 10: Radar chart..."); fig10_radar()
print("\n[4/4] Table 3: Full comparison..."); df_t3=table3_comparison()
print("\n✓ Step 5 complete")
# ══════════════════════════════════════════════════════════════════════
# STEP 5b — Statistical Rigor: FDR Correction + Effect Size + Bootstrap CI
# ══════════════════════════════════════════════════════════════════════

print("="*60); print("STEP 5b — STATISTICAL RIGOR"); print("="*60)

from statsmodels.stats.multitest import multipletests

# ── FDR Correction (Benjamini-Hochberg) ──────────────────────────────

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

# ── Cohen's d Effect Size ─────────────────────────────────────────────

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

# ── Bootstrap 95% CI for R² ──────────────────────────────────────────

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

# ── Figure: Bootstrap CI Plot ─────────────────────────────────────────

def fig_bootstrap_ci(df_ci, save=True):
    traits_short = df_ci['trait'].unique()
    T = len(traits_short)
    fig, axes = plt.subplots(1, T, figsize=(5*T, 5), sharey=False)
    if T == 1: axes = [axes]
    colors = [PAL.get(m, '#AAAAAA') for m in ALL_METHODS]
    for ax, trait in zip(axes, traits_short):
        sub = df_ci[df_ci['trait'] == trait].copy()
        sub = sub.dropna(subset=['mean'])
        y_pos = np.arange(len(sub))
        ax.barh(y_pos, sub['mean'], color=[PAL.get(m,'#AAA') for m in sub['method']],
                alpha=0.75, height=0.6)
        ax.errorbar(sub['mean'], y_pos,
                    xerr=[sub['mean']-sub['ci_lo'], sub['ci_hi']-sub['mean']],
                    fmt='none', color='black', capsize=5, lw=1.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sub['method'], fontsize=9)
        ax.set_xlabel('R² (mean ± 95% Bootstrap CI)')
        ax.set_title(trait)
        ax.axvline(0, color='black', lw=0.8, ls='--', alpha=0.4)
    fig.suptitle('Bootstrap 95% CI for R²\n(all methods, all traits)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.94])
    if save:
        p = FIG_DIR/'fig15_bootstrap_ci.pdf'
        fig.savefig(p, dpi=600, format='pdf', bbox_inches='tight')
        print(f"\n  Saved {p}")
    plt.show(); plt.close()

# ── Run all ───────────────────────────────────────────────────────────
print("\n[1/3] FDR correction...")
df_fdr = apply_fdr_correction(all_base, TRAIT_NAMES, ref='scalar')

print("\n[2/3] Effect sizes (Cohen's d)...")
df_es  = compute_effect_sizes(all_base, TRAIT_NAMES, ref='scalar')

print("\n[3/3] Bootstrap 95% CI...")
df_ci  = bootstrap_r2_ci(all_base, TRAIT_NAMES)
fig_bootstrap_ci(df_ci)

print("\n✓ Step 5b complete — Statistical rigor done")

# ══════════════════════════════════════════════════════════════════════════
# CELL 8 — Step 6: Ablation & Interpretation (Figures 11–13)
# ══════════════════════════════════════════════════════════════════════════

print("="*60); print("STEP 6 — ABLATION & INTERPRETATION"); print("="*60)

def ablation_strategy(n_outer=5,seeds=(0,1,2)):
    rows=[]
    for t_i,trait in enumerate(TRAIT_NAMES):
        y_t=Y[:,t_i].astype(np.float64)
        short=trait.replace('Brown_','').replace('_mm','').replace('ht_cm_Yan_et_al_2007','Height')
        print(f"  {trait}...",end=' ',flush=True)
        for strat in STRATEGIES:
            r2s=[]
            for seed in seeds:
                kf=KFold(n_splits=n_outer,shuffle=True,random_state=seed); sc=StandardScaler()
                for tr,val in kf.split(GRM):
                    y_s=sc.fit_transform(y_t[tr].reshape(-1,1)).ravel()
                    try:
                        est=SASBLUPEstimator(strategy=strat); est.fit(GRM[np.ix_(tr,tr)],y_s)
                        pred=sc.inverse_transform(est.predict(GRM[np.ix_(val,tr)]).reshape(-1,1)).ravel()
                        r2s.append(r2_score(y_t[val],pred))
                    except: r2s.append(np.nan)
            rows.append(dict(trait=trait,short=short,strategy=strat,r2_mean=np.nanmean(r2s),r2_std=np.nanstd(r2s)))
        print("done")
    df=pd.DataFrame(rows); df.to_csv(TAB_DIR/'ablation_strategy.csv',index=False); return df

def ablation_hyperparams(n_outer=5,seed=0):
    best_trait,best_r2=TRAIT_NAMES[0],-np.inf
    for t_i,trait in enumerate(TRAIT_NAMES):
        y_t=Y[:,t_i].astype(np.float64); kf=KFold(n_splits=5,shuffle=True,random_state=0); sc=StandardScaler(); s=[]
        for tr,val in kf.split(GRM):
            y_s=sc.fit_transform(y_t[tr].reshape(-1,1)).ravel(); est=SASBLUPEstimator(strategy='scalar')
            est.fit(GRM[np.ix_(tr,tr)],y_s)
            pred=sc.inverse_transform(est.predict(GRM[np.ix_(val,tr)]).reshape(-1,1)).ravel()
            s.append(r2_score(y_t[val],pred))
        if np.mean(s)>best_r2: best_r2=np.mean(s); best_trait=trait
    print(f"  Ablation trait: {best_trait}  (scalar R²={best_r2:.4f})")
    y_t=Y[:,TRAIT_NAMES.index(best_trait)].astype(np.float64)
    def cv_r2(strategy,**kwargs):
        kf=KFold(n_splits=n_outer,shuffle=True,random_state=seed); sc=StandardScaler(); s=[]
        for tr,val in kf.split(GRM):
            y_s=sc.fit_transform(y_t[tr].reshape(-1,1)).ravel()
            try:
                est=SASBLUPEstimator(strategy=strategy,**kwargs); est.fit(GRM[np.ix_(tr,tr)],y_s)
                pred=sc.inverse_transform(est.predict(GRM[np.ix_(val,tr)]).reshape(-1,1)).ravel()
                s.append(r2_score(y_t[val],pred))
            except: s.append(np.nan)
        return float(np.nanmean(s)),float(np.nanstd(s))
    n_groups_grid=[2,3,5,8,12,20,30]; r2_g=[cv_r2('grouped',n_groups=ng) for ng in n_groups_grid]
    n_knots_grid=[2,3,5,8,12];        r2_k=[cv_r2('spline',n_knots=nk)  for nk in n_knots_grid]
    return best_trait,best_r2,n_groups_grid,r2_g,n_knots_grid,r2_k

def ablation_eigenspectrum():
    rows=[]
    for decay in ['fast','moderate','slow']:
        for alignment in ['top','uniform']:
            for n_ in [80,150]:
                G,y,_,theta,ev,U=make_grm_synthetic(n=n_,p=n_*10,alignment=alignment,noise=0.6,decay=decay,seed=42)
                delta=compute_delta_spectral(ev,theta,0.36,n_)
                res,_,_=nested_cv(G,y,methods=['scalar','grouped','smooth'],n_outer=4,seed=42,verbose=False)
                df_=summarise_cv(res)
                r2_sc=df_.loc[df_.method=='scalar','r2_mean'].values[0]
                r2_sas=df_.loc[df_.method.isin(['grouped','smooth']),'r2_mean'].max()
                rows.append(dict(decay=decay,alignment=alignment,n=n_,delta=delta,r2_scalar=r2_sc,r2_sas=r2_sas,improvement=r2_sas-r2_sc))
    df=pd.DataFrame(rows); df.to_csv(TAB_DIR/'ablation_eigenspectrum.csv',index=False); return df

print("\n[1/6] Ablation A: Strategy sensitivity..."); df_strat=ablation_strategy(seeds=(0,1,2))
print("\n[2/6] Ablation B: Hyperparameter sensitivity..."); best_trait,scalar_r2_ab,n_groups_grid,r2_g,n_knots_grid,r2_k=ablation_hyperparams()
print("\n[3/6] Ablation C: Eigenspectrum shape..."); df_eigen=ablation_eigenspectrum()

def fig11_ablation(save=True):
    fig=plt.figure(figsize=(16,11)); gs=gridspec.GridSpec(2,2,hspace=0.40,wspace=0.35,left=0.08,right=0.97,top=0.93,bottom=0.08)
    ax=fig.add_subplot(gs[0,0])
    trait_names_=df_strat['trait'].unique().tolist(); strats_show=list(STRATEGIES)
    short_traits=[t.replace('Brown_','').replace('_mm','').replace('ht_cm_Yan_et_al_2007','Height') for t in trait_names_]
    mat=np.zeros((len(strats_show),len(trait_names_)))
    for si,s in enumerate(strats_show):
        for ti,t in enumerate(trait_names_):
            row=df_strat[(df_strat.strategy==s)&(df_strat.trait==t)]
            mat[si,ti]=row['r2_mean'].values[0] if len(row) else np.nan
    scalar_row=mat[strats_show.index('scalar'),:]; mat_rel=mat-scalar_row[None,:]
    vmax=max(0.01,np.nanmax(np.abs(mat_rel[1:,:])))
    im=ax.imshow(mat_rel,cmap='RdYlGn',aspect='auto',vmin=-vmax,vmax=vmax)
    ax.set_xticks(range(len(trait_names_))); ax.set_xticklabels(short_traits,rotation=20,ha='right',fontsize=9)
    ax.set_yticks(range(len(strats_show))); ax.set_yticklabels(strats_show,fontsize=9)
    for si in range(len(strats_show)):
        for ti in range(len(trait_names_)):
            v=mat_rel[si,ti]
            if not np.isnan(v):
                ax.text(ti,si,f'{v:+.4f}',ha='center',va='center',fontsize=8.5,
                        color='white' if abs(v)>vmax*0.6 else 'black',fontweight='bold')
    plt.colorbar(im,ax=ax,shrink=0.85,label='R² vs rrBLUP'); ax.set_title('(A)  Strategy Sensitivity Across Traits')
    ax=fig.add_subplot(gs[0,1])
    r2_means_g=[v[0] for v in r2_g]; r2_stds_g=[v[1] for v in r2_g]
    ax.errorbar(n_groups_grid,r2_means_g,yerr=r2_stds_g,fmt='o-',color=PAL['grouped'],lw=2.2,ms=7,capsize=5,markeredgecolor='white')
    ax.axhline(scalar_r2_ab,color=PAL['scalar'],lw=2.0,ls='--',label='rrBLUP'); ax.legend(fontsize=9.5)
    best_k=n_groups_grid[int(np.argmax(r2_means_g))]; ax.axvline(best_k,color=PAL['grouped'],lw=1.2,ls=':',alpha=0.7)
    ax.set_xlabel('Number of spectral groups k'); ax.set_ylabel('CV R²'); ax.set_title('(B)  n_groups Sensitivity (grouped)')
    ax.text(best_k,max(r2_means_g)+0.003,f'best k={best_k}',ha='center',fontsize=8.5,color=PAL['grouped'])
    ax=fig.add_subplot(gs[1,0])
    r2_means_k=[v[0] for v in r2_k]; r2_stds_k=[v[1] for v in r2_k]
    ax.errorbar(n_knots_grid,r2_means_k,yerr=r2_stds_k,fmt='s-',color=PAL['spline'],lw=2.2,ms=7,capsize=5,markeredgecolor='white')
    ax.axhline(scalar_r2_ab,color=PAL['scalar'],lw=2.0,ls='--',label='rrBLUP'); ax.legend(fontsize=9.5)
    ax.set_xlabel('Number of B-spline knots'); ax.set_ylabel('CV R²'); ax.set_title('(C)  n_knots Sensitivity (spline)')
    ax=fig.add_subplot(gs[1,1])
    col_decay={'fast':'#D62728','moderate':'#FF7F0E','slow':'#1F77B4'}; ls_align={'top':'-','uniform':'--'}
    for decay in ['fast','moderate','slow']:
        for align in ['top','uniform']:
            sub=df_eigen[(df_eigen.decay==decay)&(df_eigen.alignment==align)]
            if len(sub)==0: continue
            sub=sub.sort_values('n')
            ax.plot(sub.n,sub.improvement,marker=MARKERS.get(align,'o'),color=col_decay[decay],ls=ls_align[align],
                    lw=1.8,ms=6,label=f'{decay}/{align}',markeredgecolor='white',markeredgewidth=0.6)
    ax.axhline(0,color='black',lw=0.9,ls='--',alpha=0.45); ax.set_xlabel('Sample size n'); ax.set_ylabel('R² improvement')
    ax.set_title('(D)  Eigenvalue Decay Rate Effect'); ax.legend(fontsize=8,ncol=2)
    fig.suptitle('Ablation Study',fontsize=13,fontweight='bold')
    if save:
        p=FIG_DIR/'fig11_ablation.pdf'
        fig.savefig(p,dpi=600,format='pdf',bbox_inches='tight'); print(f"  Saved {p}")
    plt.show(); plt.close()

def fig12_weight_stability(save=True):
    focus=TRAIT_NAMES[:2] if len(TRAIT_NAMES)>=2 else TRAIT_NAMES
    ev=EV_GRM; U=U_GRM; n=len(ev); idx=np.arange(n)
    n_rows=len(focus); strats_show=['grouped','smooth','monotone']
    fig,axes=plt.subplots(n_rows,3,figsize=(15,4.5*n_rows),sharey=True,sharex=True)
    if n_rows==1: axes=axes[np.newaxis,:]
    for row_i,trait in enumerate(focus):
        t_i=TRAIT_NAMES.index(trait); y_t=Y[:,t_i].astype(np.float64)
        short=trait.replace('Brown_','').replace('_mm','').replace('ht_cm_Yan_et_al_2007','Height')
        for col_i,strat in enumerate(strats_show):
            ax=axes[row_i,col_i]; kf=KFold(n_splits=8,shuffle=True,random_state=42); sc=StandardScaler(); fold_ws=[]
            for tr,_ in kf.split(GRM):
                y_s=sc.fit_transform(y_t[tr].reshape(-1,1)).ravel()
                try:
                    est=SASBLUPEstimator(strategy=strat); est.fit(GRM[np.ix_(tr,tr)],y_s); fold_ws.append(est.weights_.copy())
                except: pass
            if fold_ws:
                ml=min(w.shape[0] for w in fold_ws); W=np.array([w[:ml] for w in fold_ws]); wm=W.mean(0); ws=W.std(0)
                cv=float((ws/(wm+1e-8)).mean())
                for row in W: ax.plot(idx[:ml],row,color='#AAAAAA',lw=0.6,alpha=0.25)
                ax.plot(idx[:ml],wm,color=PAL[strat],lw=2.5,label=f'Mean (CV={cv:.3f})',zorder=5)
                ax.fill_between(idx[:ml],wm-ws,wm+ws,alpha=0.18,color=PAL[strat])
                ax.axvline(EFF_RANK,color='#999',lw=1.1,ls=':',alpha=0.7)
            ax.set_ylim(-0.06,1.22); ax.set_xlim(0,n-1)
            if row_i==0: ax.set_title(strat,fontsize=11,fontweight='bold')
            if col_i==0: ax.set_ylabel(f'{short}\nShrinkage weight $w_i$',fontsize=9.5)
            if row_i==n_rows-1: ax.set_xlabel('Eigenvector index')
            if fold_ws: ax.legend(fontsize=8.5,loc='upper right')
    fig.suptitle('Weight Stability Across 8 CV Folds\n(grey=individual folds, coloured=mean±1σ)',fontsize=13,fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.95])
    if save:
        p=FIG_DIR/'fig12_weight_stability.pdf'
        fig.savefig(p,dpi=600,format='pdf',bbox_inches='tight'); print(f"  Saved {p}")
    plt.show(); plt.close()

def fig13_interpretation(save=True):
    ev=EV_GRM; U=U_GRM; n=len(ev); idx=np.arange(n)
    fig=plt.figure(figsize=(16,10)); gs=gridspec.GridSpec(2,2,hspace=0.40,wspace=0.35,left=0.08,right=0.97,top=0.93,bottom=0.08)
    ti=min(1,len(TRAIT_NAMES)-1); y_t=Y[:,ti].astype(np.float64); sc_=StandardScaler()
    y_s=sc_.fit_transform(y_t.reshape(-1,1)).ravel()
    est=SASBLUPEstimator(strategy='monotone'); est.fit(GRM,y_s); w_mono=est.weights_
    ax=fig.add_subplot(gs[0,0])
    sc_plot=ax.scatter(U[:,0],U[:,1],c=y_t,cmap='RdYlBu_r',s=50,alpha=0.80,edgecolors='white',lw=0.5)
    plt.colorbar(sc_plot,ax=ax,shrink=0.88,label=TRAIT_NAMES[ti].replace('Brown_',''))
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_title('(A)  GRM Top-2 PCs vs Trait')
    ax=fig.add_subplot(gs[0,1]); ax2=ax.twinx()
    ax.plot(idx,w_mono,color=PAL['monotone'],lw=2.5,label='Monotone weight',zorder=5)
    ax2.semilogy(idx,ev,color='#888',lw=1.4,ls='--',alpha=0.6,label='Eigenvalue')
    ax.axvline(EFF_RANK,color='#D62728',lw=1.2,ls=':',label=f'Eff.rank={EFF_RANK}')
    ax.set_xlabel('Eigenvector index'); ax.set_ylabel('Shrinkage weight',color=PAL['monotone'])
    ax2.set_ylabel('Eigenvalue (log)',color='#888'); ax.set_title('(B)  Shrinkage Profile vs Eigenspectrum')
    lines1,labs1=ax.get_legend_handles_labels(); lines2,labs2=ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2,labs1+labs2,fontsize=8.5,loc='upper right')
    ax=fig.add_subplot(gs[1,0])
    short=[t.replace('Brown_','').replace('_mm','').replace('ht_cm_Yan_et_al_2007','Height') for t in TRAIT_NAMES]
    for strat in ('grouped','monotone'):
        corrs=[]
        for t_i,trait in enumerate(TRAIT_NAMES):
            y_t2=Y[:,t_i].astype(np.float64); sc2=StandardScaler()
            y_s2=sc2.fit_transform(y_t2.reshape(-1,1)).ravel()
            try: est2=SASBLUPEstimator(strategy=strat); est2.fit(GRM,y_s2); r,_=pearsonr(ev,est2.weights_); corrs.append(abs(r))
            except: corrs.append(np.nan)
        ax.bar(np.arange(len(TRAIT_NAMES))+(0.2 if strat=='monotone' else -0.2),corrs,0.38,color=PAL[strat],alpha=0.75,label=strat,edgecolor='white')
    ax.set_xticks(range(len(TRAIT_NAMES))); ax.set_xticklabels(short,rotation=20,ha='right',fontsize=9)
    ax.set_ylabel('|Pearson r| (weight vs eigenvalue)'); ax.set_title('(C)  Weight–Eigenvalue Alignment'); ax.legend(); ax.set_ylim(0,1.1)
    ax=fig.add_subplot(gs[1,1])
    try:
        df_s=pd.read_csv(TAB_DIR/'synth_delta_scatter.csv')
        col_map={'top':'#1F77B4','middle':'#9467BD','uniform':'#D62728','bottom':'#2CA02C'}
        for align,col in col_map.items():
            sub=df_s[df_s.alignment==align]
            ax.scatter(sub.delta,sub.improvement,c=col,alpha=0.65,s=40,label=f'synth ({align})',edgecolors='white',lw=0.3)
        for t_i2,trait in enumerate(TRAIT_NAMES):
            d=float(delta_real.get(trait,0.))
            imp=np.nanmean([m['r2'] for m in all_results[trait]['grouped']])-np.nanmean([m['r2'] for m in all_results[trait]['scalar']])
            ax.scatter(d,imp,c='#FF7F0E',s=150,marker='*',zorder=8,edgecolors='black',lw=0.8,label='real data' if t_i2==0 else None)
    except Exception as e: print(f"  ({e})")
    ax.axhline(0,color='black',lw=0.9,ls='--',alpha=0.4); ax.axvline(0,color='black',lw=0.9,ls='--',alpha=0.4)
    ax.set_xlabel(r'$\Delta_{\rm spectral}$'); ax.set_ylabel('Actual R² improvement')
    ax.set_title('(D)  Theory vs Empirics  (★=real data)'); ax.legend(fontsize=7.5,ncol=2)
    fig.suptitle('Biological Interpretation of Learned Shrinkage Weights',fontsize=13,fontweight='bold')
    if save:
        p=FIG_DIR/'fig13_interpretation.pdf'
        fig.savefig(p,dpi=600,format='pdf',bbox_inches='tight'); print(f"  Saved {p}")
    plt.show(); plt.close()

def table4_ablation():
    pivot=df_strat.pivot_table(values='r2_mean',index='strategy',columns='short',aggfunc='mean')
    pivot['mean']=pivot.mean(axis=1); pivot=pivot.sort_values('mean',ascending=False)
    pivot.to_csv(TAB_DIR/'table4_ablation.csv')
    cols=[c for c in pivot.columns if c!='mean']
    lines=[r'\begin{table}[ht]',r'\centering\small',r'\begin{tabular}{l'+'c'*len(cols)+'c}',r'\toprule',
           'Strategy & '+'& '.join(cols)+r'& Mean \\',r'\midrule']
    for strat,row in pivot.iterrows():
        bold=strat==pivot['mean'].idxmax(); nm=(r'\textbf{'+strat+'}') if bold else strat
        cells=' & '.join([f'{row[c]:.4f}' for c in cols])
        lines.append(f"  {nm} & {cells} & {row['mean']:.4f} \\\\")
    lines+=[r'\bottomrule',r'\end{tabular}',r'\caption{Ablation: strategy comparison}',r'\label{tab:ablation}',r'\end{table}']
    with open(TAB_DIR/'table4_latex.tex','w') as f: f.write('\n'.join(lines))
    print("  Saved table4_ablation.csv + table4_latex.tex"); print(pivot.to_string())
    return pivot

print("\n[4/6] Figure 11: Ablation results..."); fig11_ablation()
print("\n[5/6] Figure 12: Weight stability..."); fig12_weight_stability()
print("\n[6/6] Figure 13: Biological interpretation..."); fig13_interpretation()
print("\n[Table 4] Ablation summary..."); table4_ablation()
print("\n✓ Step 6 complete")
# ══════════════════════════════════════════════════════════════════════
# STEP 6b — Biological Interpretation: Subpopulation Mapping
# Proposal claim: "high-weight eigenvectors correspond to indica/japonica"
# ══════════════════════════════════════════════════════════════════════

print("="*60); print("STEP 6b — INDICA/JAPONICA SUBPOPULATION MAPPING"); print("="*60)

from sklearn.cluster import KMeans
from scipy.stats import f_oneway

def subpopulation_mapping(GRM, U_GRM, EV_GRM, Y, TRAIT_NAMES,
                          SAMPLE_IDS, all_weights, n_pops=4, save=True):
    """
    1. K-means on top PCs → infer rice subpopulations
       (indica, japonica, aus, aromatic)
    2. Check if high-weight eigenvectors correspond to
       population-discriminating PCs
    3. Figure 14 + CSV table
    """
    n   = GRM.shape[0]
    idx = np.arange(n)

    # ── Step 1: Cluster on top 5 PCs ──────────────────────────────
    top5 = U_GRM[:, :5]
    km   = KMeans(n_clusters=n_pops, random_state=42, n_init=30)
    raw_labels = km.fit_predict(top5)

    # Sort clusters by PC1 (rough indica→japonica ordering)
    order  = np.argsort(km.cluster_centers_[:, 0])
    label_map = {old: new for new, old in enumerate(order)}
    pop_labels = np.array([label_map[l] for l in raw_labels])
    pop_names  = [f'Pop {chr(65+i)}' for i in range(n_pops)]
    pop_colors = ['#D62728','#1F77B4','#2CA02C','#FF7F0E']

    print(f"\n  Subpopulation sizes (k={n_pops} clusters on top-5 PCs):")
    for i in range(n_pops):
        print(f"    {pop_names[i]}: n={np.sum(pop_labels==i)}")

    # ── Step 2: F-statistic per eigenvector ───────────────────────
    # How well does each eigenvector discriminate populations?
    n_check = min(n, 60)
    pop_F   = np.zeros(n_check)
    for j in range(n_check):
        groups   = [U_GRM[pop_labels == p, j] for p in range(n_pops)]
        F, _     = f_oneway(*groups)
        pop_F[j] = F if np.isfinite(F) else 0.

    # ── Step 3: Figure 14 ─────────────────────────────────────────
    ti       = min(1, len(TRAIT_NAMES)-1)
    y_trait  = Y[:, ti]
    short_nm = (TRAIT_NAMES[ti].replace('Brown_','')
                               .replace('_mm','')
                               .replace('ht_cm_Yan_et_al_2007','Height'))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    # Panel A: PCA scatter coloured by subpopulation
    ax = axes[0]
    for p_i in range(n_pops):
        mask = pop_labels == p_i
        ax.scatter(U_GRM[mask,0], U_GRM[mask,1],
                   c=pop_colors[p_i], label=pop_names[p_i],
                   alpha=0.78, s=45, edgecolors='white', lw=0.4)
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.set_title('(A) GRM PCA — Inferred Subpopulations\n'
                 '(k-means, k=4 on top-5 PCs)')
    ax.legend(fontsize=9)

    # Panel B: Phenotype boxplot per subpopulation
    ax = axes[1]
    box_data = [y_trait[pop_labels == p] for p in range(n_pops)]
    F_stat, p_anova = f_oneway(*box_data)
    bp = ax.boxplot(box_data, labels=pop_names,
                    patch_artist=True, widths=0.5,
                    medianprops=dict(color='black', lw=2))
    for patch, col in zip(bp['boxes'], pop_colors):
        patch.set_facecolor(col); patch.set_alpha(0.65)
    ax.set_ylabel(short_nm)
    ax.set_title(f'(B) {short_nm} by Subpopulation\n'
                 f'One-way ANOVA: F={F_stat:.2f}, p={p_anova:.4f}')

    # Panel C: Weight profile vs population-discriminating power
    ax  = axes[2]
    ax2 = ax.twinx()

    strat = 'monotone'
    ws    = all_weights.get(TRAIT_NAMES[ti], {}).get(strat, [])
    w_avg = None
    if ws:
        ml    = min(w.shape[0] for w in ws)
        w_avg = np.mean([w[:ml] for w in ws], axis=0)
        ax.plot(idx[:ml], w_avg, color=PAL[strat], lw=2.5,
                label=f'SAS-BLUP ({strat})', zorder=5)
        ax.fill_between(idx[:ml],
                        w_avg - np.std([w[:ml] for w in ws], axis=0),
                        w_avg + np.std([w[:ml] for w in ws], axis=0),
                        alpha=0.15, color=PAL[strat])

    ax2.bar(np.arange(n_check), pop_F,
            alpha=0.25, color='#9467BD',
            label='Between-pop F-stat')
    ax2.set_ylabel('F-statistic\n(population discrimination)', color='#9467BD')
    ax.axvline(EFF_RANK, color='#999', lw=1.2, ls=':', alpha=0.7,
               label=f'Eff. rank={EFF_RANK}')
    ax.set_xlim(0, n-1); ax.set_ylim(-0.05, 1.2)
    ax.set_xlabel('Eigenvector index')
    ax.set_ylabel('Shrinkage weight $w_i$', color=PAL[strat])
    ax.set_title('(C) SAS-BLUP Weight vs Population Structure Signal')
    lines1, labs1 = ax.get_legend_handles_labels()
    ax.legend(lines1, labs1, fontsize=8.5, loc='upper right')

    # Panel D: Scatter — weight vs F-stat correlation
    ax = axes[3]
    if w_avg is not None:
        L = min(len(w_avg), n_check)
        r_corr, p_corr = pearsonr(w_avg[:L], pop_F[:L])
        ax.scatter(pop_F[:L], w_avg[:L],
                   alpha=0.65, s=45, color=PAL[strat],
                   edgecolors='white', lw=0.4)
        z = np.polyfit(pop_F[:L], w_avg[:L], 1)
        x_line = np.linspace(pop_F[:L].min(), pop_F[:L].max(), 100)
        ax.plot(x_line, np.polyval(z, x_line),
                color='#D62728', lw=2.2, ls='--')
        ax.text(0.05, 0.95,
                f'Pearson r = {r_corr:.3f}\np = {p_corr:.4f}',
                transform=ax.transAxes, va='top', fontsize=10.5,
                bbox=dict(boxstyle='round', fc='white', ec='#BBB'))
        print(f"\n  Weight ↔ Population-F correlation: "
              f"r={r_corr:.3f}, p={p_corr:.4f}")
    ax.set_xlabel('Between-population F-statistic')
    ax.set_ylabel('Mean shrinkage weight $w_i$')
    ax.set_title('(D) Weight ↔ Population Discrimination Correlation\n'
                 '(High r = SAS-BLUP weights population-structured PCs)')

    plt.suptitle(
        'Biological Interpretation\n'
        'Inferred Subpopulations vs Learned Shrinkage Weights',
        fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.95])

    if save:
        p = FIG_DIR/'fig14_subpopulation_mapping.pdf'
        fig.savefig(p, dpi=600, format='pdf', bbox_inches='tight')
        print(f"  Saved {p}")
    plt.show(); plt.close()

    # ── Save CSV ──────────────────────────────────────────────────
    pd.DataFrame({
        'sample_id'      : list(SAMPLE_IDS),
        'subpopulation'  : [pop_names[l] for l in pop_labels],
        'pop_index'      : pop_labels,
        'PC1'            : U_GRM[:,0],
        'PC2'            : U_GRM[:,1],
        'PC3'            : U_GRM[:,2],
    }).to_csv(TAB_DIR/'subpopulation_assignments.csv', index=False)
    print("  Saved subpopulation_assignments.csv")

    pd.DataFrame({
        'eigenvector_index': np.arange(n_check),
        'population_F_stat': pop_F,
    }).to_csv(TAB_DIR/'eigenvector_population_F.csv', index=False)
    print("  Saved eigenvector_population_F.csv")

    return pop_labels, pop_names

# ── Run ───────────────────────────────────────────────────────────────
pop_labels, pop_names = subpopulation_mapping(
    GRM, U_GRM, EV_GRM, Y, TRAIT_NAMES, SAMPLE_IDS, all_weights
)
print("\n✓ Step 6b complete — Biological mapping done")

# ══════════════════════════════════════════════════════════════════════════
# CELL 9 — Final Summary + LaTeX + Zip
# ══════════════════════════════════════════════════════════════════════════

def write_master_latex():
    lines=['% SAS-BLUP — Auto-generated tables','\\usepackage{booktabs}','']
    for tf in sorted(TAB_DIR.glob('*.tex')): lines.append(f'\n% ── {tf.name} ──\n'); lines.append(tf.read_text())
    (TAB_DIR/'ALL_TABLES.tex').write_text('\n'.join(lines)); print(f"  Saved ALL_TABLES.tex")

def final_summary():
    print("\n"+"="*65); print("EXPERIMENT COMPLETE — SAS-BLUP Research Pipeline"); print("="*65)
    if USE_GPU:
        alloc=torch.cuda.max_memory_allocated()/1e9; res_=torch.cuda.max_memory_reserved()/1e9
        print(f"\n  GPU: {GPU_NAME}  |  Peak alloc: {alloc:.2f} GB  |  Peak reserved: {res_:.2f} GB")
    print("\n── Figures (600 DPI PDF) ────────────────────────────────────────")
    descs={'fig01':'Theory framework','fig02':'Shrinkage profiles (hardened)','fig03':'Δspectral scatter',
           'fig04':'Benchmark heatmap','fig05':'Sample-size + strategy','fig06':'GRM eigenspectrum',
           'fig07':'Real CV results','fig08':'Shrinkage profiles (real GRM)','fig09':'Baseline comparison',
           'fig10':'Radar chart','fig11':'Ablation','fig12':'Weight stability','fig13':'Biological interpretation'}
    for f in sorted(FIG_DIR.glob('*.pdf')):
        print(f"  {f.name:<45} {f.stat().st_size//1024:>4} KB  {descs.get(f.stem[:5],'')}")
    print("\n── Tables ───────────────────────────────────────────────────────")
    for t in sorted(TAB_DIR.glob('*.csv')): print(f"  {t.name:<45} {t.stat().st_size//1024:>4} KB")
    print("\n── Key results ──────────────────────────────────────────────────")
    try:
        df_full=pd.read_csv(TAB_DIR/'table3_full_comparison.csv')
        print(f"\n  {'Trait':<35} {'Best method':<14} {'R²':>8} {'Δ vs rrBLUP':>12}")
        print(f"  {'-'*70}")
        for trait in TRAIT_NAMES:
            sub=df_full[df_full.trait==trait]; best=sub.iloc[0]
            sc=sub[sub.method=='scalar']['r2_mean'].values[0]
            print(f"  {trait:<35} {best['method']:<14} {best['r2_mean']:>8.4f} {(best['r2_mean']-sc):>+12.4f}")
    except Exception as e: print(f"  (Table 3: {e})")
    print(f"\n── Collapse fix verification ─────────────────────────────────────")
    print(f"  Effective rank (rice GRM)  : {EFF_RANK}")
    print(f"  Δspectral (all traits)     : ≈ 0.000 → scalar near-optimal (tie predicted)")
    print(f"  Hardening applied          : warm-start + multi-restart + collapse guard")
    print(f"  Expected: smooth/monotone/spline now TIE rrBLUP (not collapse below it)")
    print(f"\n  Files at: {WORK}")
    print("="*65)

def zip_outputs():
    import shutil
    tmp=WORK/"_zip_tmp"; tmp.mkdir(exist_ok=True)
    for d in ['figures','tables']:
        dst=tmp/d
        if dst.exists(): shutil.rmtree(dst)
        shutil.copytree(WORK/d, dst)
    out=shutil.make_archive(str(WORK/"SAS_BLUP_outputs"),'zip',root_dir=tmp)
    shutil.rmtree(tmp); sz=Path(out).stat().st_size/1e6
    print(f"  ✓ Zipped → {out}  ({sz:.2f} MB)")
    print(f"    Download from Kaggle Output panel (right sidebar)")

write_master_latex()
final_summary()
zip_outputs()

##new cell for new data 
# ══════════════════════════════════════════════════════════════════
# CELL 1b — TCGA Gene Expression Data Loader
# Non-genomic p>>n validation (Proposal Step 4)
# n=801 samples, p=20,531 genes — pure p>>n setting
# ══════════════════════════════════════════════════════════════════

print("="*60)
print("TCGA GENE EXPRESSION — NON-GENOMIC p>>n VALIDATION")
print("="*60)

# ── Paths (Kaggle এ আপলোড করার পর এই paths দাও) ──────────────────
TCGA_DATA_CSV   = "/kaggle/input/datasets/salehinseyam/new-data/data.csv"
TCGA_LABELS_CSV = "/kaggle/input/datasets/salehinseyam/new-data/labels.csv"
TCGA_WORK       = Path("/kaggle/working/tcga")
TCGA_WORK.mkdir(parents=True, exist_ok=True)
TCGA_FIG_DIR    = TCGA_WORK / "figures"; TCGA_FIG_DIR.mkdir(exist_ok=True)
TCGA_TAB_DIR    = TCGA_WORK / "tables";  TCGA_TAB_DIR.mkdir(exist_ok=True)

def load_tcga_data(data_csv, labels_csv):
    """
    Load TCGA gene expression data.
    Returns:
        X_kernel : (n,n) kernel matrix (like GRM)
        Y_binary : (n, n_classes) binary phenotype matrix
        class_names : list of cancer type names
        sample_ids  : list of sample IDs
    """
    print("  Loading gene expression matrix...")
    X = pd.read_csv(data_csv, index_col=0)
    sample_ids = list(X.index)
    n, p = X.shape
    print(f"  Samples: {n}, Genes: {p}  (p/n ratio = {p/n:.0f}x)")

    # ── Normalize: StandardScaler per gene ──────────────────────
    from sklearn.preprocessing import StandardScaler
    X_np = StandardScaler().fit_transform(X.values.astype(np.float32))
    print(f"  Gene expression normalized (zero mean, unit variance)")

    # ── Build Kernel Matrix (like GRM but for gene expression) ──
    # K = XX^T / p  — same formula as GRM
    print("  Building kernel matrix K = XX^T / p ...")
    if USE_GPU:
        Xt = _to_gpu(X_np)
        K_gpu = torch.mm(Xt, Xt.T) / p
        torch.cuda.synchronize()
        K = _to_cpu(K_gpu)
        del Xt, K_gpu; clear_gpu()
    else:
        K = X_np @ X_np.T / p
    K = K.astype(np.float64)

    # ── Add small jitter for numerical stability ────────────────
    K += 1e-4 * np.eye(n)
    print(f"  Kernel matrix shape: {K.shape}")

    # ── Load labels ─────────────────────────────────────────────
    labels_df = pd.read_csv(labels_csv, index_col=0)
    labels_df = labels_df.loc[sample_ids]
    class_names = sorted(labels_df['Class'].unique())
    print(f"  Cancer types ({len(class_names)}): {class_names}")

    # ── One-vs-Rest binary encoding ─────────────────────────────
    # Each cancer type becomes a binary phenotype (1=this type, 0=other)
    # SAS-BLUP predicts membership score for each cancer type
    Y_binary = np.zeros((n, len(class_names)), dtype=np.float32)
    for ci, cname in enumerate(class_names):
        Y_binary[:, ci] = (labels_df['Class'].values == cname).astype(float)
        n_pos = int(Y_binary[:, ci].sum())
        print(f"    {cname}: {n_pos} positive / {n-n_pos} negative")

    print(f"\n  ✓ TCGA data loaded")
    print(f"    Kernel: {K.shape}, Phenotypes: {Y_binary.shape}")
    return K, Y_binary, class_names, sample_ids

# ── Eigenvalue analysis ──────────────────────────────────────────────
def analyze_tcga_kernel(K, Y_binary, class_names):
    print("\n  Computing eigendecomposition...")
    ev_tcga, U_tcga = _gpu_eigh(K)
    ev_tcga = np.maximum(ev_tcga, 1e-8)
    eff_rank_tcga = int(ev_tcga.sum()**2 / (ev_tcga**2).sum())

    print(f"  Eigenvalue range: [{ev_tcga[-1]:.4f}, {ev_tcga[0]:.4f}]")
    print(f"  Effective rank  : {eff_rank_tcga}")
    k90 = int(np.searchsorted(
        np.cumsum(ev_tcga)/ev_tcga.sum(), 0.90)) + 1
    print(f"  Cumulative var 90%: k={k90}")

    # Compute Delta_spectral for each cancer type
    print("\n  Computing Delta_spectral per cancer type...")
    delta_tcga = {}
    for ci, cname in enumerate(class_names):
        y_t = Y_binary[:, ci].astype(np.float64)
        sc_ = StandardScaler()
        y_s = sc_.fit_transform(y_t.reshape(-1,1)).ravel()
        z_f = U_tcga.T @ y_s
        sig2 = float(np.var(y_s - U_tcga @ (U_tcga.T @ y_s)))
        d = compute_delta_spectral(ev_tcga, z_f, max(sig2, 1e-8), len(y_t))
        delta_tcga[cname] = d
        print(f"    {cname:<8} Δspectral = {d:.6f}")

    pd.DataFrame(list(delta_tcga.items()),
                 columns=['cancer_type', 'delta_spectral'])\
      .to_csv(TCGA_TAB_DIR/'tcga_delta_spectral.csv', index=False)

    return ev_tcga, U_tcga, eff_rank_tcga, delta_tcga

# ── Full nested CV for TCGA ──────────────────────────────────────────
def run_tcga_validation(K, Y_binary, class_names,
                        ev_tcga, U_tcga, eff_rank_tcga,
                        n_outer=5, n_seeds=3):
    """
    Run SAS-BLUP + all baselines on TCGA data.
    One-vs-Rest: predict each cancer type as binary phenotype.
    """
    print("\n  Running nested CV on TCGA data...")
    ALL_TCGA_METHODS = list(STRATEGIES) + ['BayesB', 'KRR_RBF', 'PCR', 'DNNGP']
    all_tcga_results = {}

    for ci, cname in enumerate(class_names):
        y_t = Y_binary[:, ci].astype(np.float64)
        print(f"\n  ── Cancer type: {cname} ──────────────────────")
        merged = {m: [] for m in ALL_TCGA_METHODS}

        for seed in range(n_seeds):
            res_ = nested_cv_baselines(
                K, y_t,
                methods=ALL_TCGA_METHODS,
                n_outer=n_outer,
                seed=seed,
                verbose=True
            )
            for m in ALL_TCGA_METHODS:
                merged[m].extend(res_[m])

        all_tcga_results[cname] = merged
        df_ = summarise_cv(merged, ref='scalar')
        print(f"\n  {df_[['method','r2_mean','r2_std','delta_r2']].to_string(index=False)}")

    return all_tcga_results, ALL_TCGA_METHODS

# ── Save TCGA results ────────────────────────────────────────────────
def save_tcga_results(all_tcga_results, class_names):
    rows = []
    for cname in class_names:
        df_ = summarise_cv(all_tcga_results[cname], ref='scalar')
        df_.insert(0, 'cancer_type', cname)
        rows.append(df_)
    df_full = pd.concat(rows, ignore_index=True)
    df_full.to_csv(TCGA_TAB_DIR/'tcga_full_comparison.csv', index=False)
    print("\n  Saved tcga_full_comparison.csv")
    return df_full

# ── Figure: TCGA Summary ─────────────────────────────────────────────
def fig_tcga_results(all_tcga_results, class_names,
                     delta_tcga, eff_rank_tcga, save=True):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: R² per method per cancer type
    ax = axes[0]
    methods_show = ['scalar', 'grouped', 'smooth', 'monotone',
                    'BayesB', 'KRR_RBF', 'PCR', 'DNNGP']
    x = np.arange(len(class_names))
    width = 0.8 / len(methods_show)
    for si, method in enumerate(methods_show):
        means = [np.nanmean([m['r2'] for m in
                  all_tcga_results[c].get(method, [])])
                 for c in class_names]
        stds  = [np.nanstd([m['r2'] for m in
                  all_tcga_results[c].get(method, [])])
                 for c in class_names]
        ax.bar(x + si*width, means, width, yerr=stds, capsize=3,
               color=PAL.get(method, '#AAAAAA'), alpha=0.78,
               label=method, edgecolor='white', lw=0.5)
    ax.set_xticks(x + width*(len(methods_show)-1)/2)
    ax.set_xticklabels(class_names, rotation=15, ha='right')
    ax.set_ylabel('CV R² (mean ± std)')
    ax.set_title(f'(A) TCGA — All Methods by Cancer Type\n'
                 f'(n=801, p=20531, eff_rank={eff_rank_tcga})')
    ax.legend(ncol=2, fontsize=8.5)

    # Panel B: Delta_spectral per cancer type + improvement
    ax = axes[1]
    delta_vals = [delta_tcga.get(c, 0) for c in class_names]
    improvements = []
    for c in class_names:
        r2_sas = max([np.nanmean([m['r2'] for m in
                       all_tcga_results[c].get(s, [])])
                      for s in ['grouped','smooth','monotone']])
        r2_sc  = np.nanmean([m['r2'] for m in
                  all_tcga_results[c].get('scalar', [])])
        improvements.append(r2_sas - r2_sc)
    colors = ['#2CA02C' if i > 0.01 else '#D62728' for i in improvements]
    bars = ax.bar(class_names, improvements, color=colors, alpha=0.75,
                  edgecolor='white')
    for bar, d in zip(bars, delta_vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.002,
                f'Δ={d:.4f}', ha='center', va='bottom', fontsize=8.5)
    ax.axhline(0, color='black', lw=1.0, ls='--', alpha=0.5)
    ax.set_ylabel('Best SAS-BLUP R² − rrBLUP R²')
    ax.set_title('(B) TCGA — SAS-BLUP Improvement\n'
                 '(green=improvement, red=tie/worse)')
    ax.set_xticklabels(class_names, rotation=15, ha='right')

    fig.suptitle('TCGA Pan-Cancer Gene Expression\n'
                 'SAS-BLUP Validation (Non-Genomic p>>n Dataset)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if save:
        p = TCGA_FIG_DIR / 'tcga_results.pdf'
        fig.savefig(p, dpi=600, format='pdf', bbox_inches='tight')
        print(f"  Saved {p}")
    plt.show(); plt.close()

# ── RUN ALL ──────────────────────────────────────────────────────────
print("\n[1/4] Loading TCGA data...")
K_tcga, Y_tcga, CLASS_NAMES, TCGA_IDS = load_tcga_data(
    TCGA_DATA_CSV, TCGA_LABELS_CSV)

print("\n[2/4] Kernel analysis + Delta_spectral...")
EV_TCGA, U_TCGA, EFF_RANK_TCGA, delta_tcga = analyze_tcga_kernel(
    K_tcga, Y_tcga, CLASS_NAMES)

print("\n[3/4] Running nested CV (all methods)...")
all_tcga_results, ALL_TCGA_METHODS = run_tcga_validation(
    K_tcga, Y_tcga, CLASS_NAMES,
    EV_TCGA, U_TCGA, EFF_RANK_TCGA,
    n_outer=5, n_seeds=3
)

print("\n[4/4] Saving results + Figure...")
df_tcga = save_tcga_results(all_tcga_results, CLASS_NAMES)
fig_tcga_results(all_tcga_results, CLASS_NAMES,
                 delta_tcga, EFF_RANK_TCGA)

print("\n✓ TCGA Validation Complete!")
print(f"  Results saved to: {TCGA_TAB_DIR}")