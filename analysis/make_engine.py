"""Extract the SAS-BLUP engine and baselines from the authors' notebook (definitions only, unchanged)
and add the two analytical-gradient helpers that the saved notebook calls but does not define."""
import ast, io, sys

SRC = sys.argv[1]; OUT = sys.argv[2]
src = io.open(SRC, encoding="utf-8").read()
tree = ast.parse(src)
KEEP = {"_to_gpu", "_to_cpu", "_gpu_matmul", "_gpu_eigh", "clear_gpu", "safe_eigh", "scalar_shrinkage",
        "oracle_shrinkage", "_sig", "_logit", "loo_mse", "_clear_loo_cache", "_scalar_warmstart", "_multi_restart",
        "_multi_restart_jac", "_collapse_guard", "_learn_scalar", "_learn_grouped", "_learn_smooth",
        "_learn_monotone", "_learn_spline", "SASBLUPEstimator", "compute_metrics", "summarise_cv",
        "BayesBPredictor", "KRR_RBF_Predictor", "PCRPredictor", "DNNGPPredictor", "nested_cv_baselines",
        "apply_fdr_correction", "cohens_d", "compute_effect_sizes", "bootstrap_r2_ci"}
KEEP_ASSIGN = {"STRATEGIES", "ALL_METHODS", "_GPU_LOO_CACHE"}
parts = []
for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in KEEP:
        parts.append(ast.get_source_segment(src, node))
    elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in KEEP_ASSIGN for t in node.targets):
        parts.append(ast.get_source_segment(src, node))

HEADER = '''# Auto-extracted from "source code.ipynb" (cell 2): definitions only, unchanged.
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
'''
io.open(OUT, "w", encoding="utf-8").write(HEADER + "\n\n" + "\n\n".join(parts) + "\n")
print("wrote", OUT, "with", len(parts), "definitions")
