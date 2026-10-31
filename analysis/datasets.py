"""Loaders for the revision experiments. Each loader returns a dict with
  G       genomic relationship matrix (VanRaden, + 1e-4 I)
  Z       standardised marker matrix used for G (for marker-based Bayesian baselines), or None
  traits  {name: dict(y=..., rows=index array into G, C=covariate design matrix or None, groups=... or None)}
Covariates are returned as a design matrix (intercept + dummies); adjustment is done inside each training fold.
"""
import os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RICE_PHENO = os.path.join(HERE, "..", "data", "Genotype Rice Data", "phenotype.csv")

MICE_OBESITY = ["Obesity.BMI", "Obesity.BodyLength", "Obesity.EndNormalBW"]
MICE_BIOCHEM = ["Biochem.Albumin", "Biochem.ALP", "Biochem.ALT", "Biochem.AST", "Biochem.Calcium", "Biochem.Chloride",
                "Biochem.Creatinine", "Biochem.Glucose", "Biochem.HDL", "Biochem.LDL", "Biochem.Phosphorous",
                "Biochem.Sodium", "Biochem.Tot.Cholesterol", "Biochem.Tot.Protein", "Biochem.Triglycerides",
                "Biochem.Urea"]


def standardise(X):
    f = X.mean(0) / 2.0; ok = (f > 0.01) & (f < 0.99)
    Z = (X[:, ok] - 2 * f[ok]) / np.sqrt(2 * f[ok] * (1 - f[ok]))
    return Z, ok


def _design(df):
    parts = [np.ones((len(df), 1))]
    for c in df.columns:
        if df[c].dtype.kind in "fi" and df[c].nunique() > 12:
            v = df[c].values.astype(float); parts.append(((v - v.mean()) / v.std())[:, None])
        else:
            parts.append(pd.get_dummies(df[c].astype(str), drop_first=True).values.astype(float))
    return np.hstack(parts)


def rice(markers=True):
    G = np.load(os.path.join(DATA, "rice_GRM.npy")) + 1e-4 * np.eye(191)
    ids = np.load(os.path.join(DATA, "rice_ids.npy")).tolist()
    ph = pd.read_csv(RICE_PHENO).set_index("Taxa"); ph.index = ph.index.astype(str); ph = ph.loc[ids]
    Z = None
    if markers and os.path.exists(os.path.join(DATA, "rice_X.npy")):
        Z, _ = standardise(np.load(os.path.join(DATA, "rice_X.npy")).astype(np.float64))
    cl = rice_clusters(G)
    traits = {}
    for t in ph.columns:
        y = ph[t].values.astype(float); rows = np.where(~np.isnan(y))[0]
        traits[t] = dict(y=y[rows], rows=rows, C=None, groups=cl[rows])
    return dict(name="rice", G=G, Z=Z, traits=traits)


def rice_clusters(G, k=4, n_pc=3):
    """Subpopulation labels for the structure-aware CV (E6): k-means on the leading GRM eigenvectors."""
    from sklearn.cluster import KMeans
    ev, U = np.linalg.eigh(G)
    pcs = U[:, ::-1][:, :n_pc] * np.sqrt(ev[::-1][:n_pc])
    return KMeans(k, n_init=50, random_state=0).fit_predict(pcs)


def wheat():
    import pyreadr
    r = pyreadr.read_r(os.path.join(DATA, "wheat.RData"))
    Z, _ = standardise(r["wheat.X"].values.astype(np.float64) * 2.0)
    G = Z @ Z.T / Z.shape[1] + 1e-4 * np.eye(len(Z))
    Y = r["wheat.Y"]; traits = {}
    for c in Y.columns:
        traits["Env%s" % c] = dict(y=Y[c].values.astype(float), rows=np.arange(len(Y)), C=None, groups=None)
    return dict(name="wheat", G=G, Z=Z, traits=traits)


def mice(autosomes_only=False):
    import pyreadr
    r = pyreadr.read_r(os.path.join(DATA, "mice.RData"))
    X = r["mice.X"].values.astype(np.float64)
    if autosomes_only:
        X = X[:, (r["mice.map"]["chr"].astype(str) != "X").values]
    Z, _ = standardise(X)
    G = Z @ Z.T / Z.shape[1] + 1e-4 * np.eye(len(Z))
    ph = r["mice.pheno"]; traits = {}
    ph = ph.assign(ObDate=ph["Obesity.Date.Year"].astype(str) + ph["Obesity.Date.Season"].astype(str),
                   BcDate=ph["Biochem.Date.Year"].astype(str) + ph["Biochem.Date.Season"].astype(str))
    for t in MICE_OBESITY + MICE_BIOCHEM:
        cov = ["GENDER", "Litter", "CageDensity"] + (["ObDate"] if t.startswith("Obesity") else ["BcDate", "Biochem.Age"])
        df = ph[[t] + cov]
        ok = df[t].notna() & df[cov].notna().all(axis=1) & ~df[cov].astype(str).apply(lambda s: s.str.contains("nan")).any(axis=1)
        rows = np.where(ok.values)[0]
        traits[t] = dict(y=df[t].values[rows].astype(float), rows=rows, C=_design(df[cov].iloc[rows]),
                         groups=ph["cage"].values[rows])
    return dict(name="mice" + ("_auto" if autosomes_only else ""), G=G, Z=Z, traits=traits)


def rdp1():
    """Rice Diversity Panel 1 (413 accessions; pre-registered traits, 09 §3); built by build_rdp1.py."""
    G = np.load(os.path.join(DATA, "rdp1_G.npy")) + 1e-4 * np.eye(413); Z = np.load(os.path.join(DATA, "rdp1_Z.npy"))
    P = pd.read_csv(os.path.join(DATA, "rdp1_pheno.csv"), index_col=0); traits = {}
    for c in P.columns:
        rows = np.where(P[c].notna().values)[0]
        traits[c] = dict(y=P[c].values[rows].astype(float), rows=rows, C=None, groups=None)
    return dict(name="rdp1", G=G, Z=Z, traits=traits)


def sorghum():
    """Sorghum diversity panel as formatted by Azodi et al. (2019); built by build_sorghum.py (09 §3)."""
    G = np.load(os.path.join(DATA, "sorghum_G.npy")); G = G + 1e-4 * np.eye(len(G)); Z = np.load(os.path.join(DATA, "sorghum_Z.npy"))
    P = pd.read_csv(os.path.join(DATA, "sorghum_pheno.csv"), index_col=0); traits = {}
    for c in P.columns:
        rows = np.where(P[c].notna().values)[0]
        traits[c] = dict(y=P[c].values[rows].astype(float), rows=rows, C=None, groups=None)
    return dict(name="sorghum", G=G, Z=Z, traits=traits)


LOADERS = {"rice": rice, "wheat": wheat, "mice": mice, "mice_auto": lambda: mice(True), "rdp1": rdp1, "sorghum": sorghum}
