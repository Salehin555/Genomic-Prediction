"""Build the sorghum inputs (pre-registration 09 §3) from the Azodi et al. (2019) Dryad files.

usage: python build_sorghum.py GENO_CSV PHENO_CSV
Genotypes are coded -1/0/1 (Azodi et al.); they are shifted to 0/1/2 and given the same QC as build_rdp1.py
(marker call rate >= 0.9, individual call rate >= 0.8, MAF >= 0.01, mean imputation).
writes data/sorghum_Z.npy, data/sorghum_G.npy, data/sorghum_ids.npy, data/sorghum_pheno.csv, data/sorghum_qc.json
"""
import os, sys, json, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")

if __name__ == "__main__":
    geno, pheno = sys.argv[1], sys.argv[2]
    Gd = pd.read_csv(geno, index_col=0); X = Gd.values.astype(float) + 1.0; ids = Gd.index.astype(str).values
    X[(X < 0) | (X > 2)] = np.nan; qc = dict(n_raw=int(X.shape[0]), p_raw=int(X.shape[1]), values=sorted(map(float, np.unique(Gd.values))))
    keep_m = np.mean(~np.isnan(X), 0) >= 0.9; X = X[:, keep_m]
    keep_i = np.mean(~np.isnan(X), 1) >= 0.8; X, ids = X[keep_i], ids[keep_i]
    f = np.nanmean(X, 0) / 2; maf = np.minimum(f, 1 - f); keep_f = maf >= 0.01; X, f = X[:, keep_f], f[keep_f]
    X = np.where(np.isnan(X), 2 * f, X)
    Z = (X - 2 * f) / np.sqrt(2 * f * (1 - f)); G = Z @ Z.T / Z.shape[1]
    P = pd.read_csv(pheno, index_col=0); P.index = P.index.astype(str); P = P.reindex(ids)
    qc.update(markers_callrate=int(keep_m.sum()), individuals_callrate=int(keep_i.sum()), markers_final=int(Z.shape[1]),
              mean_diag_G=float(np.mean(np.diag(G))), traits=list(P.columns), n_per_trait={c: int(P[c].notna().sum()) for c in P.columns})
    np.save(os.path.join(DATA, "sorghum_Z.npy"), Z.astype(np.float32)); np.save(os.path.join(DATA, "sorghum_G.npy"), G)
    np.save(os.path.join(DATA, "sorghum_ids.npy"), ids); P.to_csv(os.path.join(DATA, "sorghum_pheno.csv"))
    json.dump(qc, open(os.path.join(DATA, "sorghum_qc.json"), "w"), indent=1); print(json.dumps(qc))
