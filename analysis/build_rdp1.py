"""Build the Rice Diversity Panel 1 inputs (pre-registration 09 §3) from the original ricediversity.org files.

usage: python build_rdp1.py PED_FILE PHENO_FILE
writes data/rdp1_Z.npy (standardised markers, float32), data/rdp1_G.npy (GRM = Z Z' / p), data/rdp1_ids.npy (NSFTV ids),
       data/rdp1_pheno.csv (the 31 pre-registered traits, rows in GRM order) and data/rdp1_qc.json (QC counts).
Missing alleles are coded 0 in the PED file.
"""
import os, sys, json, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
EXCLUDE = ("Leaf pubescence", "Awn presence", "Seed color", "Pericarp color", "Blast resistance")


def read_ped(path):
    ids, rows = [], []
    with open(path) as fh:
        for line in fh:
            f = line.split(); ids.append(int(f[1])); rows.append(f[6:])
    A = np.array(rows, dtype="<U1"); return np.array(ids), A[:, 0::2], A[:, 1::2]


def dosage(a1, a2):
    """0/1/2 count of the minor allele per marker; NaN where either allele is missing ('0')."""
    n, p = a1.shape; X = np.full((n, p), np.nan)
    for j in range(p):
        al = np.concatenate([a1[:, j], a2[:, j]]); al = al[al != "0"]
        if al.size == 0: continue
        u, c = np.unique(al, return_counts=True); minor = u[np.argmin(c)] if len(u) > 1 else None
        ok = (a1[:, j] != "0") & (a2[:, j] != "0")
        X[ok, j] = 0.0 if minor is None else (a1[ok, j] == minor).astype(float) + (a2[ok, j] == minor)
    return X


if __name__ == "__main__":
    ped, pheno = sys.argv[1], sys.argv[2]
    ids, a1, a2 = read_ped(ped); X = dosage(a1, a2); qc = dict(n_raw=int(X.shape[0]), p_raw=int(X.shape[1]))
    keep_m = np.mean(~np.isnan(X), 0) >= 0.9; X = X[:, keep_m]
    keep_i = np.mean(~np.isnan(X), 1) >= 0.8; X, ids = X[keep_i], ids[keep_i]
    f = np.nanmean(X, 0) / 2; maf = np.minimum(f, 1 - f); keep_f = maf >= 0.01; X, f = X[:, keep_f], f[keep_f]
    X = np.where(np.isnan(X), 2 * f, X)
    Z = (X - 2 * f) / np.sqrt(2 * f * (1 - f)); G = Z @ Z.T / Z.shape[1]
    qc.update(markers_callrate=int(keep_m.sum()), individuals_callrate=int(keep_i.sum()), markers_final=int(Z.shape[1]),
              mean_diag_G=float(np.mean(np.diag(G))))
    P = pd.read_csv(pheno, sep="\t").set_index("NSFTVID")
    traits = [c for c in P.columns[1:] if c not in EXCLUDE]
    T = P.reindex(ids)[traits].apply(pd.to_numeric, errors="coerce")
    sel = [c for c in traits if T[c].notna().sum() >= 300 and T[c].nunique() >= 30]
    qc.update(traits=sel, n_traits=len(sel))
    np.save(os.path.join(DATA, "rdp1_Z.npy"), Z.astype(np.float32)); np.save(os.path.join(DATA, "rdp1_G.npy"), G)
    np.save(os.path.join(DATA, "rdp1_ids.npy"), ids); T[sel].to_csv(os.path.join(DATA, "rdp1_pheno.csv"), index_label="NSFTVID")
    json.dump(qc, open(os.path.join(DATA, "rdp1_qc.json"), "w"), indent=1); print(json.dumps({k: v for k, v in qc.items() if k != "traits"}))
