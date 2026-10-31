"""Do the 191 accessions of the original rice panel overlap the 413 accessions of RDP1 (the pre-registered panel)?

The two files use different identifiers (GSOR numbers vs NSFTV numbers) and no mapping is distributed with them, so
the check is genetic: on SNPs at the same chromosome and position in both files, compute the genotype concordance of
every rice-RDP1 pair. A duplicated accession shows concordance near 1; unrelated inbred accessions are far lower.
writes results/rice_rdp1_overlap.json;  usage: python check_rice_overlap.py
"""
import os, json, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RICE = os.path.join(HERE, "..", "data", "Genotype Rice Data", "genotype.csv")
PL = os.path.join(HERE, "data", "new_raw", "rice_rdp1", "plink", "RiceDiversity_44K_Genotypes_PLINK", "sativas413")

mp = pd.read_csv(PL + ".map", sep="\t", header=None, names=["chr", "id", "cm", "pos"])
key = {(int(c), int(p)): i for i, (c, p) in enumerate(zip(mp["chr"], mp["pos"]))}

hits, calls, alle = [], [], []
cols = None
for ch in pd.read_csv(RICE, skiprows=1, chunksize=200000, low_memory=False):
    if cols is None:
        cols = [c for c in ch.columns if str(c).startswith("GSOR")]
    m = [(int(c), int(p)) in key for c, p in zip(pd.to_numeric(ch["chrom"], errors="coerce").fillna(-1), pd.to_numeric(ch["pos"], errors="coerce").fillna(-1))]
    sub = ch[m]
    if len(sub):
        hits += [key[(int(c), int(p))] for c, p in zip(sub["chrom"], sub["pos"])]
        calls.append(sub[cols].astype(str).values); alle += [set(str(a).split("/")) for a in sub["alleles"]]
R = np.vstack(calls)                                   # shared SNPs x 191, single-letter calls
print("shared positions:", len(hits), "rice accessions:", len(cols), flush=True)

ped = pd.read_csv(PL + ".ped", sep=r"\s+", header=None, dtype=str)
ids = ped[1].tolist(); A = ped.iloc[:, 6:].values      # 413 x 2*36901 alleles
a1, a2 = A[:, 0::2], A[:, 1::2]
D = np.where(a1 == a2, a1, "H")[:, hits].T             # shared SNPs x 413; heterozygous or missing marked
# keep only positions whose alleles agree (after a strand flip where needed): the two files use slightly different
# coordinate systems, and a position with different alleles is not the same SNP
comp = str.maketrans("ACGT", "TGCA"); keep = []
for k, r in enumerate(alle):
    d = set(np.unique(D[k])) & set("ACGT")
    if d and d <= r: keep.append(k)
    elif d and {x.translate(comp) for x in d} <= r:
        D[k] = np.array([x.translate(comp) for x in D[k]]); keep.append(k)
R, D, hits = R[keep], D[keep], [hits[k] for k in keep]
print("positions with matching alleles:", len(keep), flush=True)
ok_r = np.isin(R, list("ACGT")); ok_d = np.isin(D, list("ACGT"))
conc = np.zeros((R.shape[1], D.shape[1])); n = np.zeros_like(conc)
for j in range(D.shape[1]):
    both = ok_r & ok_d[:, [j]]
    n[:, j] = both.sum(0); conc[:, j] = ((R == D[:, [j]]) & both).sum(0) / np.maximum(n[:, j], 1)
best = conc.max(1); arg = conc.argmax(1)
second = np.sort(conc, 1)[:, -2]
# control: the closest other accession within RDP1 on the same SNPs (no RDP1 accession is listed twice)
okD = np.isin(D, list("ACGT")); nnD = np.zeros(D.shape[1])
for j in range(D.shape[1]):
    both = okD & okD[:, [j]]; cj = ((D == D[:, [j]]) & both).sum(0) / np.maximum(both.sum(0), 1); cj[j] = -1; nnD[j] = cj.max()
print("within-RDP1 nearest-neighbour concordance quantiles", np.round(np.percentile(nnD, [0, 25, 50, 75, 100]), 3))
print("rice best minus second-best: quantiles", np.round(np.percentile(best - second, [0, 50, 90, 100]), 3))
out = dict(shared_snps=len(hits), rice=len(cols), rdp1=len(ids), median_pair_concordance=float(np.median(conc)),
           best_match=[dict(rice=cols[i], rdp1=ids[arg[i]], concordance=float(best[i]), n=int(n[i, arg[i]])) for i in range(len(cols))],
           n_rice_with_match_ge_0_98=int(np.sum(best >= 0.98)), n_rice_with_match_ge_0_95=int(np.sum(best >= 0.95)),
           rdp1_nearest_neighbour_median=float(np.median(nnD)), best_minus_second_median=float(np.median(best - second)))
json.dump(out, open(os.path.join(HERE, "results", "rice_rdp1_overlap.json"), "w"), indent=1)
q = np.percentile(best, [0, 25, 50, 75, 100])
print("median pair concordance", round(out["median_pair_concordance"], 3), "best-match quantiles", np.round(q, 3))
print("rice accessions with a best match >= 0.98:", out["n_rice_with_match_ge_0_98"], " >= 0.95:", out["n_rice_with_match_ge_0_95"])
