import os
"""Rebuild the rice GRM exactly as the notebook's build_grm (snps_as_rows branch), vectorised for speed.

Encoding (notebook IUPAC_MAP): A,C -> 0 ; T,G -> 2 ; R,Y,S,W,K,M,B,D,H,V,N,- -> 1 ; anything else -> NaN,
NaN imputed by the per-SNP median, keep SNPs with 0.01 < freq < 0.99 where freq = mean/2,
standardise (M - 2 freq) / sqrt(2 freq (1 - freq)), GRM = sum M M^T / p_used (+1e-4 I added at load time).
"""
import time, numpy as np, pandas as pd

GENO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "Genotype Rice Data", "genotype.csv")
PHENO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "Genotype Rice Data", "phenotype.csv")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "work")
MAF_MIN = 0.01

pheno = pd.read_csv(PHENO).set_index("Taxa"); pheno.index = pheno.index.astype(str)
hdr = pd.read_csv(GENO, nrows=2, header=None).iloc[1]
gsor_cols = [i for i, v in enumerate(hdr) if str(v).startswith("GSOR")]
all_ids = [str(hdr.iloc[i]) for i in gsor_cols]
common = sorted(set(all_ids) & set(pheno.index))
idx = {s: i for i, s in enumerate(all_ids)}
order = [idx[s] for s in common]
n = len(common)
print("common samples", n, flush=True)

LUT = np.full(128, np.nan, dtype=np.float32)
for ch, v in {"A": 0, "T": 2, "C": 0, "G": 2, "R": 1, "Y": 1, "S": 1, "W": 1, "K": 1, "M": 1,
              "B": 1, "D": 1, "H": 1, "V": 1, "N": 1, "-": 1}.items():
    LUT[ord(ch)] = v
    LUT[ord(ch.lower())] = v

G = np.zeros((n, n)); p_used = 0; total = 0; t0 = time.time()
reader = pd.read_csv(GENO, header=None, skiprows=2, usecols=gsor_cols, chunksize=100_000, dtype=str,
                     keep_default_na=False)
for chunk in reader:
    a = chunk.values[:, order]                       # SNPs x samples (strings)
    s = np.char.strip(np.char.upper(a.astype("U4")))
    single = np.char.str_len(s) == 1
    codes = np.zeros(s.shape, dtype=np.int64)
    codes[single] = np.frombuffer(s[single].astype("U1").tobytes(), dtype=np.uint32)
    M = np.where(single & (codes < 128), LUT[np.clip(codes, 0, 127)], np.nan).astype(np.float32).T  # samples x SNPs
    med = np.nanmedian(M, axis=0)
    nan = np.isnan(M)
    if nan.any():
        M[nan] = np.take(med, np.where(nan)[1])
    freq = M.mean(0) / 2.0
    ok = (freq > MAF_MIN) & (freq < 1 - MAF_MIN)
    if ok.any():
        M = M[:, ok]; f = freq[ok]
        sc = np.sqrt(2 * f * (1 - f)); sc[sc == 0] = 1.0
        X = ((M - 2 * f) / sc).astype(np.float64)
        G += X @ X.T; p_used += X.shape[1]
    total += len(chunk)
    print(f"{total:,} SNPs read, {p_used:,} used, {time.time() - t0:.0f}s", flush=True)

G /= max(p_used, 1)
np.save(OUT + r"\rice_GRM.npy", G)
np.save(OUT + r"\rice_ids.npy", np.array(common))
print("done; p_used =", p_used, "time", round(time.time() - t0), "s", flush=True)
