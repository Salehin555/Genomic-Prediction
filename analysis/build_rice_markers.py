"""Rice marker matrix for marker-based Bayesian baselines: same decoding and MAF filter as build_rice_grm.py,
keeping every 100th SNP that passes the filter (about 28k markers). Writes data/rice_X.npy (dosages, float32)."""
import os, time, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
GENO = os.path.join(HERE, "..", "data", "Genotype Rice Data", "genotype.csv")
ids = np.load(os.path.join(HERE, "data", "rice_ids.npy")).tolist()
hdr = pd.read_csv(GENO, nrows=2, header=None).iloc[1]
gsor = [i for i, v in enumerate(hdr) if str(v).startswith("GSOR")]
all_ids = [str(hdr.iloc[i]) for i in gsor]; pos = {s: i for i, s in enumerate(all_ids)}
order = [pos[s] for s in ids]
LUT = np.full(128, np.nan, dtype=np.float32)
for ch, v in {"A": 0, "T": 2, "C": 0, "G": 2, "R": 1, "Y": 1, "S": 1, "W": 1, "K": 1, "M": 1,
              "B": 1, "D": 1, "H": 1, "V": 1, "N": 1, "-": 1}.items():
    LUT[ord(ch)] = v; LUT[ord(ch.lower())] = v
keep, seen, t0 = [], 0, time.time()
for chunk in pd.read_csv(GENO, header=None, skiprows=2, usecols=gsor, chunksize=100_000, dtype=str, keep_default_na=False):
    s = np.char.strip(np.char.upper(chunk.values[:, order].astype("U4")))
    single = np.char.str_len(s) == 1
    codes = np.zeros(s.shape, dtype=np.int64)
    codes[single] = np.frombuffer(s[single].astype("U1").tobytes(), dtype=np.uint32)
    M = np.where(single & (codes < 128), LUT[np.clip(codes, 0, 127)], np.nan).astype(np.float32).T
    med = np.nanmedian(M, axis=0); nan = np.isnan(M)
    if nan.any(): M[nan] = np.take(med, np.where(nan)[1])
    f = M.mean(0) / 2.0; ok = np.where((f > 0.01) & (f < 0.99))[0]
    sel = ok[(np.arange(len(ok)) + seen) % 100 == 0]; seen += len(ok)
    keep.append(M[:, sel])
    print(f"{seen:,} filtered SNPs scanned, {sum(k.shape[1] for k in keep):,} kept, {time.time()-t0:.0f}s", flush=True)
X = np.concatenate(keep, axis=1)
np.save(os.path.join(HERE, "data", "rice_X.npy"), X)
print("DONE", X.shape, round(time.time() - t0), "s")
