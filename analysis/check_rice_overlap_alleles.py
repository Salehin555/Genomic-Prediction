"""Positive control for check_rice_overlap.py: do the shared (chromosome, position) pairs carry the same alleles in both
files? If the two coordinate systems agreed, most shared positions would show the same allele pair (up to strand).
usage: python check_rice_overlap_alleles.py
"""
import os, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RICE = os.path.join(HERE, "..", "data", "Genotype Rice Data", "genotype.csv")
PL = os.path.join(HERE, "data", "new_raw", "rice_rdp1", "plink", "RiceDiversity_44K_Genotypes_PLINK", "sativas413")
mp = pd.read_csv(PL + ".map", sep="\t", header=None, names=["chr", "id", "cm", "pos"])
key = {(int(c), int(p)): i for i, (c, p) in enumerate(zip(mp["chr"], mp["pos"]))}
ped = pd.read_csv(PL + ".ped", sep=r"\s+", header=None, dtype=str); A = ped.iloc[:, 6:].values
comp = str.maketrans("ACGT", "TGCA")
same = flip = other = 0
for ch in pd.read_csv(RICE, skiprows=1, chunksize=200000, low_memory=False, usecols=["alleles", "chrom", "pos"]):
    for al, c, p in zip(ch["alleles"], pd.to_numeric(ch["chrom"], errors="coerce").fillna(-1), pd.to_numeric(ch["pos"], errors="coerce").fillna(-1)):
        i = key.get((int(c), int(p)))
        if i is None: continue
        r = set(str(al).split("/")); d = set(np.unique(A[:, 2 * i: 2 * i + 2])) & set("ACGT")
        if not d: continue
        if d <= r: same += 1
        elif {x.translate(comp) for x in d} <= r: flip += 1
        else: other += 1
tot = same + flip + other
print(f"shared positions with calls: {tot}; same alleles {same} ({same / tot:.0%}), strand-flipped {flip} ({flip / tot:.0%}), different {other} ({other / tot:.0%})")
