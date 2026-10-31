"""Leakage-free check of the diagnostic: in every fold, the REML gamma = 1 test computed on the TRAINING part only
(e13_reml_foldiag.json) vs the held-out gain of each method over rrBLUP_CV in the same fold (e4 predictions).
Uncertainty: bootstrap over traits (whole traits resampled), 2000 draws. writes results/fold_association.json"""
import json, sys, numpy as np

def fold_r2(rec, m):
    y = np.array(rec["y"]); p = np.array(rec["p"][m]); return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)

OUT = []
F = {(t["dataset"], t["trait"]): {(f["seed"], f["fold"]): f for f in t["folds"]} for t in json.load(open("results/e13_reml_foldiag.json"))}
for ds in ("mice", "wheat", "rice"):
    E = {t["trait"]: t for t in json.load(open(f"results/e4_{ds}_kfold_spectral.json"))}
    E11 = {t["trait"]: {(f["seed"], f["fold"]): f["r2"] for f in t["folds"]} for t in json.load(open(f"results/e11_epi_{ds}.json"))}
    for m in ("safe_CV", "power_CV", "epi"):
        rows = []                                                       # (trait index, rejected, gain)
        for k, (tr, T) in enumerate(E.items()):
            for rec in T["preds"]:
                key = (rec["seed"], rec["fold"]); base = fold_r2(rec, "rrBLUP_CV")
                gain = (E11[tr][key] if m == "epi" else fold_r2(rec, m)) - base
                rows.append((k, F[(ds, tr)][key]["p_chi2"] < .05, gain))
        a = np.array(rows, float); rej = a[:, 1] == 1
        if rej.sum() == 0:
            print(f"{ds:5s} {m:8s} no rejected folds"); continue
        diff = a[rej, 2].mean() - a[~rej, 2].mean()
        rng = np.random.default_rng(0); traits = np.unique(a[:, 0]); bs = []
        for _ in range(2000):
            s = np.concatenate([a[a[:, 0] == t] for t in rng.choice(traits, len(traits))]); r = s[:, 1] == 1
            if r.any() and (~r).any(): bs.append(s[r, 2].mean() - s[~r, 2].mean())
        lo, hi = np.percentile(bs, [2.5, 97.5])
        OUT.append(dict(dataset=ds, method=m, folds_rejected=int(rej.sum()), folds=len(a), gain_rejected=float(a[rej, 2].mean()),
                        gain_not=float(a[~rej, 2].mean()), diff=float(diff), lo=float(lo), hi=float(hi)))
        print(f"{ds:5s} {m:8s} folds rejected {int(rej.sum()):3d}/{len(a)}  mean gain rejected {a[rej, 2].mean():+.4f}  not {a[~rej, 2].mean():+.4f}  "
              f"difference {diff:+.4f} [95% trait-bootstrap CI {lo:+.4f}, {hi:+.4f}]  P(gain>0 | rej) {np.mean(a[rej, 2] > 0):.2f} vs {np.mean(a[~rej, 2] > 0):.2f}")
json.dump(OUT, open("results/fold_association.json", "w"), indent=1)
