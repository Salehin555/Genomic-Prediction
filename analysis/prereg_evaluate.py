"""Pre-registration step 2 (09 §4): compare the locked predictions with the held-out results.

usage: python prereg_evaluate.py PANEL
reads results/prereg_predictions_PANEL.json, results/e4_PANEL_kfold_spectral.json, results/e4_PANEL_kfold_kernel.json
writes results/prereg_evaluation_PANEL.json and prints the trait table and H1-H3.
"""
import os, sys, json, numpy as np
from scipy import stats
from analyze_all import corrected_t, bh

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")


def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (c - h, c + h)


if __name__ == "__main__":
    panel = sys.argv[1]
    P = {r["trait"]: r for r in json.load(open(os.path.join(RES, f"prereg_predictions_{panel}.json")))["traits"]}
    S = {t["trait"]: t for t in json.load(open(os.path.join(RES, f"e4_{panel}_kfold_spectral.json")))}
    K = {t["trait"]: t for t in json.load(open(os.path.join(RES, f"e4_{panel}_kfold_kernel.json")))}
    rows = []
    for tr, pr in P.items():
        base = np.array(S[tr]["r2"]["rrBLUP_CV"]); n = S[tr]["n"]; n_te = n / 5; n_tr = n - n_te; row = dict(trait=tr, category=pr["category"],
               gamma=pr["gamma"], p_boot=pr["p_boot"], rrBLUP=float(base.mean()))
        for m, src in (("power_CV", S), ("safe_CV", S), ("gammaBLUP_ML", S), ("KRR_RBF", K)):
            d = np.array(src[tr]["r2"][m]) - base; row[m] = float(np.nanmean(d)); row["p_" + m] = corrected_t(d, n_tr, n_te)[1]
        rows.append(row)
    for m in ("power_CV", "safe_CV", "gammaBLUP_ML", "KRR_RBF"):
        q = bh([r["p_" + m] for r in rows])
        for r, qq in zip(rows, q): r["q_" + m] = float(qq)

    def match(r):
        d, ds = r["power_CV"], r["safe_CV"]; sig_gain = (r["q_power_CV"] < .05 and d > 0) or (r["q_safe_CV"] < .05 and ds > 0)
        if r["category"] == "R+": return d > 0
        if r["category"] == "R-": return d < 0.005
        return abs(d) < 0.005 and not sig_gain
    for r in rows: r["prediction_met"] = bool(match(r))

    print(f"{'cat':3s} {'trait':38s} {'gamma':>6s} {'p':>6s} {'rrBLUP':>7s} {'dPower':>8s} {'q':>5s} {'dSafe':>8s} {'q':>5s} {'dGamML':>8s} {'dKRR':>8s} {'q':>5s} met")
    for r in sorted(rows, key=lambda r: (r["category"], -r["gamma"])):
        print(f"{r['category']:3s} {r['trait'][:38]:38s} {r['gamma']:6.2f} {r['p_boot']:6.3f} {r['rrBLUP']:7.3f} {r['power_CV']:+8.4f} {r['q_power_CV']:5.2f} "
              f"{r['safe_CV']:+8.4f} {r['q_safe_CV']:5.2f} {r['gammaBLUP_ML']:+8.4f} {r['KRR_RBF']:+8.4f} {r['q_KRR_RBF']:5.2f} {'yes' if r['prediction_met'] else 'NO'}")
    out = dict(panel=panel, rows=rows)
    rp = np.array([r["power_CV"] for r in rows if r["category"] == "R+"]); rn = np.array([r["power_CV"] for r in rows if r["category"] == "N"])
    if len(rp) and len(rn):
        obs = rp.mean() - rn.mean(); allv = np.concatenate([rp, rn]); rng = np.random.default_rng(0)
        perm = [(lambda s: s[:len(rp)].mean() - s[len(rp):].mean())(rng.permutation(allv)) for _ in range(20000)]
        out["H1"] = dict(mean_Rplus=float(rp.mean()), mean_N=float(rn.mean()), diff=float(obs), p_one_sided=float((1 + np.sum(np.array(perm) >= obs)) / 20001))
    else:
        out["H1"] = "not testable: no R+ trait"
    rho, p2 = stats.spearmanr([r["gamma"] for r in rows], [r["power_CV"] for r in rows]) if len(rows) > 2 else (np.nan, np.nan)
    out["H2"] = dict(spearman=float(rho), p_one_sided=float(p2 / 2 if rho > 0 else 1 - p2 / 2))
    k = sum(r["prediction_met"] for r in rows); out["H3"] = dict(met=k, n=len(rows), share=k / len(rows), wilson95=wilson(k, len(rows)))
    print("\nH1:", out["H1"]); print("H2:", out["H2"]); print("H3:", out["H3"])
    json.dump(out, open(os.path.join(RES, f"prereg_evaluation_{panel}.json"), "w"), indent=1)
