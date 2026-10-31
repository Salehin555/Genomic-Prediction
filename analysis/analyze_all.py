"""Summarise every revision experiment into Markdown tables (results/summary.md) for the gate review.
Each section is skipped if its result file does not exist yet.  usage: python analyze_all.py"""
import os, json, glob
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
OUT = []


def md(df, floatfmt=4):
    return df.to_markdown(floatfmt=f".{floatfmt}f") if hasattr(df, "to_markdown") else df.to_string()


def h(t): OUT.append(f"\n## {t}\n")


def load(fn):
    p = os.path.join(RES, fn); return json.load(open(p)) if os.path.exists(p) else None


def corrected_t(d, n_tr, n_te):
    """Nadeau & Bengio (2003) corrected resampled t-test for paired fold differences d."""
    d = np.asarray(d, float); d = d[~np.isnan(d)]; J = len(d)
    if J < 2 or d.std(ddof=1) == 0: return 0.0, 1.0
    t = d.mean() / np.sqrt((1 / J + n_te / n_tr) * d.var(ddof=1))
    return float(t), float(2 * stats.t.sf(abs(t), J - 1))


def bh(p):
    p = np.asarray(p); o = np.argsort(p); q = np.empty_like(p); m = len(p)
    q[o] = np.minimum.accumulate((p[o] * m / np.arange(1, m + 1))[::-1])[::-1]
    return np.minimum(q, 1)


# ------------------------------------------------------------------ E1
def e1():
    R = load("e1_regimes.json")
    if R is None: return
    rows = []
    for r in R:
        b = r["r2"]["rrBLUP_CV"]
        d = dict(scn=r["scn"], n=r["n"], h2=r["h2"], p=r["test"]["p"], gam=r["test"]["gamma"])
        for m in ("oracle", "rrBLUP_ML", "gammaBLUP_ML", "power_CV", "safe_CV", "PCR_CV"): d[m] = r["r2"][m] - b
        for m, v in r["cross"].items(): d["x_" + m] = v
        d["mse_or"] = r["mse_g"]["oracle"]; rows.append(d)
    D = pd.DataFrame(rows); order = ["g0.5", "g0.75", "g1", "g1.25", "g1.5", "g2", "g3", "spiked"]
    D["scn"] = pd.Categorical(D.scn, order)
    h("E1 regime map: mean ΔR² vs rrBLUP_CV (×1000), averaged over h²")
    for m in ("oracle", "safe_CV", "power_CV", "gammaBLUP_ML", "PCR_CV"):
        OUT.append(f"\n**{m}**\n\n" + md(D.pivot_table(index="scn", columns="n", values=m, aggfunc="mean", observed=True) * 1000, 1))
    h("E1 safe selector: worst cell (mean ΔR² ×1000 over reps, per scenario × n × h²) and fraction of cells below −3×10⁻³")
    c = D.groupby(["scn", "n", "h2"], observed=True)["safe_CV"].mean() * 1000
    OUT.append(f"worst cell = {c.min():.2f} at {c.idxmin()};  cells below −3: {(c < -3).mean():.1%} of {len(c)}")
    h("E1 T1 check: cross term E[(f − oracle)(oracle − g)] (should be 0); mean ± 2 SE, ×1000")
    cr = {m[2:]: (D[m].mean() * 1000, 2 * D[m].std() / np.sqrt(len(D)) * 1000) for m in D.columns if m.startswith("x_")}
    OUT.append("\n".join(f"- {m}: {v[0]:+.3f} ± {v[1]:.3f}" for m, v in cr.items()))
    OUT.append(f"\n(for scale: mean oracle MSE vs g = {D.mse_or.mean()*1000:.1f} ×10⁻³)")
    h("E1/E2 diagnostic (χ² LRT on training data): rejection rate at 5%")
    OUT.append(md(D.assign(rej=D.p < 0.05).pivot_table(index="scn", columns="n", values="rej", aggfunc="mean", observed=True), 2))
    h("E1/E2 diagnostic as a screen: rejection vs held-out gain of safe_CV and power_CV over rrBLUP_CV")
    for g in ("safe_CV", "power_CV"):
        for nmin in (100, 400):
            S = D[D.n >= nmin]; pos = S.p < 0.05; gain = S[g] > 0
            tp, fp, fn, tn = (pos & gain).sum(), (pos & ~gain).sum(), (~pos & gain).sum(), (~pos & ~gain).sum()
            OUT.append(f"- {g}, n ≥ {nmin}: sensitivity {tp/(tp+fn):.2f}, specificity {tn/(tn+fp):.2f}, "
                       f"PPV {tp/max(tp+fp,1):.2f}, NPV {tn/max(tn+fn,1):.2f}; mean gain if rejected {S[g][pos].mean()*1000:+.2f}, "
                       f"if not rejected {S[g][~pos].mean()*1000:+.2f} (×10⁻³)")


# ------------------------------------------------------------------ E2/E3
def e3():
    R = load("e3_size.json")
    if R is not None:
        D = pd.DataFrame(R); h("E2 size and power of the γ = 1 test on real GRMs (h² = 0.5)")
        t = D.groupby(["data", "scn"]).agg(chi2=("p_chi2", lambda p: (p < .05).mean()), boot=("p_boot", lambda p: (p < .05).mean() if (p < 1).any() else np.nan), reps=("p_chi2", "size"))
        OUT.append(md(t, 3))
    R = load("e3_regimes.json")
    if R is not None:
        rows = [dict(data=r["data"], scn=r["scn"], h2=r["h2"], p=r["test"]["p"],
                     **{m: r["r2"][m] - r["r2"]["rrBLUP_CV"] for m in ("power_CV", "safe_CV", "gammaBLUP_ML")}) for r in R]
        D = pd.DataFrame(rows); h("E3 semi-synthetic on real GRMs: mean ΔR² vs rrBLUP_CV (×1000) and test rejection rate")
        t = D.groupby(["data", "scn"]).agg(safe=("safe_CV", "mean"), power=("power_CV", "mean"), gammaML=("gammaBLUP_ML", "mean"),
                                           reject=("p", lambda p: (p < .05).mean()))
        t[["safe", "power", "gammaML"]] *= 1000; OUT.append(md(t, 2))


# ------------------------------------------------------------------ E4/E5/E6/E9
def gather(dname, scheme):
    out = {}
    for f in glob.glob(os.path.join(RES, f"e4_{dname}_{scheme}_*.json")):
        for r in json.load(open(f)):
            o = out.setdefault(r["trait"], dict(n=r["n"], r2={}, preds={}, diag=None, info={}))
            o["r2"].update(r["r2"]); o["info"].update(r["info"])
            if r.get("diag"): o["diag"] = r["diag"]
            for rec in r["preds"]:
                k = (rec["seed"], rec["fold"]); o["preds"].setdefault(k, dict(y=rec["y"], p={}))["p"].update(rec["p"])
    if dname in ("rice", "wheat") and scheme == "kfold":                 # corrected SAS families from run_fixed.py (same splits)
        F = load(f"fixed_{dname}.json") or {}
        names = {"Env1": "Env1", "Env2": "Env2", "Env4": "Env4", "Env5": "Env5"}
        for t, v in F.items():
            tt = names.get(t, t)
            if tt in out:
                for fam in ("grouped", "smooth", "monotone", "spline"): out[tt]["r2"]["SAS_" + fam] = v["folds"][fam]
    return out


def e4(dname, scheme, title):
    T = gather(dname, scheme)
    if not T: return None
    h(title); rows = []
    for t, o in T.items():
        base = np.array(o["r2"].get("rrBLUP_CV", [np.nan])); n = o["n"]; row = dict(trait=t, n=n)
        if o["diag"]: row.update(gamma=o["diag"]["gamma"], p_boot=o["diag"]["p_boot"], p_chi2=o["diag"]["p_chi2"])
        row["rrBLUP_CV"] = np.nanmean(base)
        for m, v in o["r2"].items():
            if m == "rrBLUP_CV": continue
            v = np.array(v, float)
            if len(v) == len(base):
                row["Δ" + m] = np.nanmean(v - base); row["p_" + m] = corrected_t(v - base, 0.8 * n, 0.2 * n)[1]
            else:                                                       # baselines run on a subset of seeds
                k = len(v); row["Δ" + m] = np.nanmean(v - base[:k]); row["p_" + m] = corrected_t(v - base[:k], 0.8 * n, 0.2 * n)[1]
        rows.append(row)
    D = pd.DataFrame(rows).set_index("trait")
    for m in [c[2:] for c in D.columns if c.startswith("p_") and c not in ("p_boot", "p_chi2")]:
        D["q_" + m] = bh(D["p_" + m].fillna(1).values)
    show = ["n", "gamma", "p_boot", "rrBLUP_CV"] + [c for c in D.columns if c.startswith("Δ")]
    OUT.append(md(D[[c for c in show if c in D.columns]], 4))
    sig = [c for c in D.columns if c.startswith("q_")]
    OUT.append("\nBH-adjusted q < 0.05 (method: traits with significant gain / significant loss vs rrBLUP_CV):\n")
    for c in sig:
        m = c[2:]; g = ((D[c] < .05) & (D["Δ" + m] > 0)).sum(); l = ((D[c] < .05) & (D["Δ" + m] < 0)).sum()
        OUT.append(f"- {m}: {g} gains, {l} losses (of {len(D)})")
    if "p_boot" in D.columns and "Δsafe_CV" in D.columns:
        pos = D.p_boot < 0.05
        OUT.append(f"\nDiagnostic screen (bootstrap p < 0.05): rejected {pos.sum()}/{len(D)}; mean Δsafe_CV if rejected "
                   f"{D['Δsafe_CV'][pos].mean():+.4f}, if not {D['Δsafe_CV'][~pos].mean():+.4f}; "
                   f"traits with Δsafe>0: {(D['Δsafe_CV'][pos] > 0).sum()}/{pos.sum()} rejected vs {(D['Δsafe_CV'][~pos] > 0).sum()}/{(~pos).sum()} not rejected")
        OUT.append(f"Spearman(γ̂, Δpower_CV) = {stats.spearmanr(D.gamma, D['Δpower_CV'])[0]:+.2f}; "
                   f"Spearman(−log p_boot, Δsafe_CV) = {stats.spearmanr(-np.log(D.p_boot), D['Δsafe_CV'])[0]:+.2f}")
    # E9: selection overlap and accuracy ratio
    if scheme == "kfold":
        ov = []
        for t, o in T.items():
            for k, rec in o["preds"].items():
                y = np.array(rec["y"]); P = rec["p"]
                if "rrBLUP_CV" not in P: continue
                for m in ("safe_CV", "power_CV"):
                    if m not in P: continue
                    a, b = np.array(P["rrBLUP_CV"]), np.array(P[m]); q = max(1, int(round(0.1 * len(y))))
                    top = lambda v: set(np.argsort(-v)[:q])
                    ov.append(dict(trait=t, m=m, overlap10=len(top(a) & top(b)) / q,
                                   r_ratio=np.corrcoef(y, b)[0, 1] / np.corrcoef(y, a)[0, 1]))
        if ov:
            O = pd.DataFrame(ov).groupby(["trait", "m"]).mean().unstack("m")
            h(f"E9 practical effect ({dname}): top-10% overlap with rrBLUP and accuracy ratio r_m / r_rrBLUP (response ratio)")
            OUT.append(md(O, 3))
    return D


# ------------------------------------------------------------------ E7, E8, E10
def e7_e8_e10():
    S = load("e7_stability.json")
    if S:
        h("E7 stability: γ̂ with 95% bootstrap CI")
        OUT.append(md(pd.DataFrame([dict(data=r["dataset"], trait=r["trait"], gamma=r["gamma"], lo=r["ci"][0], hi=r["ci"][1]) for r in S["gamma_boot"]]).set_index(["data", "trait"]), 2))
        OUT.append("\nRice: F statistic of eigenvectors 1–10 on k-means subpopulation: " + ", ".join(f"{f:.0f}" for f in S["rice_F"][:10]))
    S = load("e8_scaling.json")
    if S:
        h("E8 scaling (CPU only, seconds)"); OUT.append(md(pd.DataFrame(S["results"]).set_index("n"), 2))
    S = load("e10_ablation.json")
    if S:
        D = pd.DataFrame(S); h("E10 ablation: mean held-out R² (and mean in-sample LOO) by variant")
        OUT.append(md(D.groupby(["dataset", "trait", "family", "variant"]).agg(r2=("r2", "mean"), worst=("r2", "min"), loo=("loo", "mean")).unstack("variant"), 3))


if __name__ == "__main__":
    e1(); e3()
    e4("rice", "kfold", "E4 rice (5×5 CV): R² of rrBLUP_CV and ΔR² of every method vs rrBLUP_CV")
    e4("wheat", "kfold", "E4 wheat (5×5 CV)")
    e4("mice", "kfold", "E4 mice (5×5 CV; covariates adjusted within folds)")
    e4("mice", "cage", "E6 mice, cage-out CV (whole cages held out; 3 seeds)")
    e4("mice_auto", "kfold", "E6 mice, autosomal-SNP GRM (3 seeds)")
    e4("rice", "subpop", "E6 rice, leave-one-subpopulation-out")
    e7_e8_e10()
    open(os.path.join(RES, "summary.md"), "w", encoding="utf-8").write("# Revision experiments — summary\n" + "\n".join(OUT))
    print("\n".join(OUT))
