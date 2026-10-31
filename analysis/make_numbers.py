"""Every number the revised manuscript and the response may quote, computed from results/ (Planning/08 section 3.3).

Each entry: key -> {value, text, claim, source}. `text` is the exact string the manuscript uses; the number checker
(_tools/check_numbers.py) accepts a number in revised text only if it equals some `text` here or is whitelisted.
Claims refer to Planning/10 (K1-K25) and registers rows to Planning/07.
usage: python make_numbers.py          writes results/numbers.json and prints key = text
"""
import os, json, glob, sys
import numpy as np
from scipy import stats
from analyze_all import corrected_t, bh

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
NUM = {}


def J(fn): return json.load(open(os.path.join(RES, fn)))


def put(key, value, text, claim, source):
    NUM[key] = dict(value=value if not isinstance(value, np.generic) else value.item(), text=text, claim=claim, source=source)


def f(x, d=3, sign=False):
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", "−")


def by_trait(fn): return {t["trait"]: t for t in J(fn)}


# ------------------------------------------------------------ simulation E1 (K1, K3-K6)
def e1():
    R = J("e1_regimes.json"); src = "e1_regimes.json"
    put("e1.runs", len(R), f"{len(R):,}", "K1", src)
    cross = {m: np.array([r["cross"][m] for r in R]) for m in R[0]["cross"]}
    ok = all(abs(v.mean()) <= 2 * v.std() / np.sqrt(len(v)) for v in cross.values())
    put("e1.cross_within_2se", bool(ok), "within 2 SE of 0", "K1", src)
    d = lambda r, m: r["r2"][m] - r["r2"]["rrBLUP_CV"]
    def cell(scn, n, m):
        v = np.array([d(r, m) for r in R if r["scn"] == scn and r["n"] == n]); se = v.std(ddof=1) / np.sqrt(len(v))
        return v.mean(), v.mean() - 1.96 * se, v.mean() + 1.96 * se
    ns = sorted({r["n"] for r in R})
    for scn in ("g2", "g3", "g0.5", "spiked"):
        for n in ns:
            for m in ("safe_CV", "power_CV", "gammaBLUP_ML", "oracle"):
                mu, lo, hi = cell(scn, n, m)
                put(f"e1.{scn}.n{n}.{m}", [mu, lo, hi], f(mu, 4, True), "K3-K6", src)
    big = [cell(s, n, "safe_CV") for s in ("g2", "g3") for n in ns if n >= 400]
    put("e1.safe_gain_g2plus.range", [min(b[0] for b in big), max(b[0] for b in big)],
        f"{f(min(b[0] for b in big), 3, True)} to {f(max(b[0] for b in big), 3, True)}", "K3", src)
    put("e1.safe_gain_g2plus.all_ci_above_0", all(b[1] > 0 for b in big), "all 95% CIs above 0", "K3", src)
    pl = [cell("g0.5", n, "gammaBLUP_ML")[0] for n in ns]
    put("e1.g05.plugin.range", [min(pl), max(pl)], f"{f(min(pl), 3)} to {f(max(pl), 3)}", "K4", src)
    for n in (400, 800, 1600):
        put(f"e1.g05.n{n}.safe_vs_oracle", [cell("g0.5", n, "safe_CV")[0], cell("g0.5", n, "oracle")[0]],
            f"{f(cell('g0.5', n, 'safe_CV')[0], 3, True)} of {f(cell('g0.5', n, 'oracle')[0], 3, True)}", "K4", src)
    sp = [cell("spiked", n, "safe_CV")[0] for n in (400, 800, 1600)]
    put("e1.spiked.safe", sp, " / ".join(f(x, 4, True) for x in sp), "K5", src)
    cells = {}
    for r in R: cells.setdefault((r["scn"], r["n"], r["h2"]), []).append(d(r, "safe_CV"))
    cm = {k: np.mean(v) for k, v in cells.items()}; worst = min(cm, key=cm.get)
    put("e1.safe.worst_cell", [cm[worst], list(worst)], f(cm[worst], 3), "K6", src)
    share = np.mean([v >= -0.003 for v in cm.values()])
    put("e1.safe.share_cells_loss_lt_0.003", share, f"{share:.0%}", "K6", src)
    put("e1.cells", len(cm), str(len(cm)), "K6", src)


# ------------------------------------------------------------ REML test (K9-K11, B7)
def reml_test():
    R = J("e1_reml_test.json"); src = "e1_reml_test.json"
    rej = lambda scn, n: np.mean([r["p_chi2"] < .05 for r in R if r["scn"] == scn and r["n"] == n])
    ns = sorted({r["n"] for r in R})
    size = [rej("g1", n) for n in ns]
    put("test.e1.size.range", [min(size), max(size)], f"{min(size):.2f}–{max(size):.2f}", "K9", src)
    for scn in ("g0.5", "g2", "g3"):
        for n in ns: put(f"test.power.{scn}.n{n}", rej(scn, n), f"{rej(scn, n):.2f}", "K10", src)
    put("test.power.g05.small_n", [rej("g0.5", n) for n in ns if n <= 200], "–".join(f"{x:.2f}" for x in sorted(rej("g0.5", n) for n in ns if n <= 200)), "K10", src)
    sp = [rej("spiked", n) for n in ns if n >= 400]
    put("test.spiked.reject.range", [min(sp), max(sp)], f"{min(sp):.0%}–{max(sp):.0%}".replace("%–", "–"), "K11", src)
    # R5.5: how well a rejection predicts that the safe selector beats rrBLUP on the new individuals of the same run
    from scipy.stats import spearmanr
    E = {(r["scn"], r["n"], r["h2"], r["rep"], r["seed"]): r for r in J("e1_regimes.json")}
    rows = [(r["n"], r["p_chi2"] < .05, -np.log10(max(r["p_chi2"], 1e-300)),
             E[(r["scn"], r["n"], r["h2"], r["rep"], r["seed"])]["r2"]["safe_CV"] - E[(r["scn"], r["n"], r["h2"], r["rep"], r["seed"])]["r2"]["rrBLUP_CV"])
            for r in R]
    def screen(sub, tag):
        rej = np.array([x[1] for x in sub]); pos = np.array([x[3] > 0 for x in sub])
        st = dict(sens=np.mean(rej[pos]), spec=np.mean(~rej[~pos]), ppv=np.mean(pos[rej]), npv=np.mean(~pos[~rej]),
                  prev=np.mean(pos), rejrate=np.mean(rej))
        for k, v in st.items(): put(f"test.screen.{tag}.{k}", float(v), f"{v:.2f}", "K10", src)
        put(f"test.screen.{tag}.runs", len(sub), f"{len(sub):,}", "K10", src)
        return st
    screen(rows, "all")
    for n in ns: screen([x for x in rows if x[0] == n], f"n{n}")
    gr, gn = np.mean([x[3] for x in rows if x[1]]), np.mean([x[3] for x in rows if not x[1]])
    put("test.screen.gain_if_rejected", float(gr), f(gr, 4, True), "K10", src)
    put("test.screen.gain_if_not", float(gn), f(gn, 4, True), "K10", src)
    rho = spearmanr([x[2] for x in rows], [x[3] for x in rows])[0]
    put("test.screen.spearman", float(rho), f"{rho:.2f}", "K10", src)
    S = J("e13_reml_size.json"); src = "e13_reml_size.json"
    for dn in ("rice", "mice"):
        rows = [r for r in S if r["data"] == dn]
        for k, lab in (("p_old", "old"), ("p_reml", "reml"), ("p_reml_boot", "reml_boot")):
            v = np.mean([r[k] < .05 for r in rows if r[k] is not None]); put(f"test.size.{dn}.{lab}", v, f"{v:.3f}", "K9" if lab != "old" else "B7", src)


# ------------------------------------------------------------ real data, REML diagnostic and gains (K12-K16, K19)
def gains(fn, m, base="rrBLUP_CV", k=None):
    out = {}
    for t, T in by_trait(fn).items():
        b = np.array(T["r2"][base], float); v = np.array(T["r2"][m], float); L = min(len(b), len(v))
        dd = v[:L] - b[:L]; n = T["n"]; out[t] = (float(np.nanmean(dd)), corrected_t(dd, 0.8 * n, 0.2 * n)[1])
    q = bh([p for _, p in out.values()])
    return {t: (mu, p, qq) for (t, (mu, p)), qq in zip(out.items(), q)}


def fisher_split(rej, G):
    a = sum(1 for t in rej if rej[t] and G[t][0] > 0); b = sum(rej.values()); c = sum(1 for t in rej if not rej[t] and G[t][0] > 0); dn = len(rej) - b
    p = stats.fisher_exact([[a, b - a], [c, dn - c]], alternative="greater")[1]
    return a, b, c, dn, p


def real():
    E = J("e13_reml_real.json")
    rejected = {(r["dataset"], r["trait"]): r["p_boot"] < .05 for r in E}
    for ds in ("rice", "wheat", "mice", "mice_auto"):
        k = sum(v for (d, _), v in rejected.items() if d == ds); n = sum(1 for (d, _) in rejected if d == ds)
        put(f"real.{ds}.rejected", [k, n], f"{k}/{n}", "K12", "e13_reml_real.json")
    # R2.5: the diagnostic on the rice panel (Table "The diagnostic on the rice panel")
    gap = {(r["dataset"], r["trait"]): r for r in J("gap_real.json")}
    rice = [r for r in E if r["dataset"] == "rice"]
    for r in rice:
        t = r["trait"].replace("/", "_"); g = gap[("rice", r["trait"])]
        put(f"real.rice.{t}.gamma", r["gamma"], f"{r['gamma']:.2f}", "K15", "e13_reml_real.json")
        put(f"real.rice.{t}.p_boot", r["p_boot"], f"{r['p_boot']:.2f}", "K12", "e13_reml_real.json")
        put(f"real.rice.{t}.gap", g["gap_r2"], f"{g['gap_r2']:.4f}", "K17", "gap_real.json")
    put("real.rice.p_boot.range", [min(r["p_boot"] for r in rice), max(r["p_boot"] for r in rice)],
        f"{min(r['p_boot'] for r in rice):.2f} to {max(r['p_boot'] for r in rice):.2f}", "K12", "e13_reml_real.json")
    put("real.rice.gamma.range", [min(r["gamma"] for r in rice), max(r["gamma"] for r in rice)],
        f"{min(r['gamma'] for r in rice):.2f} to {max(r['gamma'] for r in rice):.2f}", "K15", "e13_reml_real.json")
    mg = max(gap[("rice", r["trait"])]["gap_r2"] for r in rice)
    put("real.rice.gap.max", mg, f"{mg:.4f}", "K17", "gap_real.json")
    for ds, scheme, lab in (("mice", "kfold", "kfold"), ("mice", "cage", "cage"), ("mice_auto", "kfold", "auto")):
        rej = {t: v for (d, t), v in rejected.items() if d == ds}
        for m, fn in (("power_CV", f"e4_{ds}_{scheme}_spectral.json"), ("safe_CV", f"e4_{ds}_{scheme}_spectral.json"), ("KRR_RBF", f"e4_{ds}_{scheme}_kernel.json")):
            if not os.path.exists(os.path.join(RES, fn)): continue
            G = gains(fn, m, base="rrBLUP_CV") if m != "KRR_RBF" else None
            if m == "KRR_RBF":
                S = by_trait(f"e4_{ds}_{scheme}_spectral.json"); K = by_trait(fn); G = {}
                for t in K:
                    b = np.array(S[t]["r2"]["rrBLUP_CV"]); v = np.array(K[t]["r2"][m]); L = min(len(b), len(v)); G[t] = (float(np.nanmean(v[:L] - b[:L])),)
            a, b, c, dn, p = fisher_split(rej, G)
            put(f"assoc.{lab}.{m}", [a, b, c, dn, p], f"{a}/{b} vs {c}/{dn}, p = {p:.3f}", "K12", fn)
    for ds in ("rice", "wheat", "mice"):
        for m in ("power_CV", "safe_CV", "gammaBLUP_ML", "PCR_CV"):
            G = gains(f"e4_{ds}_kfold_spectral.json", m)
            sg = sum(1 for v in G.values() if v[2] < .05 and v[0] > 0); sl = sum(1 for v in G.values() if v[2] < .05 and v[0] < 0)
            mus = [v[0] for v in G.values()]
            put(f"real.{ds}.{m}.sig", [sg, sl, len(G)], f"{sg} gains, {sl} losses of {len(G)}", "K16", f"e4_{ds}_kfold_spectral.json")
            put(f"real.{ds}.{m}.range", [min(mus), max(mus)], f"{f(min(mus), 3, True)} to {f(max(mus), 3, True)}", "K16", f"e4_{ds}_kfold_spectral.json")
            put(f"real.{ds}.{m}.max_q", [max(mus), min(v[2] for v in G.values())], f"{f(max(mus), 4, True)}", "K16", f"e4_{ds}_kfold_spectral.json")
    for ds in ("rice", "mice"):
        b = np.mean([np.nanmean(T["r2"]["rrBLUP_CV"]) for T in J(f"e4_{ds}_kfold_spectral.json")])
        put(f"real.{ds}.kfold.rrBLUP_mean", b, f"{b:.3f}", "K19", f"e4_{ds}_kfold_spectral.json")
    c = np.mean([np.nanmean(T["r2"]["rrBLUP_CV"]) for T in J("e4_mice_cage_spectral.json")])
    put("real.mice.cage.rrBLUP_mean", c, f"{c:.3f}", "K19", "e4_mice_cage_spectral.json")
    G = gains("e4_rice_subpop_spectral.json", "safe_CV"); mus = [v[0] for v in G.values()]
    put("real.rice.subpop.safe.range", [min(mus), max(mus)], f"{f(min(mus), 3, True)} to {f(max(mus), 3, True)}", "K19", "e4_rice_subpop_spectral.json")
    rr = [np.nanmean(T["r2"]["rrBLUP_CV"]) for T in J("e4_rice_subpop_spectral.json")]
    put("real.rice.subpop.rrBLUP.range", [min(rr), max(rr)], f"{f(max(rr), 2)} to {f(min(rr), 2)}", "K19", "e4_rice_subpop_spectral.json")
    F = {(t["dataset"], t["trait"]): t["folds"] for t in J("e13_reml_foldiag.json")}
    for lab, sel in (("rejected", True), ("other", False)):
        v = [np.mean([x["p_chi2"] < .05 for x in fl]) for (d, t), fl in F.items() if d == "mice" and rejected[("mice", t)] == sel]
        put(f"foldiag.mice.{lab}", np.mean(v), f"{np.mean(v):.0%}", "K13", "e13_reml_foldiag.json")


# ------------------------------------------------------------ other model classes (K20-K22)
def other():
    for ds in ("rice", "wheat", "mice"):
        for m in ("BRR", "BayesC", "BayesR"):
            v = np.mean([np.nanmean(T["r2"][m]) for T in J(f"e4_{ds}_kfold_bayes_long.json")])
            put(f"bayes_long.{ds}.{m}.mean", v, f"{v:.3f}", "K21", f"e4_{ds}_kfold_bayes_long.json")
            z = np.array([abs(i["geweke"]) for T in J(f"e4_{ds}_kfold_bayes_long.json") for i in T["info"][m] if isinstance(i, dict) and i.get("geweke") is not None])
            z = z[np.isfinite(z)]; put(f"geweke_long.{ds}.{m}", float(np.mean(z > 1.96)), f"{np.mean(z > 1.96):.0%}", "K22", f"e4_{ds}_kfold_bayes_long.json")
            a = by_trait(f"e4_{ds}_kfold_bayes.json"); b = by_trait(f"e4_{ds}_kfold_bayes_long.json")
            dd = [np.nanmean(b[t]["r2"][m]) - np.nanmean(np.array(a[t]["r2"][m])[:len(b[t]["r2"][m])]) for t in b]
            put(f"chainlen.{ds}.{m}.mean_change", float(np.mean(dd)), f(float(np.mean(dd)), 3, True), "K22", "e4_*_bayes{,_long}.json")
            put(f"chainlen.{ds}.{m}.max_abs_change", float(np.max(np.abs(dd))), f"{np.max(np.abs(dd)):.3f}", "K22", "e4_*_bayes{,_long}.json")
    S = by_trait("e4_mice_kfold_spectral.json"); B = by_trait("e4_mice_kfold_bayes_long.json"); rows = []
    for t in B:
        base = np.array(S[t]["r2"]["rrBLUP_CV"])[:5]; dd = np.array(B[t]["r2"]["BayesR"])[:5] - base; n = B[t]["n"]
        rows.append((t, float(np.nanmean(dd)), corrected_t(dd, 0.8 * n, 0.2 * n)[1]))
    q = bh([r[2] for r in rows])
    for (t, mu, p), qq in zip(rows, q):
        if t in ("Biochem.HDL", "Biochem.Tot.Cholesterol"):
            put(f"bayesR_long.{t}", [mu, qq], f"{f(mu, 3, True)} (q = {qq:.3f})", "K20", "e4_mice_kfold_bayes_long.json")
    for scheme in ("kfold", "cage"):
        S = by_trait(f"e4_mice_{scheme}_spectral.json"); K = by_trait(f"e4_mice_{scheme}_kernel.json"); rows = []
        for t in K:
            b = np.array(S[t]["r2"]["rrBLUP_CV"]); v = np.array(K[t]["r2"]["KRR_RBF"]); L = min(len(b), len(v)); dd = v[:L] - b[:L]; n = K[t]["n"]
            rows.append((t, float(np.nanmean(dd)), corrected_t(dd, 0.8 * n, 0.2 * n)[1]))
        q = bh([r[2] for r in rows]); mu = np.mean([r[1] for r in rows])
        sg = sum(1 for r, qq in zip(rows, q) if qq < .05 and r[1] > 0)
        put(f"krr.mice.{scheme}.mean", mu, f(mu, 3, True), "K20", f"e4_mice_{scheme}_kernel.json")
        put(f"krr.mice.{scheme}.sig_gains", sg, str(sg), "K20", f"e4_mice_{scheme}_kernel.json")


# ------------------------------------------------------------ new panels, oracle gap, gamma ranges (K15, K17, K18)
def panels():
    for p in ("rdp1", "sorghum"):
        e = J(f"prereg_evaluation_{p}.json")
        put(f"prereg.{p}.H3", [e["H3"]["met"], e["H3"]["n"]] + list(e["H3"]["wilson95"]), f"{e['H3']['met']}/{e['H3']['n']}", "K18", f"prereg_evaluation_{p}.json")
        if isinstance(e["H1"], dict): put(f"prereg.{p}.H1.p", e["H1"]["p_one_sided"], f"{e['H1']['p_one_sided']:.2f}", "K18", f"prereg_evaluation_{p}.json")
        put(f"prereg.{p}.H2", [e["H2"]["spearman"], e["H2"]["p_one_sided"]], f"ρ = {e['H2']['spearman']:.3f}", "K18", f"prereg_evaluation_{p}.json")
        sig = sum(1 for r in e["rows"] for m in ("power_CV", "safe_CV", "KRR_RBF") if r["q_" + m] < .05)
        put(f"prereg.{p}.sig_any", sig, str(sig), "K16", f"prereg_evaluation_{p}.json")
    w = J("prereg_evaluation_rdp1.json")["H3"]["wilson95"]; put("prereg.rdp1.H3.wilson", w, f"{w[0]:.2f}–{w[1]:.2f}", "K18", "prereg_evaluation_rdp1.json")
    g = J("gap_real.json"); v = np.array([x["gap_r2"] for x in g])
    put("gap.traits", len(v), str(len(v)), "K17", "gap_real.json")
    put("gap.median", float(np.median(v)), f"{np.median(v):.4f}", "K17", "gap_real.json")
    put("gap.share_lt_0.001", float(np.mean(v < .001)), f"{np.mean(v < .001):.0%}", "K17", "gap_real.json")
    put("gap.share_lt_0.005", float(np.mean(v < .005)), f"{np.mean(v < .005):.0%}", "K17", "gap_real.json")
    put("gap.max", float(v.max()), f"{v.max():.3f}", "K17", "gap_real.json")
    for p in ("rice", "wheat", "mice", "rdp1", "sorghum"):
        gs = [x["gamma"] for x in g if x["dataset"] == p]
        put(f"gamma.{p}.range", [min(gs), max(gs)], f"{min(gs):.2f}–{max(gs):.2f}", "K15", "gap_real.json")
    put("gamma.n_ge_2", sum(x["gamma"] >= 2 for x in g), str(sum(x["gamma"] >= 2 for x in g)), "K15", "gap_real.json")


# ------------------------------------------------------------ practical, runtime, ablation, stability, old benchmark
def practical():
    a = J("e8_scaling_run1.json")["results"]; b = J("e8_scaling_run2.json")["results"]
    for r in b:
        for k in ("eigh_s", "rrBLUP_CV_s", "power_CV_s", "safe_CV_s", "grm_GB"):
            if k in r: put(f"e8.n{r['n']}.{k}", r[k], f"{r[k]:.3g}", "K24", "e8_scaling_run2.json")
    ratio = [r["power_CV_s"] / r["rrBLUP_CV_s"] for r in b if r["n"] >= 1000]
    put("e8.power_over_rrblup", [min(ratio), max(ratio)], f"{min(ratio):.2f}–{max(ratio):.2f}", "K24", "e8_scaling_run2.json")
    agree = max(abs(x[k] - y[k]) / y[k] for x, y in zip(a, b) if x["n"] >= 1000 for k in x if k.endswith("_s"))
    put("e8.run_agreement", agree, f"{int(np.ceil(100 * agree))}%", "K24", "e8_scaling_run1/2.json")   # an upper bound: "within 13%"
    # R2.8: cost of the safe selector relative to tuned rrBLUP, and growth of the eigendecomposition
    sr = [r["safe_CV_s"] / r["rrBLUP_CV_s"] for r in b if "safe_CV_s" in r]
    put("e8.safe_over_rrblup", [min(sr), max(sr)], f"{min(sr):.1f} to {max(sr):.1f}", "K24", "e8_scaling_run2.json")
    e = {r["n"]: r["eigh_s"] for r in b}
    put("e8.eigh_ratio_4000_8000", e[8000] / e[4000], f"{e[8000] / e[4000]:.1f}", "K24", "e8_scaling_run2.json")
    put("e8.markers", 10000, "10,000", "K24", "e7_e8_e10.py (scaling)")
    put("e8.cpu_threads", 12, "12", "K24", "e8_scaling_run2.json")
    E = J("e10_ablation.json")
    for v in ("original", "bounded_only", "guard_only", "both"):
        r2 = np.mean([x["r2"] for x in E if x["dataset"] == "wheat" and x["family"] == "smooth" and x["variant"] == v])
        put(f"e10.wheat.smooth.{v}", r2, f(r2, 3), "K7", "e10_ablation.json")
    # R5.3: per panel, family and variant, the range over traits of the trait-mean held-out R2 (10 folds per trait)
    for dset in ("wheat", "rice"):
        for fam in ("grouped", "smooth"):
            for v in ("original", "bounded_only", "guard_only", "both"):
                tm = [np.mean([x["r2"] for x in E if (x["dataset"], x["family"], x["variant"], x["trait"]) == (dset, fam, v, t)])
                      for t in sorted({x["trait"] for x in E if x["dataset"] == dset})]
                put(f"e10.{dset}.{fam}.{v}.range", [min(tm), max(tm)], f"{f(min(tm), 2)} to {f(max(tm), 2)}", "K7", "e10_ablation.json")
    lo = [np.mean([x["loo"] for x in E if (x["dataset"], x["family"], x["variant"], x["trait"]) == ("wheat", "smooth", "original", t)])
          for t in sorted({x["trait"] for x in E if x["dataset"] == "wheat"})]
    put("e10.wheat.smooth.original.loo_mse", [min(lo), max(lo)], f"{min(lo):.3f} to {max(lo):.3f}", "K7", "e10_ablation.json")
    put("e10.wheat.smooth.original.loo_r2_min", 1 - max(lo), f"{1 - max(lo):.2f}", "K7", "e10_ablation.json")
    spread = max(np.ptp([np.mean([x["r2"] for x in E if (x["dataset"], x["family"], x["variant"], x["trait"]) == ("rice", fam, v, t)])
                         for v in ("original", "bounded_only", "guard_only", "both")])
                 for fam in ("grouped", "smooth") for t in sorted({x["trait"] for x in E if x["dataset"] == "rice"}))
    put("e10.rice.max_spread", spread, f"{spread:.3f}", "K7", "e10_ablation.json")
    put("e10.folds", len({(x["seed"], x["fold"]) for x in E}), str(len({(x["seed"], x["fold"]) for x in E})), "K7", "e10_ablation.json")
    if os.path.exists(os.path.join(RES, "e7_reml_boot.json")):
        S = J("e7_reml_boot.json")
        put("e7.traits", len(S), str(len(S)), "K25", "e7_reml_boot.json")
        w = np.array([x["ci"][1] - x["ci"][0] for x in S])
        put("e7.ci_width_median", float(np.median(w)), f"{np.median(w):.2f}", "K25", "e7_reml_boot.json")
        put("e7.n_ci_at_upper_bound", int(sum(x["ci"][1] >= 5.99 for x in S)), str(int(sum(x["ci"][1] >= 5.99 for x in S))), "K25", "e7_reml_boot.json")
        put("e7.n_ci_reach_2", int(sum(x["ci"][1] >= 2 for x in S)), str(int(sum(x["ci"][1] >= 2 for x in S))), "K15", "e7_reml_boot.json")
        put("e7.n_ci_exclude_1", int(sum(x["ci"][1] < 1 or x["ci"][0] > 1 for x in S)), str(int(sum(x["ci"][1] < 1 or x["ci"][0] > 1 for x in S))), "K25", "e7_reml_boot.json")
        for x in S: put(f"e7.{x['dataset']}.{x['trait']}", [x["gamma"]] + x["ci"], f"{x['gamma']:.2f} ({x['ci'][0]:.2f}–{x['ci'][1]:.2f})", "K25", "e7_reml_boot.json")
    d = J("fixed_synth.json"); fams = ("grouped", "smooth", "monotone", "spline")
    dl = np.array([[x["r2"][m] - x["r2"]["scalar"] for m in fams] for x in d])
    put("b8.family_means", dl.mean(0).tolist(), " / ".join(f(x, 4, True) for x in dl.mean(0)), "B8", "fixed_synth.json")
    put("b8.best_of_4", float(dl.max(1).mean()), f(float(dl.max(1).mean()), 4, True), "B8", "fixed_synth.json")
    gp = np.array([x["gap"] for x in d]); cr = np.corrcoef(gp, dl.max(1))[0, 1]
    cells = {}
    for x, row in zip(d, dl):
        if x["exp"] == "grid": cells.setdefault((x["n"], x["decay"], x["par"], x["align"], x["snr"]), []).append(row)
    cm = np.array([np.mean(v, 0) for v in cells.values()])
    put("b8.grid_cells", len(cm), str(len(cm)), "B8", "fixed_synth.json")
    put("b8.share_cell_family_negative", float(np.mean(cm < 0)), f"{np.mean(cm < 0):.0%}", "B8", "fixed_synth.json")
    put("b8.share_cell_best_negative", float(np.mean(cm.max(1) < 0)), f"{np.mean(cm.max(1) < 0):.0%}", "B8", "fixed_synth.json")
    put("b8.worst_cell_family", float(cm.min()), f(float(cm.min()), 3), "B8", "fixed_synth.json")
    put("b8.share_pairs_loss_gt_0.003", float(np.mean(dl < -0.003)), f"{np.mean(dl < -0.003):.1%}", "B8", "fixed_synth.json")
    put("b8.configs", len(d), str(len(d)), "B8", "fixed_synth.json")
    put("b8.gap_gain_corr", float(cr), f(cr, 2), "B8", "fixed_synth.json")


# ------------------------------------------------------------ R5.6: small n (E1 by training size) and close eigenvalues (E14)
def small_n_close_eigs():
    R = J("e1_regimes.json"); src = "e1_regimes.json"
    d = lambda r, m: r["r2"][m] - r["r2"]["rrBLUP_CV"]
    ns = sorted({r["n"] for r in R}); cells = {}
    for r in R: cells.setdefault((r["scn"], r["n"], r["h2"]), []).append(r)
    for n in ns:
        lag = np.mean([np.mean([r["r2"]["oracle"] - r["r2"]["safe_CV"] for r in v]) for k, v in cells.items() if k[1] == n])
        put(f"e1.oracle_minus_safe.n{n}", float(lag), f(float(lag), 3), "K6", src)
        for m, tag in (("safe_CV", "safe"), ("power_CV", "power")):
            cm = [np.mean([d(r, m) for r in v]) for k, v in cells.items() if k[1] == n]
            put(f"e1.{tag}.cells_loss_gt_0.003.n{n}", int(sum(x < -0.003 for x in cm)), str(int(sum(x < -0.003 for x in cm))), "K6", src)
        v = np.array([d(r, "safe_CV") for r in R if r["n"] == n])
        put(f"e1.safe.share_ds_loss_gt_0.01.n{n}", float(np.mean(v < -0.01)), f"{np.mean(v < -0.01):.1%}", "K6", src)
    put("e1.cells_per_n", len([k for k in cells if k[1] == ns[0]]), str(len([k for k in cells if k[1] == ns[0]])), "K6", src)
    src = "e14_close_eigs.json"
    for x in J(src):
        p = x["panel"]
        put(f"e14.{p}.close", [x["n_close"], x["n_top"]], f"{x['n_close']} of {x['n_top']}", "K8", src)
        put(f"e14.{p}.cos_close", x["eigvec_cos_close"], "—" if x["eigvec_cos_close"] is None else f"{x['eigvec_cos_close']:.2f}", "K8", src)
        put(f"e14.{p}.cos_far", x["eigvec_cos_far"], f"{x['eigvec_cos_far']:.2f}", "K8", src)
        put(f"e14.{p}.ratio_power", x["rel_change_power"] / x["rel_change_G"], f"{x['rel_change_power'] / x['rel_change_G']:.1f}", "K8", src)
        put(f"e14.{p}.ratio_grouped", x["rel_change_grouped"] / x["rel_change_G"], f"{x['rel_change_grouped'] / x['rel_change_G']:.1f}", "K8", src)
        put(f"e14.{p}.lip", [x["max_lip_ratio"], x["L_h"]], f"{x['max_lip_ratio']:.2f} ≤ {x['L_h']:.2f}", "K8", src)
    put("e14.bound_holds_all", all(x["bound_holds"] for x in J(src)), "holds in every replicate", "K8", src)


# ------------------------------------------------------------ R5.4: fixed effects not removed (E15, replaces the pilot behind K14)
def adjustment():
    src = "e15_adjust.json"
    if not os.path.exists(os.path.join(RES, src)): return
    E = J(src)
    gain = {lab: np.array([np.mean(r[lab]["r2"]["power_CV"]) - np.mean(r[lab]["r2"]["rrBLUP_CV"]) for r in E]) for lab in ("adjusted", "unadjusted")}
    rej = {lab: int(sum(r[lab]["p_chi2"] < .05 for r in E)) for lab in ("adjusted", "unadjusted")}
    put("e15.traits", len(E), str(len(E)), "K14", src)
    for lab in ("adjusted", "unadjusted"):
        put(f"e15.{lab}.rejected", rej[lab], f"{rej[lab]} of {len(E)}", "K14", src)
        gm = float(gain[lab].mean()); put(f"e15.{lab}.gain_mean", gm, "0.0000" if abs(gm) < 5e-5 else f(gm, 4, True), "K14", src)
        k = int(np.sum(gain[lab] > 0.005)); put(f"e15.{lab}.gain_gt_0.005", k, f"{k} of {len(E)}", "K14", src)
    e = next(r for r in E if r["trait"] == "Obesity.EndNormalBW")
    for lab in ("adjusted", "unadjusted"):
        put(f"e15.EndNormalBW.{lab}.gamma", e[lab]["gamma"], f"{e[lab]['gamma']:.2f}", "K14", src)
        g = np.mean(e[lab]["r2"]["power_CV"]) - np.mean(e[lab]["r2"]["rrBLUP_CV"])
        put(f"e15.EndNormalBW.{lab}.gain", float(g), f(float(g), 4, True), "K14", src)


# ------------------------------------------------------------ R2.2: panel sizes, pre-registered categories, rice-RDP1 overlap check
def panels_r22():
    P = J("panel_info.json"); src = "panel_info.json"
    for p, d in P.items():
        put(f"panel.{p}.n", d["n"], f"{d['n']:,}", "K16", src)
        put(f"panel.{p}.markers", d["markers"], f"{d['markers']:,}", "K16", src)
        put(f"panel.{p}.traits", d["traits"], str(d["traits"]), "K16", src)
        put(f"panel.{p}.n_range", [d["n_min"], d["n_max"]], f"{d['n_min']:,} to {d['n_max']:,}", "K16", src)
        r = d["markers"] / d["n"]; put(f"panel.{p}.p_over_n", r, f"{r:.1f}" if r < 10 else f"{r:.0f}", "K16", src)
    put("panel.traits_total", sum(d["traits"] for d in P.values()), str(sum(d["traits"] for d in P.values())), "K16", src)
    for p in ("rdp1", "sorghum"):
        rows = J(f"prereg_evaluation_{p}.json")["rows"]; src = f"prereg_evaluation_{p}.json"
        for c in ("R+", "R-", "N"):
            put(f"prereg.{p}.cat.{c}", sum(r["category"] == c for r in rows), str(sum(r["category"] == c for r in rows)), "K18", src)
        for m in ("power_CV", "safe_CV"):
            v = [r[m] for r in rows]; put(f"prereg.{p}.{m}.range", [min(v), max(v)], f"{f(min(v), 3, True)} to {f(max(v), 3, True)}", "K16", src)
    rows = J("prereg_evaluation_rdp1.json")["rows"] + J("prereg_evaluation_sorghum.json")["rows"]
    v = [r["safe_CV"] for r in rows]; put("prereg.both.safe_CV.range", [min(v), max(v)], f"{f(min(v), 3, True)} to {f(max(v), 3, True)}", "K16", "prereg_evaluation_*.json")
    o = J("rice_rdp1_overlap.json"); src = "rice_rdp1_overlap.json"
    b = [x["concordance"] for x in o["best_match"]]
    put("overlap.snps", o["shared_snps"], f"{o['shared_snps']:,}", "K18", src)
    put("overlap.rdp1_nn_median", o["rdp1_nearest_neighbour_median"], f"{o['rdp1_nearest_neighbour_median']:.2f}", "K18", src)
    put("overlap.best.range", [min(b), max(b)], f"{min(b):.2f} to {max(b):.2f}", "K18", src)
    put("overlap.n_ge_0.95", o["n_rice_with_match_ge_0_95"], str(o["n_rice_with_match_ge_0_95"]), "K18", src)


# ------------------------------------------------------------ R2.7 / R5.9: every baseline against rrBLUP on identical folds
def baselines():
    """Per panel and method: mean held-out R2, mean difference from rrBLUP_CV over traits, and the number of traits
    with a BH-significant gain or loss (corrected resampled t-test on the paired folds, BH within panel and method)."""
    for ds in ("rice", "wheat", "mice"):
        S = by_trait(f"e4_{ds}_kfold_spectral.json")
        src = {m: (f"e4_{ds}_kfold_spectral.json", None) for m in ("power_CV", "safe_CV", "PCR_CV")}
        src.update({m: (f"e4_{ds}_kfold_kernel.json", None) for m in ("KRR_RBF", "DNNGP")})
        src.update({m: (f"e4_{ds}_kfold_bayes_long.json", None) for m in ("BRR", "BayesC", "BayesR")})
        L = {t["trait"]: [x["r2"] for x in t["folds"]] for t in J(f"e11_ldak_{ds}.json")}
        E = {t["trait"]: [x["r2"] for x in t["folds"]] for t in J(f"e11_epi_{ds}.json")}
        rr = np.mean([np.nanmean(S[t]["r2"]["rrBLUP_CV"]) for t in S])
        put(f"base.{ds}.rrBLUP.mean", rr, f"{rr:.3f}", "K21", f"e4_{ds}_kfold_spectral.json")
        for m in list(src) + ["LDAK", "EPI"]:
            rows = []
            for t in S:
                b = np.array(S[t]["r2"]["rrBLUP_CV"])
                v = np.array(L[t]) if m == "LDAK" else np.array(E[t]) if m == "EPI" else np.array(by_trait(src[m][0])[t]["r2"][m])
                k = min(len(b), len(v)); dd = v[:k] - b[:k]; n = S[t]["n"]
                rows.append((float(np.nanmean(v[:k])), float(np.nanmean(dd)), corrected_t(dd, 0.8 * n, 0.2 * n)[1]))
            q = bh([r[2] for r in rows])
            g = sum(1 for r, qq in zip(rows, q) if qq < .05 and r[1] > 0); l = sum(1 for r, qq in zip(rows, q) if qq < .05 and r[1] < 0)
            mu = float(np.mean([r[1] for r in rows])); r2 = float(np.mean([r[0] for r in rows]))
            fsrc = f"e11_ldak_{ds}.json" if m == "LDAK" else f"e11_epi_{ds}.json" if m == "EPI" else src[m][0]
            put(f"base.{ds}.{m}.mean", r2, f"{r2:.3f}", "K21", fsrc)
            put(f"base.{ds}.{m}.delta", mu, f(mu, 3, True), "K21", fsrc)
            put(f"base.{ds}.{m}.sig", [g, l], f"{g}/{l}", "K21", fsrc)


# ------------------------------------------------------------ R2.10, R5.7, R2.3: practical effect, rice in detail, structure
def practical_rice_structure():
    from scipy import stats as st
    # practical effect on selection (E9): top-10% overlap with rrBLUP and the accuracy (response) ratio r_safe / r_rrBLUP
    acc = {}
    for ds in ("rice", "wheat", "mice"):
        if any("y" not in rec for T in J(f"e4_{ds}_kfold_spectral.json") for rec in T["preds"]):
            # the public copy of the rice results omits the phenotypes (available on request): keep the stored values
            for k, v in json.load(open(os.path.join(RES, "numbers.json"), encoding="utf-8")).items():
                if k.startswith(f"e9.{ds}."): NUM[k] = v
            continue
        ov, rr = {}, {}
        for T in J(f"e4_{ds}_kfold_spectral.json"):
            o, r = [], []
            for rec in T["preds"]:
                y = np.array(rec["y"]); a = np.array(rec["p"]["rrBLUP_CV"]); b = np.array(rec["p"]["safe_CV"]); q = max(1, int(round(0.1 * len(y))))
                o.append(len(set(np.argsort(-a)[:q]) & set(np.argsort(-b)[:q])) / q)
                ca, cb = np.corrcoef(y, a)[0, 1], np.corrcoef(y, b)[0, 1]
                if ca > 0: r.append(cb / ca)
            ov[T["trait"]] = np.mean(o); rr[T["trait"]] = np.mean(r)
            if ds == "mice": acc[T["trait"]] = float(np.mean(T["r2"]["rrBLUP_CV"]))
        if ds == "mice":
            low = [t for t in ov if ov[t] < 0.9]
            put("e9.mice.n_overlap_below_0.9", len(low), str(len(low)), "K23", "e4_mice_kfold_spectral.json")
            put("e9.mice.low_overlap_max_r2", max(acc[t] for t in low), f"{max(acc[t] for t in low):.3f}", "K23", "e4_mice_kfold_spectral.json")
        put(f"e9.{ds}.overlap10", [min(ov.values()), max(ov.values())], f"{min(ov.values()):.2f} to {max(ov.values()):.2f}", "K23", f"e4_{ds}_kfold_spectral.json")
        put(f"e9.{ds}.response_ratio", [min(rr.values()), max(rr.values())], f"{min(rr.values()):.3f} to {max(rr.values()):.3f}", "K23", f"e4_{ds}_kfold_spectral.json")
    # the largest simulated gain in selection terms: response is proportional to accuracy r = sqrt(R2)
    R = J("e1_regimes.json"); rat = []
    for scn in ("g2", "g3"):
        for n in (400, 800, 1600):
            v = [r for r in R if r["scn"] == scn and r["n"] == n]
            a = np.mean([r["r2"]["rrBLUP_CV"] for r in v]); b = np.mean([r["r2"]["safe_CV"] for r in v])
            rat.append(np.sqrt(b) / np.sqrt(a))
    put("e1.response_gain_pct", [min(rat), max(rat)], f"{100 * (min(rat) - 1):.1f}% to {100 * (max(rat) - 1):.1f}%", "K23", "e1_regimes.json")
    # rice in detail: per trait, rrBLUP held-out R2 and the safe selector's difference, with corrected 95% CIs
    S = by_trait("e4_rice_kfold_spectral.json")
    names = {"ht_cm_Yan_et_al_2007": "height", "Brown_Grain_Length_mm": "length", "Brown_Grain_Width_mm": "width", "Brown_Grain_L/Wratio": "ratio"}
    for t, T in S.items():
        n = T["n"]; b = np.array(T["r2"]["rrBLUP_CV"]); J_ = len(b)
        se = np.sqrt((1 / J_ + 0.25) * b.var(ddof=1)); tq = st.t.ppf(0.975, J_ - 1)
        put(f"rice.{names[t]}.rrBLUP", [b.mean(), b.mean() - tq * se, b.mean() + tq * se], f"{b.mean():.3f} ({f(b.mean() - tq * se, 3)} to {f(b.mean() + tq * se, 3)})", "K16", "e4_rice_kfold_spectral.json")
        put(f"rice.{names[t]}.fold_sd", float(b.std(ddof=1)), f"{b.std(ddof=1):.3f}", "K16", "e4_rice_kfold_spectral.json")
        for m in ("safe_CV", "power_CV"):
            d = np.array(T["r2"][m]) - b; se = np.sqrt((1 / J_ + 0.25) * d.var(ddof=1))
            put(f"rice.{names[t]}.{m}", [d.mean(), d.mean() - tq * se, d.mean() + tq * se], f"{f(d.mean(), 3, True)} ({f(d.mean() - tq * se, 3, True)} to {f(d.mean() + tq * se, 3, True)})", "K16", "e4_rice_kfold_spectral.json")
    sd = [np.array(T["r2"]["rrBLUP_CV"]).std(ddof=1) for T in S.values()]
    put("rice.fold_sd.range", [min(sd), max(sd)], f"{min(sd):.3f} to {max(sd):.3f}", "K16", "e4_rice_kfold_spectral.json")
    # rice structure: F statistic of each leading eigenvector across the four k-means subpopulations (datasets.rice_clusters)
    import datasets
    if not os.path.exists(os.path.join(datasets.DATA, "rice_GRM.npy")):
        # the rice GRM is not public: use the F statistics stored by the stability run (same clusters, same GRM)
        for k, F in enumerate(J("e7_stability.json")["rice_F"][:6]):
            put(f"rice.F.pc{k + 1}", float(F), f"{F:.1f}", "K19", "e7_stability.json (rice_F)")
        put("rice.subpops", 4, "4", "K19", "datasets.rice_clusters")
        return
    D = datasets.rice(markers=False); G = D["G"]; cl = datasets.rice_clusters(G)
    ev, U = np.linalg.eigh(G); U = U[:, ::-1]
    for k in range(6):
        F = st.f_oneway(*[U[cl == c, k] for c in np.unique(cl)]).statistic
        put(f"rice.F.pc{k + 1}", float(F), f"{F:.1f}", "K19", "rice_GRM.npy + datasets.rice_clusters")
    put("rice.subpops", int(len(np.unique(cl))), str(len(np.unique(cl))), "K19", "datasets.rice_clusters")


def epistatic_mice():
    # E11: epistatic (Hadamard GRM) kernel on mice, paired with rrBLUP on the same seed and fold, as in Fig. 9a
    def r2(rec, m):
        y = np.array(rec["y"]); p = np.array(rec["p"][m]); return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    for scheme, fn in (("kfold", "e11_epi_mice.json"), ("cage", "e11_epi_mice_cage.json")):
        S = by_trait(f"e4_mice_{scheme}_spectral.json"); rows = []
        for T in J(fn):
            base = {(r["seed"], r["fold"]): r2(r, "rrBLUP_CV") for r in S[T["trait"]]["preds"]}
            dd = np.array([x["r2"] - base[(x["seed"], x["fold"])] for x in T["folds"] if (x["seed"], x["fold"]) in base])
            n = S[T["trait"]]["n"]; rows.append((float(dd.mean()), corrected_t(dd, 0.8 * n, 0.2 * n)[1]))
        q = bh([p for _, p in rows]); mu = float(np.mean([m for m, _ in rows]))
        put(f"epi.mice.{scheme}.mean", mu, f(mu, 3, True), "K20", fn)
        put(f"epi.mice.{scheme}.sig_gains", sum(1 for (m, _), qq in zip(rows, q) if qq < .05 and m > 0),
            str(sum(1 for (m, _), qq in zip(rows, q) if qq < .05 and m > 0)), "K20", fn)


def stability_power():
    # R5.10: how stable the tuned power a and the safe selector's choice are across the 25 folds, and how much the
    # predictions of the power filter and rrBLUP differ (Pearson correlation of held-out predictions)
    med = {}
    for ds in ("rice", "wheat", "mice", "rdp1", "sorghum"):
        D = J(f"e4_{ds}_kfold_spectral.json"); iqr, pick, cor = [], [], []
        for T in D:
            a = np.array([i.get("a") for i in T["info"]["power_CV"]], float); q = np.percentile(a, [25, 75]); iqr.append(q[1] - q[0])
            pick += [i.get("pick") == "power_CV" for i in T["info"]["safe_CV"]]
            cor += [np.corrcoef(r["p"]["rrBLUP_CV"], r["p"]["power_CV"])[0, 1] for r in T["preds"]]
        put(f"stab.{ds}.a_iqr", float(np.median(iqr)), f"{np.median(iqr):.2f}", "K25", f"e4_{ds}_kfold_spectral.json")
        put(f"stab.{ds}.pick_power", float(np.mean(pick)), f"{np.mean(pick):.0%}", "K25", f"e4_{ds}_kfold_spectral.json")
        med[ds] = float(np.median(cor))
    put("stab.corr_median.range", [min(med.values()), max(med.values())], f"{min(med.values()):.3f} to {max(med.values()):.3f}", "K25", "e4_*_kfold_spectral.json")


def robustness_checks():
    """Robustness of the main results: all eight simulated architectures, recovery of gamma and gains on the real
    GRMs (E3), equivalence tests and a pooled estimate over the 61 traits, the oracle gap at the ends of the gamma-hat
    interval, and correlations under leave-one-subpopulation-out CV."""
    from scipy import stats as st
    # simulation: safe selector and oracle gain over rrBLUP_CV by architecture and n (E1)
    R = J("e1_regimes.json"); src = "e1_regimes.json"
    for scn in ("g0.5", "g0.75", "g1", "g1.25", "g1.5", "g2", "g3", "spiked"):
        for n in (400, 800, 1600):
            v = [r for r in R if r["scn"] == scn and r["n"] == n]
            d = np.array([r["r2"]["safe_CV"] - r["r2"]["rrBLUP_CV"] for r in v]); o = np.array([r["r2"]["oracle"] - r["r2"]["rrBLUP_CV"] for r in v])
            h = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
            put(f"sim.{scn}.{n}.safe", [float(d.mean()), float(d.mean() - h), float(d.mean() + h)], f(float(d.mean()), 3, True), "K3", src)
            put(f"sim.{scn}.{n}.oracle", float(o.mean()), f(float(o.mean()), 3, True), "K3", src)
            put(f"sim.{scn}.{n}.ci_excl0", bool(d.mean() - h > 0 or d.mean() + h < 0), "yes" if (d.mean() - h > 0 or d.mean() + h < 0) else "no", "K3", src)
    share = []
    for scn in ("g0.5", "g2", "g3"):
        for n in (400, 800, 1600):
            v = [r for r in R if r["scn"] == scn and r["n"] == n]
            share.append(np.mean([r["r2"]["safe_CV"] - r["r2"]["rrBLUP_CV"] for r in v]) / np.mean([r["r2"]["oracle"] - r["r2"]["rrBLUP_CV"] for r in v]))
    put("sim.recovered_share.range", [min(share), max(share)], f"{min(share):.0%} to {max(share):.0%}", "K3", src)
    rat = []
    for n in (400, 800, 1600):
        v = [r for r in R if r["scn"] == "g0.5" and r["n"] == n]
        rat.append(np.sqrt(np.mean([r["r2"]["safe_CV"] for r in v])) / np.sqrt(np.mean([r["r2"]["rrBLUP_CV"] for r in v])))
    put("sim.g0.5.response_gain_pct", [min(rat), max(rat)], f"{100 * (min(rat) - 1):.1f}% to {100 * (max(rat) - 1):.1f}%", "K23", src)
    # E3: semi-synthetic phenotypes on the real GRMs (gamma defined on the panel's own eigenvalues)
    E = J("e3_regimes.json"); src = "e3_regimes.json"
    for ds in ("rice", "wheat", "mice"):
        for scn in ("g0.5", "g1", "g2", "g3", "structure"):
            v = [x for x in E if x["data"] == ds and x["scn"] == scn]
            d = np.array([x["r2"]["safe_CV"] - x["r2"]["rrBLUP_CV"] for x in v]); h = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
            T = [t for t in J("e3_reml_test.json") if t["data"] == ds and t["scn"] == scn]   # REML test of Section 3.8
            gh = np.array([t["gamma"] for t in T]); rj = np.mean([t["p_chi2"] < .05 for t in T])
            put(f"e3.{ds}.{scn}.safe", [float(d.mean()), float(d.mean() - h), float(d.mean() + h)], f(float(d.mean()), 4, True), "K3", src)
            put(f"e3.{ds}.{scn}.gamma_med", float(np.median(gh)), f"{np.median(gh):.2f}", "K3", src)
            put(f"e3.{ds}.{scn}.reject", float(rj), f"{rj:.2f}", "K9", src)
            put(f"e3.{ds}.{scn}.reject_pct", float(rj), f"{rj:.0%}", "K9", src)
    put("e3.reps", len([x for x in E if x["data"] == "mice" and x["scn"] == "g1"]), str(len([x for x in E if x["data"] == "mice" and x["scn"] == "g1"])), "K3", src)
    # equivalence (two one-sided tests, margin 0.01 and 0.02 in R2) and an equal-weight average over the 61 traits
    for m, lab in (("safe_CV", "safe"), ("power_CV", "power")):
        rows = []
        for ds in ("rice", "wheat", "mice", "rdp1", "sorghum"):
            for t, T in by_trait(f"e4_{ds}_kfold_spectral.json").items():
                b = np.array(T["r2"]["rrBLUP_CV"]); d = np.array(T["r2"][m]) - b; J_ = len(d)
                se = max(np.sqrt((1 / J_ + 0.25) * d.var(ddof=1)), 1e-6); tq = st.t.ppf(0.95, J_ - 1)
                rows.append((d.mean(), se, d.mean() - tq * se, d.mean() + tq * se, ds))
        for mg in (0.005, 0.01, 0.02):
            k = sum(1 for r in rows if r[2] > -mg and r[3] < mg)
            put(f"equiv.{lab}.{mg}", k, str(k), "K16", "e4_*_kfold_spectral.json")
        # average over the 61 traits, each trait weighted equally; its standard error treats traits as independent
        mu = np.array([r[0] for r in rows]); se = np.array([r[1] for r in rows])
        re = float(mu.mean()); h = 1.96 * float(np.sqrt((se ** 2).sum())) / len(mu)
        put(f"pooled.{lab}", [float(re), float(re - h), float(re + h)], f"{f(float(re), 4, True)} (95% CI {f(float(re - h), 4, True)} to {f(float(re + h), 4, True)})", "K16", "e4_*_kfold_spectral.json")
        # the same average allowing full positive correlation between traits of one panel (panel SEs add linearly)
        pan = [se[[i for i, r in enumerate(rows) if r[4] == ds]].sum() for ds in ("rice", "wheat", "mice", "rdp1", "sorghum")]
        hd = 1.96 * float(np.sqrt(np.sum(np.square(pan)))) / len(mu)
        put(f"pooled_dep.{lab}", [float(re - hd), float(re + hd)], f"{f(float(re - hd), 4, True)} to {f(float(re + hd), 4, True)}", "K16", "e4_*_kfold_spectral.json")
    # oracle gap at the ends of the bootstrap interval of gamma-hat
    C = J("gap_ci.json"); up = np.array([max(r["gap_lo"], r["gap_hi"]) for r in C])
    put("gapci.median", float(np.median(up)), f"{np.median(up):.4f}", "K17", "gap_ci.json")
    put("gapci.below_0.005", float(np.mean(up < 0.005)), f"{np.mean(up < 0.005):.0%}", "K17", "gap_ci.json")
    put("gapci.below_0.01", float(np.mean(up < 0.01)), f"{np.mean(up < 0.01):.0%}", "K17", "gap_ci.json")
    put("gapci.max", float(up.max()), f"{up.max():.3f}", "K17", "gap_ci.json")
    # accuracy of rrBLUP at gamma = 0.5 (n = 400 to 1,600, all h2), to read the relative response gain
    a = [np.mean([r["r2"]["rrBLUP_CV"] for r in R if r["scn"] == "g0.5" and r["n"] == n and r["h2"] == h])
         for n in (400, 800, 1600) for h in (0.3, 0.5, 0.7)]
    put("sim.g0.5.rr_r2.range", [min(a), max(a)], f"{min(a):.3f} to {max(a):.2f}", "K23", "e1_regimes.json")
    # pre-registration: a rule predicting no material difference for every trait, and the kind of misses
    for p in ("rdp1", "sorghum"):
        E_ = J(f"prereg_evaluation_{p}.json")["rows"]
        nr = sum(1 for r in E_ if abs(r["power_CV"]) < 0.005 and not ((r["q_power_CV"] < .05 and r["power_CV"] > 0) or (r["q_safe_CV"] < .05 and r["safe_CV"] > 0)))
        put(f"prereg.always_n.{p}", nr, str(nr), "K18", f"prereg_evaluation_{p}.json")
        ml = sum(1 for r in E_ if not r["prediction_met"] and r["category"] == "N" and r["power_CV"] <= -0.005)
        put(f"prereg.miss_nloss.{p}", ml, str(ml), "K18", f"prereg_evaluation_{p}.json")
    # leave-one-subpopulation-out: correlation between prediction and phenotype
    S = J("e4_rice_subpop_spectral.json")
    r0 = [float(np.nanmean(np.array(T["r"]["rrBLUP_CV"], float))) for T in S]
    dr = [float(np.nanmean(np.array(T["r"]["safe_CV"], float) - np.array(T["r"]["rrBLUP_CV"], float))) for T in S]
    put("subpop.r.rrBLUP.range", [min(r0), max(r0)], f"{min(r0):.2f} to {max(r0):.2f}", "K19", "e4_rice_subpop_spectral.json")
    put("subpop.r.safe_diff.range", [min(dr), max(dr)], f"{f(min(dr), 3, True)} to {f(max(dr), 3, True)}", "K19", "e4_rice_subpop_spectral.json")


def audit_checks():
    """Numbers added by the pre-resubmission audit (2026-10-31): largest simulated gains, screening by architecture,
    replicates at n = 1,600, the plug-in gamma-BLUP on real data, the mouse traits with gamma-hat below 0.5, the rice
    gap at the ends of the interval, leave-one-subpopulation-out groups, and the real-GRM test counts."""
    R = J("e1_regimes.json"); src = "e1_regimes.json"
    cells = lambda m: [np.mean([r["r2"][m] - r["r2"]["rrBLUP_CV"] for r in R if r["scn"] == s_ and r["n"] == n])
                       for s_ in {r["scn"] for r in R} for n in {r["n"] for r in R}]
    for m, lab in (("safe_CV", "safe"), ("power_CV", "power"), ("gammaBLUP_ML", "gblup")):
        put(f"sim.max_gain.{lab}", float(max(cells(m))), f"{max(cells(m)):.4f}", "K3", src)
    mx = max(max(cells("safe_CV")), max(cells("power_CV")), max(cells("gammaBLUP_ML")))
    put("sim.max_gain.any", float(mx), "0.01" if mx <= 0.01 else f"{mx:.3f}", "K3", src)
    for s_ in ("g0.5", "g0.75", "g1", "g1.25", "g1.5", "g2", "g3", "spiked"):
        put(f"sim.n1600.runs.{s_}", sum(1 for r in R if r["scn"] == s_ and r["n"] == 1600), str(sum(1 for r in R if r["scn"] == s_ and r["n"] == 1600)), "K10", src)
    # screening accuracy for each architecture (chi2 p-value, gain = any positive held-out difference)
    T = J("e1_reml_test.json"); E = {(r["scn"], r["n"], r["h2"], r["rep"], r["seed"]): r for r in R}
    rec = [(t["scn"], t["p_chi2"] < .05, E[(t["scn"], t["n"], t["h2"], t["rep"], t["seed"])]["r2"]["safe_CV"]
            - E[(t["scn"], t["n"], t["h2"], t["rep"], t["seed"])]["r2"]["rrBLUP_CV"] > 0) for t in T]
    for s_ in ("g0.5", "g0.75", "g1", "g1.25", "g1.5", "g2", "g3", "spiked"):
        rj = np.array([x[1] for x in rec if x[0] == s_]); ps = np.array([x[2] for x in rec if x[0] == s_])
        st_ = dict(runs=len(rj), reject=rj.mean(), gain=ps.mean(), sens=rj[ps].mean(), spec=(~rj[~ps]).mean(),
                   ppv=ps[rj].mean(), npv=(~ps[~rj]).mean())
        for k, v in st_.items():
            put(f"test.arch.{s_}.{k}", float(v), str(v) if k == "runs" else f"{v:.2f}", "K10", "e1_reml_test.json")
    # plug-in gamma-BLUP on the real panels
    losses, worst = 0, (None, 0.0)
    for ds in ("rice", "wheat", "mice", "rdp1", "sorghum"):
        G = gains(f"e4_{ds}_kfold_spectral.json", "gammaBLUP_ML")
        for t, (mu, p, q) in G.items():
            if q < .05 and mu < 0: losses += 1
            if q < .05 and mu > 0: raise SystemExit("gamma-BLUP gains significantly; revise the text")
            if mu < worst[1]: worst = (f"{ds}:{t}", mu)
    put("real.gblup.sig_losses", losses, str(losses), "K21", "e4_*_kfold_spectral.json")
    put("real.gblup.worst", worst[1], f(worst[1], 2, True), "K21", "e4_*_kfold_spectral.json")
    put("real.gblup.worst_trait", worst[0], worst[0], "K21", "e4_*_kfold_spectral.json")
    # mouse traits with gamma-hat below 0.5
    Rm = [r for r in J("e13_reml_real.json") if r["dataset"] == "mice" and r["gamma"] < 0.5]
    Gs, Gp = gains("e4_mice_kfold_spectral.json", "safe_CV"), gains("e4_mice_kfold_spectral.json", "power_CV")
    gs = [Gs[r["trait"]][0] for r in Rm]; gp = [Gp[r["trait"]][0] for r in Rm]
    put("mice.lowgamma.n", len(Rm), str(len(Rm)), "K15", "e13_reml_real.json")
    put("mice.lowgamma.gamma.range", [min(r["gamma"] for r in Rm), max(r["gamma"] for r in Rm)],
        f"{min(r['gamma'] for r in Rm):.2f} to {max(r['gamma'] for r in Rm):.2f}", "K15", "e13_reml_real.json")
    put("mice.lowgamma.safe.range", [min(gs), max(gs)], f"{f(min(gs), 4, True)} to {f(max(gs), 4, True)}", "K16", "e4_mice_kfold_spectral.json")
    put("mice.lowgamma.power.range", [min(gp), max(gp)], f"{f(min(gp), 4, True)} to {f(max(gp), 4, True)}", "K16", "e4_mice_kfold_spectral.json")
    put("mice.lowgamma.any_sig", any(Gs[r["trait"]][2] < .05 or Gp[r["trait"]][2] < .05 for r in Rm), "none significant", "K16", "e4_mice_kfold_spectral.json")
    # rice: oracle gap at both ends of the interval; RDP1 traits not rejected
    C = [r for r in J("gap_ci.json") if r["dataset"] == "rice"]
    rg = max(max(r["gap_lo"], r["gap_hi"]) for r in C)
    put("rice.gapci.max", float(rg), f"{rg:.4f}", "K17", "gap_ci.json")
    # leave-one-subpopulation-out: group-level R2 of rrBLUP
    S = J("e4_rice_subpop_spectral.json")
    g = [x for T_ in S for x in np.array(T_["r2"]["rrBLUP_CV"], float)]
    put("subpop.r2.groups", len(g), str(len(g)), "K19", "e4_rice_subpop_spectral.json")
    put("subpop.r2.groups_positive", int(sum(x > 0 for x in g)), str(int(sum(x > 0 for x in g))), "K19", "e4_rice_subpop_spectral.json")
    # pre-registration: size of the power-filter losses behind the N misses
    for p_ in ("rdp1", "sorghum"):
        L = sorted(-r["power_CV"] for r in J(f"prereg_evaluation_{p_}.json")["rows"]
                   if not r["prediction_met"] and r["category"] == "N" and r["power_CV"] <= -0.005)
        put(f"prereg.miss_nloss.{p_}.range", [L[0], L[-1]], f"{L[0]:.3f} to {L[-1]:.3f}", "K18", f"prereg_evaluation_{p_}.json")
    # average over the 61 traits from the spread across traits (t-interval)
    for m, lab in (("safe_CV", "safe"), ("power_CV", "power")):
        mu = np.array([np.mean(np.array(T["r2"][m]) - np.array(T["r2"]["rrBLUP_CV"]))
                       for ds in ("rice", "wheat", "mice", "rdp1", "sorghum") for T in by_trait(f"e4_{ds}_kfold_spectral.json").values()])
        h = stats.t.ppf(.975, len(mu) - 1) * mu.std(ddof=1) / np.sqrt(len(mu))
        put(f"pooled_t.{lab}", [float(mu.mean() - h), float(mu.mean() + h)], f"{f(float(mu.mean() - h), 4, True)} to {f(float(mu.mean() + h), 4, True)}", "K16", "e4_*_kfold_spectral.json")
    # accuracy of rrBLUP at gamma = 0.5, pooled over h2 for each n
    a_ = [np.mean([r["r2"]["rrBLUP_CV"] for r in R if r["scn"] == "g0.5" and r["n"] == n]) for n in (400, 800, 1600)]
    put("sim.g0.5.rr_r2_pooled.range", [min(a_), max(a_)], f"{min(a_):.2f} to {max(a_):.2f}", "K23", src)
    # leakage-free fold-level association on mice (check_fold_association.py)
    for r in J("fold_association.json"):
        if r["dataset"] == "mice" and r["method"] in ("safe_CV", "power_CV"):
            k = "safe" if r["method"] == "safe_CV" else "power"
            put(f"fold.mice.{k}.rej", r["gain_rejected"], f(r["gain_rejected"], 4, True), "K13", "fold_association.json")
            put(f"fold.mice.{k}.not", r["gain_not"], f(r["gain_not"], 4, True), "K13", "fold_association.json")
            put(f"fold.mice.{k}.diff", [r["diff"], r["lo"], r["hi"]], f"{f(r['diff'], 4, True)} (95% CI {f(r['lo'], 4, True)} to {f(r['hi'], 4, True)})", "K13", "fold_association.json")
    # oracle gap: largest value over each trait's bootstrap interval of gamma-hat (grid of 21; check_gap_ci_grid.py)
    Gg = J("gap_ci_grid.json"); up = np.array([r["gap_max"] for r in Gg])
    put("gapgrid.median", float(np.median(up)), f"{np.median(up):.4f}", "K17", "gap_ci_grid.json")
    put("gapgrid.below_0.005", float(np.mean(up < 0.005)), f"{np.mean(up < 0.005):.0%}", "K17", "gap_ci_grid.json")
    put("gapgrid.below_0.01", float(np.mean(up < 0.01)), f"{np.mean(up < 0.01):.0%}", "K17", "gap_ci_grid.json")
    put("gapgrid.max", float(up.max()), f"{up.max():.2f}", "K17", "gap_ci_grid.json")
    put("gapgrid.max_gamma", float(max(Gg, key=lambda r: r["gap_max"])["gamma_at_max"]), f"{max(Gg, key=lambda r: r['gap_max'])['gamma_at_max']:.1f}", "K17", "gap_ci_grid.json")
    rr = max(r["gap_max"] for r in Gg if r["dataset"] == "rice")
    put("rice.gapgrid.max", float(rr), f"{rr:.4f}", "K17", "gap_ci_grid.json")
    # real-GRM test (REML): number of rejections at gamma = 1 on wheat
    W = [t for t in J("e3_reml_test.json") if t["data"] == "wheat" and t["scn"] == "g1"]
    put("e3.wheat.g1.reject_n", int(sum(t["p_chi2"] < .05 for t in W)), f"{int(sum(t['p_chi2'] < .05 for t in W))} of {len(W)}", "K9", "e3_reml_test.json")


def audit_checks2():
    """Numbers added after the second review round: calibration of the corrected resampled t-test on the 61 traits,
    pooled and small-n test size, the downward bias of gamma-hat below 1, and the sorghum H2 statistic."""
    for m, lab in (("safe_CV", "safe"), ("power_CV", "power")):
        p_, mu, se = [], [], []
        for ds in ("rice", "wheat", "mice", "rdp1", "sorghum"):
            for T in by_trait(f"e4_{ds}_kfold_spectral.json").values():
                d = np.array(T["r2"][m]) - np.array(T["r2"]["rrBLUP_CV"]); J_ = len(d)
                s_ = np.sqrt((1 / J_ + 0.25) * d.var(ddof=1))
                p_.append(2 * stats.t.sf(abs(d.mean() / s_), J_ - 1) if s_ > 0 else 1.0); mu.append(d.mean()); se.append(s_)
        p_ = np.array(p_)
        put(f"calib.p.{lab}.min", float(p_.min()), f"{p_.min():.2f}", "K16", "e4_*_kfold_spectral.json")
        put(f"calib.p.{lab}.median", float(np.median(p_)), f"{np.median(p_):.2f}", "K16", "e4_*_kfold_spectral.json")
        r_ = float(np.sqrt(np.mean(np.square(se))) / np.std(mu, ddof=1))
        put(f"calib.se_ratio.{lab}", r_, f"{r_:.1f}", "K16", "e4_*_kfold_spectral.json")
    T = J("e1_reml_test.json")
    nul = [t["p_chi2"] < .05 for t in T if t["scn"] == "g1"]
    put("test.e1.size.pooled", float(np.mean(nul)), f"{sum(nul)} of {len(nul)}", "K9", "e1_reml_test.json")
    for s_ in ("g0.5", "g1"):
        r_ = [t["p_chi2"] < .05 for t in T if t["scn"] == s_ and t["n"] == 100]
        put(f"test.n100.{s_}.reject_n", int(sum(r_)), f"{sum(r_)} of {len(r_)}", "K9", "e1_reml_test.json")
    med = lambda s_, n: float(np.median([t["gamma"] for t in T if t["scn"] == s_ and t["n"] == n]))
    a_ = [med("g0.5", n) for n in (200, 400, 800)]
    put("gammahat.g0.5.n200_800", [min(a_), max(a_)], f"{min(a_):.2f} to {max(a_):.2f}", "K9", "e1_reml_test.json")
    put("gammahat.g0.5.n1600", med("g0.5", 1600), f"{med('g0.5', 1600):.2f}", "K9", "e1_reml_test.json")
    b_ = [med("g0.75", n) for n in (200, 400, 800, 1600)]
    put("gammahat.g0.75.n200_1600", [min(b_), max(b_)], f"{min(b_):.2f} to {max(b_):.2f}", "K9", "e1_reml_test.json")
    H = J("prereg_evaluation_sorghum.json")["H2"]
    put("prereg.sorghum.H2.rho", H["spearman"], f(H["spearman"], 2), "K18", "prereg_evaluation_sorghum.json")
    put("prereg.sorghum.H2.p", H["p_one_sided"], f"{H['p_one_sided']:.2f}", "K18", "prereg_evaluation_sorghum.json")


def audit_checks3():
    """Numbers for the technical examiner's points: the largest simulated gain at a single heritability, ties in the
    screening analysis, bootstrap intervals that exclude 1 without a rejection, replicates at n = 1,600, and the
    non-degenerate example of Remark 10 (check_loo_bias_example.py)."""
    R = J("e1_regimes.json"); best = 0.0
    for m in ("safe_CV", "power_CV", "gammaBLUP_ML"):
        for key in {(r["scn"], r["n"], r["h2"]) for r in R}:
            v = [r["r2"][m] - r["r2"]["rrBLUP_CV"] for r in R if (r["scn"], r["n"], r["h2"]) == key]
            best = max(best, float(np.mean(v)))
    put("sim.max_gain_h2.any", best, f"{best:.3f}", "K3", "e1_regimes.json")
    c = [sum(1 for r in R if r["scn"] == "g1" and r["n"] == 1600 and r["h2"] == h) for h in (0.3, 0.5, 0.7)]
    put("sim.n1600.g1.reps_by_h2", c, f"{c[0]}, {c[1]} and {c[2]}", "K10", "e1_regimes.json")
    T = J("e1_reml_test.json"); E = {(r["scn"], r["n"], r["h2"], r["rep"], r["seed"]): r for r in R}
    d = np.array([E[(t["scn"], t["n"], t["h2"], t["rep"], t["seed"])]["r2"]["safe_CV"]
                  - E[(t["scn"], t["n"], t["h2"], t["rep"], t["seed"])]["r2"]["rrBLUP_CV"] for t in T])
    rj = np.array([t["p_chi2"] < .05 for t in T]); tie = d == 0
    put("screen.ties.all", float(tie.mean()), f"{tie.mean():.0%}", "K10", "e1_reml_test.json")
    put("screen.ties.not_rejected", float(tie[~rj].mean()), f"{tie[~rj].mean():.0%}", "K10", "e1_reml_test.json")
    put("screen.ties.rejected", float(tie[rj].mean()), f"{tie[rj].mean():.0%}", "K10", "e1_reml_test.json")
    npv = float((~(d > 0.002))[~rj].mean())
    put("screen.npv_0.002", npv, f"{npv:.2f}", "K10", "e1_reml_test.json")
    P = {(r["dataset"], r["trait"]): r["p_boot"] for r in J("e13_reml_real.json")}
    for p_ in ("rdp1", "sorghum"):
        for r in J(f"prereg_evaluation_{p_}.json")["rows"]: P[(p_, r["trait"])] = r["p_boot"]
    S = J("e7_reml_boot.json")
    nr = [x for x in S if (x["ci"][1] < 1 or x["ci"][0] > 1) and P[(x["dataset"], x["trait"])] >= .05]
    assert all(x["gamma"] < 1 for x in nr)
    put("e7.n_excl1_not_rejected", len(nr), str(len(nr)), "K25", "e7_reml_boot.json + e13_reml_real.json")
    for x in J("loo_bias_example.json"):
        if x["gamma"] == 0.5:
            put("bias_ex.n", x["n"], str(x["n"]), "K7", "loo_bias_example.json")
            put("bias_ex.h", x["h"], f"{x['h']:.0f}", "K7", "loo_bias_example.json")
            put("bias_ex.rbar", [x["a_rbar"], x["delta_rbar"]], f"({x['a_rbar']:.2f}, {x['delta_rbar']:.2f})", "K7", "loo_bias_example.json")
            put("bias_ex.r", [x["a_r"], x["delta_r"]], f"({x['a_r']:.2f}, {x['delta_r']:.2f})", "K7", "loo_bias_example.json")
            put("bias_ex.excess", x["excess"], f"{x['excess']:.3f}", "K7", "loo_bias_example.json")


def audit_checks4():
    """Equivalence with multiplicity adjustment: TOST p-values from the corrected resampled t-test, adjusted by
    Benjamini-Hochberg within each panel and method (as the difference tests are)."""
    for m, lab in (("safe_CV", "safe"), ("power_CV", "power")):
        for mg in (0.005, 0.01, 0.02):
            k_raw = k_bh = 0
            for ds in ("rice", "wheat", "mice", "rdp1", "sorghum"):
                ps = []
                for T in by_trait(f"e4_{ds}_kfold_spectral.json").values():
                    d = np.array(T["r2"][m]) - np.array(T["r2"]["rrBLUP_CV"]); J_ = len(d)
                    se = max(np.sqrt((1 / J_ + 0.25) * d.var(ddof=1)), 1e-6)
                    ps.append(max(stats.t.sf((d.mean() + mg) / se, J_ - 1), stats.t.cdf((d.mean() - mg) / se, J_ - 1)))
                k_raw += sum(p < .05 for p in ps); k_bh += sum(q < .05 for q in bh(ps))
            assert k_raw == NUM[f"equiv.{lab}.{mg}"]["value"], (lab, mg)   # same rule as the 90% interval
            put(f"equiv_bh.{lab}.{mg}", int(k_bh), str(int(k_bh)), "K16", "e4_*_kfold_spectral.json")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")   # the minus signs print on any console
    except AttributeError:
        pass
    for fn in (e1, reml_test, real, other, panels, practical, small_n_close_eigs, adjustment, panels_r22, baselines,
               practical_rice_structure, epistatic_mice, stability_power, robustness_checks, audit_checks,
               audit_checks2, audit_checks3, audit_checks4): fn()
    json.dump(NUM, open(os.path.join(RES, "numbers.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    for k, v in NUM.items(): print(f"{k:45s} = {v['text']}")
    print(f"\n{len(NUM)} numbers written to results/numbers.json")
