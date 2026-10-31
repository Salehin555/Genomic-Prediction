"""The test of gamma = 1 on the semi-synthetic real-GRM data sets of e3_semisynth.py, recomputed with the REML test of
Section 3.8 (intercept projected out; chi2 p-value, as in the simulation of Section 5.2). The phenotypes are regenerated
from the stored seeds, so they are the same data sets as in results/e3_regimes.json; the predictive results there do
not depend on the test and are not recomputed.  writes results/e3_reml_test.json
usage: python e3_reml_recheck.py
"""
import os, json, time
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import e3_semisynth as E3
from e13_reml_diag import reml_basis, reml_test

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results", "e3_regimes.json")))
out, t0, basis = [], time.time(), {}
for r in sorted(R, key=lambda r: r["seed"]):
    d = r["data"]
    if d not in basis:
        G, _, _ = E3.eig(d); basis[d] = reml_basis(G, np.ones((len(G), 1)))
    rng = np.random.default_rng(r["seed"])
    y = E3.phenotype(d, r["scn"], r["h2"], rng)
    lam, A = basis[d]
    t = reml_test(lam, A, y, 0, r["seed"])
    out.append(dict(data=d, scn=r["scn"], h2=r["h2"], rep=r["rep"], seed=r["seed"], gamma=t["gamma"], p_chi2=t["p_chi2"],
                    old_gamma=r["test"]["gamma"], old_p=r["test"].get("p", r["test"].get("p_chi2"))))
json.dump(out, open(os.path.join(HERE, "results", "e3_reml_test.json"), "w"), indent=1)
print(len(out), "records", round(time.time() - t0), "s")
for d in ("mice", "wheat", "rice"):
    for s in ("g0.5", "g1", "g2", "g3", "structure"):
        v = [o for o in out if o["data"] == d and o["scn"] == s]
        print(d, s, "median gamma %.2f (old %.2f)  reject %.2f (old %.2f)" % (
            np.median([o["gamma"] for o in v]), np.median([o["old_gamma"] for o in v]),
            np.mean([o["p_chi2"] < .05 for o in v]), np.mean([o["old_p"] < .05 for o in v])))
