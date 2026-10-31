"""How much could ANY spectral filter gain over the best ridge on the real traits? (T2 applied to data)

For every trait of every panel: REML basis (e13_reml_diag), ML fit of z_i ~ N(0, c lam_i^g + s2), then the exact oracle gap
    Delta(g) = min_delta mean_i (tau_i^2 + s2) (lam_i/(lam_i+delta) - w_i(g))^2,   tau_i^2 = c lam_i^g,  w_i = tau_i^2/(tau_i^2+s2),
divided by the total variance mean(tau^2 + s2): the in-sample R^2 an oracle gamma-filter would add over the best ridge.
It is an upper bound for the in-sample gain of every member of the spectral family under the fitted model.
writes results/gap_real.json
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import json, numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
PANELS = ("rice", "wheat", "mice", "rdp1", "sorghum")


def run(args):
    from e13_reml_diag import load, reml_basis
    from spectral_filters import ml_fit
    name, trait = args; T = load(name)["traits"][trait]; rows = T["rows"]; G = load(name)["G"][np.ix_(rows, rows)]
    X = np.ones((len(rows), 1)) if T["C"] is None else T["C"]
    lam, A = reml_basis(G, X); z = A @ T["y"]; z2 = (z / z.std()) ** 2
    c, s2, g, _ = ml_fit(lam, z2); tau2 = c * lam ** g; w = tau2 / (tau2 + s2)
    f = lambda ld: np.mean((tau2 + s2) * (lam / (lam + np.exp(ld)) - w) ** 2)
    gap = minimize_scalar(f, bounds=(-15, 15), method="bounded", options={"xatol": 1e-10}).fun
    return dict(dataset=name, trait=trait, gamma=float(g), h2=float(np.mean(tau2) / np.mean(tau2 + s2)), gap_r2=float(gap / np.mean(tau2 + s2)))


if __name__ == "__main__":
    from e13_reml_diag import load
    tasks = [(p, t) for p in PANELS for t in load(p)["traits"]]
    with Pool(int(os.environ.get("GAP_PROCS", 4))) as pool:
        res = pool.map(run, tasks, chunksize=1)
    json.dump(res, open(os.path.join(HERE, "results", "gap_real.json"), "w"), indent=1)
    g = np.array([r["gap_r2"] for r in res])
    print(f"traits {len(res)}; oracle in-sample gap (R^2): median {np.median(g):.5f}, 90th pct {np.percentile(g, 90):.5f}, max {g.max():.5f}")
    for r in sorted(res, key=lambda r: -r["gap_r2"])[:8]:
        print(f"  {r['dataset']:7s} {r['trait'][:30]:30s} gamma {r['gamma']:.2f} h2 {r['h2']:.2f} gap {r['gap_r2']:.4f}")
