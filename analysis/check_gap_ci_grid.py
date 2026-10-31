"""Uncertainty of the oracle gap, worst case over the interval: for every trait, the gap of check_gap_real.py on a grid of
21 values of gamma across the 95% bootstrap interval of gamma-hat (e7_reml_boot.json), with c and s2 refitted by ML at
each fixed gamma. The gap is not monotone in gamma, so the two ends of the interval (check_gap_ci.py) need not be the
largest value; the grid maximum is. writes results/gap_ci_grid.json
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import json, numpy as np
from multiprocessing import Pool
from check_gap_ci import gap_at

HERE = os.path.dirname(os.path.abspath(__file__))


def run(args):
    from e13_reml_diag import load, reml_basis
    name, trait, lo, hi = args; T = load(name)["traits"][trait]; rows = T["rows"]; G = load(name)["G"][np.ix_(rows, rows)]
    X = np.ones((len(rows), 1)) if T["C"] is None else T["C"]
    lam, A = reml_basis(G, X); z = A @ T["y"]; z2 = (z / z.std()) ** 2
    grid = np.linspace(lo, hi, 21); gaps = [gap_at(lam, z2, g) for g in grid]; k = int(np.argmax(gaps))
    return dict(dataset=name, trait=trait, ci=[lo, hi], gap_max=float(gaps[k]), gamma_at_max=float(grid[k]))


if __name__ == "__main__":
    B = {(x["dataset"], x["trait"]): x["ci"] for x in json.load(open(os.path.join(HERE, "results", "e7_reml_boot.json")))}
    tasks = [(d, t, ci[0], ci[1]) for (d, t), ci in B.items()]
    with Pool(int(os.environ.get("GAP_PROCS", 4))) as pool:
        res = pool.map(run, tasks, chunksize=1)
    json.dump(res, open(os.path.join(HERE, "results", "gap_ci_grid.json"), "w"), indent=1)
    up = np.array([r["gap_max"] for r in res])
    print(f"traits {len(res)}; largest gap over the interval: median {np.median(up):.4f}, <0.005 {np.mean(up < 0.005):.0%}, "
          f"<0.01 {np.mean(up < 0.01):.0%}, max {up.max():.4f}")
    for r in sorted(res, key=lambda r: -r["gap_max"])[:8]:
        print(f"  {r['dataset']:7s} {r['trait'][:28]:28s} ci {r['ci'][0]:.2f}-{r['ci'][1]:.2f} max {r['gap_max']:.4f} at {r['gamma_at_max']:.2f}")
    print("rice max", max(r["gap_max"] for r in res if r["dataset"] == "rice"))
