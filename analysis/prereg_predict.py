"""Pre-registration step 1 (09 §4): REML gamma = 1 diagnostic on every pre-registered trait and the theory's prediction,
written BEFORE any accuracy is computed on the panel.

usage: python prereg_predict.py PANEL        (PANEL: a datasets.LOADERS key, e.g. rdp1 or sorghum)
writes results/prereg_predictions_PANEL.json
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import json, time, numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))


def predict(args):
    from e13_reml_diag import real_task
    r = real_task(args)
    if r["p_boot"] < 0.05 and r["gamma"] > 1: r["category"], r["prediction"] = "R+", "gain: mean delta > 0"
    elif r["p_boot"] < 0.05: r["category"], r["prediction"] = "R-", "no material spectral gain: mean delta < 0.005"
    else: r["category"], r["prediction"] = "N", "no material difference: |delta| < 0.005, no significant gain"
    return r


if __name__ == "__main__":
    import datasets
    panel = sys.argv[1]; traits = list(datasets.LOADERS[panel]()["traits"])
    with Pool(int(os.environ.get("PREREG_PROCS", 6))) as pool:
        res = pool.map(predict, [(panel, t) for t in traits], chunksize=1)
    out = dict(panel=panel, written=time.strftime("%Y-%m-%d %H:%M:%S %Z"), rule="09_Preregistration_New_Data.md section 4", traits=res)
    json.dump(out, open(os.path.join(HERE, "results", f"prereg_predictions_{panel}.json"), "w"), indent=1)
    for r in res: print(f"{r['category']:3s} {r['trait'][:40]:40s} n={r['n']:3d} gamma={r['gamma']:.2f} p_boot={r['p_boot']:.3f}")
