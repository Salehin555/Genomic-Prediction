"""Size of every real panel as analysed (individuals, markers after QC, traits, individuals per trait) for Table 2.
writes results/panel_info.json;  usage: python panel_info.py
"""
import os, json, numpy as np
import datasets

HERE = os.path.dirname(os.path.abspath(__file__))
out = {}
for name in ("rice", "wheat", "mice", "rdp1", "sorghum"):
    D = datasets.LOADERS[name]()
    ns = [len(t["y"]) for t in D["traits"].values()]
    out[name] = dict(n=int(len(D["G"])), markers=int(D["Z"].shape[1]) if D["Z"] is not None else None,
                     traits=len(D["traits"]), n_min=int(min(ns)), n_max=int(max(ns)),
                     covariates=any(t["C"] is not None for t in D["traits"].values()))
    print(name, out[name], flush=True)
json.dump(out, open(os.path.join(HERE, "results", "panel_info.json"), "w"), indent=1)
