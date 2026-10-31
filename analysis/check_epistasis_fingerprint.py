"""Does non-additive signal look like gamma < 1 in the additive GRM's eigenbasis?
If g ~ N(0, K_true), the expected signal variance on additive eigen-direction i is tau_i^2 = u_i' K_true u_i.
For K_true = G (additive) tau^2 = lambda (gamma = 1). For epistatic kernels (G o G, RBF) we fit log tau^2 on log lambda
over the informative part of the spectrum, and also run the gamma ML fit on phenotypes simulated from each kernel."""
import numpy as np, datasets
from spectral_filters import ml_fit
for name in ("mice", "wheat", "rice"):
    G = datasets.rice(markers=False)["G"] if name == "rice" else datasets.LOADERS[name]()["G"]
    ev, U = np.linalg.eigh(G); lam = np.maximum(ev, 1e-10); n = len(lam)
    d = np.diag(G); D2 = d[:, None] + d[None, :] - 2 * G; RBF = np.exp(-D2 / np.median(D2[D2 > 0]))
    kernels = {"additive G": G, "AxA epistasis G o G": G * G, "RBF (KRR)": RBF}
    top = lam > np.quantile(lam, 0.05)                      # drop the near-null tail
    print(f"\n{name} (n = {n})")
    rng = np.random.default_rng(0)
    for kname, Kt in kernels.items():
        tau2 = np.einsum("ij,jk,ki->i", U.T, Kt, U)
        slope = np.polyfit(np.log(lam[top]), np.log(np.maximum(tau2[top], 1e-12)), 1)[0]
        L = np.linalg.cholesky(Kt / np.mean(np.diag(Kt)) + 1e-6 * np.eye(n)); gh = []
        for r in range(20):                                  # simulate h2 = 0.5 phenotypes from the kernel
            y = L @ rng.standard_normal(n) * np.sqrt(0.5) + rng.standard_normal(n) * np.sqrt(0.5)
            z2 = (U.T @ ((y - y.mean()) / y.std())) ** 2
            gh.append(ml_fit(lam / lam.mean(), z2)[2])
        print(f"  {kname:22s} spectral slope of tau^2 on lambda = {slope:5.2f}   ML gamma-hat on simulated phenotypes: median {np.median(gh):.2f} [{np.percentile(gh,10):.2f}, {np.percentile(gh,90):.2f}]")
