"""E14 (R5.6): stability of eigenvalue-indexed and rank-indexed shrinkage when GRM eigenvalues are closely spaced.

For each panel, the GRM is recomputed from a random 90% of the markers (B replicates). We compare the smoother
S = U diag(w) U^T before and after, for
  power    w_i = h(lambda_i) = lambda_i^2 / (lambda_i^2 + delta)       (eigenvalue-indexed, as in (A4))
  grouped  the same weights averaged within K contiguous blocks of ranks (rank-indexed, Section 3.6)
and record how often neighbouring eigenvalues are close, how much the eigenvectors move, and the Frobenius
Lipschitz bound ||h(G') - h(G)||_F <= L_h ||G' - G||_F (Appendix A).
usage: python e14_close_eigs.py [B]
"""
import os, sys, json, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import datasets  # noqa: E402

B = int(sys.argv[1]) if len(sys.argv) > 1 else 20
CLOSE = 0.02                                  # relative gap (lambda_i - lambda_{i+1}) / lambda_i below 2%


def grm(Z):
    G = Z @ Z.T / Z.shape[1]
    return G / np.mean(np.diag(G))


def eig(G):
    lam, U = np.linalg.eigh(G)
    return np.clip(lam[::-1], 0, None), U[:, ::-1]


def panel(name, rng):
    Z = datasets.LOADERS[name]()["Z"]
    n, p = Z.shape
    G = grm(Z); lam, U = eig(G)
    rho = int(round(lam.sum() ** 2 / (lam ** 2).sum()))
    delta = np.median(lam[lam > 1e-8]) ** 2
    h = lambda l: l ** 2 / (l ** 2 + delta)
    grid = np.linspace(0, lam.max() * 1.5, 200001); L_h = float(np.max(np.abs(np.gradient(h(grid), grid))))
    K = max(rho, 2); blocks = np.array_split(np.arange(n), K)

    def smoothers(lam, U):
        w = h(lam); wg = np.empty(n)
        for b in blocks: wg[b] = w[b].mean()
        return (U * w) @ U.T, (U * wg) @ U.T

    top = np.arange(min(rho, n - 1))
    gap = (lam[top] - lam[top + 1]) / lam[top]
    close = top[gap < CLOSE]; far = top[gap >= CLOSE]
    Sp, Sg = smoothers(lam, U)
    rec = dict(dG=[], dP=[], dGr=[], lip=[], cos_close=[], cos_far=[])
    for _ in range(B):
        keep = rng.random(p) < 0.9
        G2 = grm(Z[:, keep]); lam2, U2 = eig(G2)
        Sp2, Sg2 = smoothers(lam2, U2)
        dG = np.linalg.norm(G2 - G); rec["dG"].append(dG / np.linalg.norm(G))
        rec["dP"].append(np.linalg.norm(Sp2 - Sp) / np.linalg.norm(Sp))
        rec["dGr"].append(np.linalg.norm(Sg2 - Sg) / np.linalg.norm(Sg))
        rec["lip"].append(np.linalg.norm(Sp2 - Sp) / dG)
        c = np.abs(np.sum(U[:, top] * U2[:, top], axis=0))
        rec["cos_close"] += c[np.isin(top, close)].tolist(); rec["cos_far"] += c[np.isin(top, far)].tolist()
    med = lambda v: float(np.median(v)) if len(v) else None
    return dict(panel=name, n=n, markers=p, rho_eff=rho, n_top=len(top), n_close=int(len(close)),
                L_h=L_h, max_lip_ratio=float(np.max(rec["lip"])), bound_holds=bool(np.max(rec["lip"]) <= L_h * (1 + 1e-9)),
                rel_change_G=med(rec["dG"]), rel_change_power=med(rec["dP"]), rel_change_grouped=med(rec["dGr"]),
                eigvec_cos_close=med(rec["cos_close"]), eigvec_cos_far=med(rec["cos_far"]), B=B)


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    out = [panel(name, rng) for name in ("rice", "wheat", "mice")]
    for r in out: print(r)
    json.dump(out, open(os.path.join(HERE, "results", "e14_close_eigs.json"), "w"), indent=1)
