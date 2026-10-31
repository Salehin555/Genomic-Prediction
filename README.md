# SAS-BLUP: Spectral Adaptive Shrinkage BLUP

Code and stored results for the paper *Spectral Adaptive Shrinkage BLUP for Genomic Prediction: A Likelihood-Ratio
Test of the GBLUP Assumption and a Multi-Panel Evaluation* by Md. Salehin Seyam, Abdur Rahman and Zhaohua Peng.

GBLUP (equivalently rrBLUP) shrinks each eigenvector of the genomic relationship matrix (GRM) by a fixed rule. SAS-BLUP
learns this shrinkage with a two-parameter power filter and chooses between the filter and rrBLUP on held-out data. A
restricted-likelihood ratio test checks the GBLUP assumption (gamma = 1); it uses the phenotype but is computed before
the predictive model is fitted. The scripts here
reproduce every number, table and figure of the paper from the stored results.

## Contents

| Path | What it holds |
|---|---|
| `analysis/` | Implementation, experiments and checks (Python). [`analysis/SCRIPTS.md`](analysis/SCRIPTS.md) maps each script to the paper. |
| `analysis/results/` | Stored results of every experiment (JSON). `numbers.json` holds every number quoted in the paper, with its source file. |
| `analysis/figures/` | Figure scripts; `out/` holds every figure as PDF and PNG. |
| `preregistration/` | Analysis plan and predictions for RDP1 and sorghum, fixed before any accuracy was computed, with SHA-256 hashes. |
| `notebook/` | Notebook with the rank-indexed families and leave-one-out tuning (ablation in Section 5.9, Supplementary Section S3). |

Core methods: `analysis/spectral_filters.py` (rrBLUP, the power filter and the safe selector) and
`analysis/gamma_fit.py` with `analysis/e13_reml_diag.py` (power-law fit and the likelihood-ratio test).

## Quick start

Python 3.12 (tested with 3.12.7) and a CPU are enough.

```
pip install -r requirements.txt
cd analysis
python make_numbers.py          # rebuilds results/numbers.json from the stored results
cd figures
python fig_real.py              # Figs. 5 to 7 and 9
python fig_workflow_supp.py     # Supplementary Figs. S1 to S3
```

Each script states its usage in its first lines. These commands rebuild the numbers and figures from the stored
results in minutes; rerunning an experiment recomputes its results, takes hours and needs the data below.
[`analysis/SCRIPTS.md`](analysis/SCRIPTS.md) gives the exact command that regenerates the 2,760 simulated data sets.

## Data

Raw genotype files are not redistributed here. Place them in `analysis/data/`, except the two rice files, which go in
`data/Genotype Rice Data/` one level above `analysis/`:

- Wheat and mice: the `wheat` and `mice` data sets of the BGLR package for R (Pérez and de los Campos, 2014); save
  them as `analysis/data/wheat.RData` and `analysis/data/mice.RData`.
- RDP1: `RiceDiversity.44K.MSU6.Genotypes_PLINK.zip` and `RiceDiversity_44K_Phenotypes_34traits_PLINK.txt` from
  ricediversity.org (Zhao et al., 2011); then run `build_rdp1.py`.
- Sorghum: the files of Azodi et al. (2019), Dryad doi:10.5061/dryad.xksn02vb9, from Fernandes et al. (2018); then run
  `build_sorghum.py`.
- Rice panel (191 accessions): available from the corresponding author on reasonable request; then run
  `build_rice_grm.py` and `build_rice_markers.py`. For the same reason, the stored rice results
  (`results/e4_rice_*.json`) keep the fold predictions and accuracies but not the phenotypes; `make_numbers.py` then
  keeps the stored values of the rice selection measures.

## License

MIT; see [LICENSE](LICENSE).
