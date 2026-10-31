# 09 — Pre-registration: new real-data test of the regime theory

**Written 2026-10-29**, before any genotype file of the new panels was opened and before any diagnostic or accuracy was computed on them. The SHA-256 hash of this file is recorded in `01_Findings_and_Evidence_Log.md` at the time of writing. Nothing in sections 1–5 may change after the analysis starts. Any deviation is listed in section 6 with its reason.

## 1. Purpose

Test the theory's predictions on panels that were not used to build it:
- **T3:** spectral adaptation gains when γ > 1.
- **T1:** no gain when γ = 1.
- **P4:** no material spectral gain when γ < 1.

The panels were chosen **a priori** for strong population structure and many traits, where γ > 1 traits are most likely. They were not chosen for any observed result.

## 2. Data

| Panel | Source | Individuals | Markers | Traits |
|---|---|---|---|---|
| Rice Diversity Panel 1 (Zhao et al. 2011, *Nat Commun* 2:467) | ricediversity.org: `RiceDiversity.44K.MSU6.Genotypes_PLINK.zip`, `RiceDiversity_44K_Phenotypes_34traits_PLINK.txt` | 413 | ~44K (MSU6) | 36 columns |
| Sorghum diversity panel (as formatted by Azodi et al. 2019, *G3*; Dryad doi:10.5061/dryad.xksn02vb9, CC0) | `sorghum_geno.csv`, `sorghum_pheno.csv` | 451 | ~56K | 3 |

## 3. Pre-specified processing

1. **Markers:** coded 0/1/2. Remove markers with call rate < 0.9 or MAF < 0.01. Remove individuals with call rate < 0.8. Impute missing genotypes with the marker mean. Standardise as in `datasets.standardise`. GRM = ZZᵀ/p.
2. **Trait inclusion (rice), fixed now from the phenotype file only:** a trait is analysed if it has n ≥ 300 non-missing values and ≥ 30 distinct values. This excludes the binary and coarse ordinal traits (Leaf pubescence, Awn presence, Seed color, Pericarp color, Blast resistance) and keeps **31 traits**. Sorghum: all three traits.
3. **No covariates.** Fixed effects = intercept only. Phenotypes are used as provided.
4. **Diagnostic:** REML γ = 1 test (`e13_reml_diag.py`, B = 200 bootstrap) on all individuals with the trait.
5. **Accuracy:** 5 × 5 cross-validation (seeds 0–4), identical splits for all methods: rrBLUP_CV, power_CV, safe_CV, gammaBLUP_ML, plus KRR_RBF. Δ = method − rrBLUP_CV in held-out R². Nadeau–Bengio corrected t-test, then BH across traits within each panel.

## 4. Pre-registered predictions

The prediction for each trait is written to `results/prereg_predictions_<panel>.json` **after the diagnostic and before any accuracy is computed**. The rules:

| Diagnostic outcome | Prediction for the spectral filters (power_CV, safe_CV) |
|---|---|
| **R+:** rejected (p_boot < 0.05) with γ̂ > 1 | Gain: mean Δ > 0 |
| **N:** not rejected | No material difference: \|Δ\| < 0.005 and no BH-significant gain |
| **R−:** rejected with γ̂ < 1 | No material spectral gain (P4): mean Δ < 0.005 |

**Primary confirmatory tests** (decided now):
- **H1:** mean Δpower over R+ traits > mean Δpower over N traits. One-sided permutation test over trait labels; α = 0.05.
- **H2:** Spearman correlation between γ̂ and Δpower across all traits > 0. One-sided; α = 0.05.
- **H3:** the share of traits whose outcome matches its prediction, reported with a 95% Wilson interval.

If no trait is R+, H1 cannot be tested. This is reported as "the γ > 1 regime did not occur in this panel", and only H2 and H3 are reported.

## 5. Reporting

Every analysed trait is reported, whatever its result. The paper states that the panels and analyses were pre-registered in this file, and the file goes into the reproducibility repository.

## 6. Deviations

(none yet)
