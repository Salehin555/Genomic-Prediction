# Scripts and the paper

## Main files

| File | Role | Paper |
|---|---|---|
| `spectral_filters.py` | rrBLUP, the power filter and the safe selector on a kernel's eigenbasis | Sections 3.6, 3.7 |
| `gamma_fit.py`, `e13_reml_diag.py` | Power-law fit and the restricted-likelihood test of gamma = 1 (fixed effects projected out) | Section 3.8 |
| `e1_regimes.py`, `e1_reml_test.py` | Simulation with new individuals (2,760 data sets) and the test's size and power | Sections 4.2, 5.1, 5.2 |
| `e4_real.py`, `bayes_gibbs.py`, `e11_extra.py` | Real-data comparison on identical folds (spectral filters, KRR, an MLP stand-in for DNNGP, PCR, BRR, BayesC, BayesR, LDAK-type GBLUP, epistatic kernel; k-fold, cage-out, subpopulation-out) | Sections 4.3, 4.4, 5.3, 5.6 to 5.8 |
| `check_gap_real.py` | Oracle gap on the 61 traits | Section 5.5 |
| `check_gap_ci.py`, `check_gap_ci_grid.py` | Oracle gap at the ends of, and the largest value over, each trait's bootstrap interval of gamma-hat | Section 5.5 |
| `e3_semisynth.py`, `e3_reml_recheck.py` | Simulation on the real rice, wheat and mouse GRMs; the REML test on the same data sets | Sections 4.2, 5.1 |
| `check_fold_association.py` | Leakage-free check of the test within folds | Section 5.2 |
| `prereg_predict.py`, `prereg_evaluate.py` | Pre-registration: predictions before accuracy, then evaluation | Sections 4.6, 5.4 |
| `e7_reml_boot.py`, `e7_e8_e10.py`, `e14_close_eigs.py`, `e15_adjust.py` | Bootstrap stability, run time, LOO ablation, close eigenvalues, fixed-effect adjustment | Sections 3.4, 5.8 to 5.12 |
| `check_theory_results.py`, `check_T2_expansion.py`, `check_T7.py`, `check_loo_bias_example.py` | Numerical checks of the theory (Theorem 1, Proposition 3, Remarks 3, 9 and 10) | Section 3, Appendices A and B |
| `make_numbers.py` | Writes `results/numbers.json` | all |
| `datasets.py`, `build_*.py`, `panel_info.py` | Data preparation | Section 4.1 |
| `sasblup_engine.py`, `sasblup_fixed.py`, `synth_fixed.py`, `run_fixed.py` | Rank-indexed families with leave-one-out tuning, and the same families with the safeguards of Section 3.7 | Section 5.9, Supplementary Material |

## Notes on rebuilding

`make_numbers.py` recomputes the F-statistics of Section 5.6 when the rice GRM is present and otherwise reads them
from `results/e7_stability.json`; every other number needs only the stored results. `fig_method_flow.py`,
`fig2_regime_map.py` and `fig_theory_sim_test.py` also need `analysis/data/mice.RData` (BGLR), and panel b of Fig. 2
needs the rice GRM; the other figures need only the stored results.

## Rebuilding the paper versus rerunning the experiments

Rebuilding uses the stored results: `make_numbers.py` and the figure scripts recreate every number, table and figure
in minutes. Rerunning recomputes the stored results themselves, one experiment script at a time, and takes hours.

The simulation (`results/e1_regimes.json`, 2,760 data sets) is deterministic. Each data set is fixed by its seed,
which is 20000 plus its position in the design built with REPS = 25; `results/e1_design.json` lists the 2,760 seeds.
To regenerate exactly these data sets from scratch, delete `analysis/results/e1_regimes.partial.json` if it exists
(the script resumes from it), then run:

```
cd analysis
python e1_regimes.py 25 25 --exact
```

The design has 25 replicates per heritability for every scenario at n = 100 to 800. At n = 1,600 it has 25 for
gamma = 0.5 and 0.75, 25, 25 and 10 for gamma = 1 (heritability 0.3, 0.5 and 0.7), and 10 for the other scenarios
(Section 4.2 of the paper). The stored file came from a first run that finished 200 data sets at n = 1,600 and was
then stopped, followed by `python e1_regimes.py 25 10`, which skips the data sets already done (its log reports "200
done, 2560 to run"); both steps used the same seeds. Recomputed data sets match the stored ones to within 1e-8 in
R2 (checked on a sample).

The figure files keep their working names. In the paper they are numbered as follows: `fig1_method_flow`
Fig. 1, `fig3_theory` Fig. 2, `fig4_simulation` Fig. 3, `fig5_test_calibration` Fig. 4, `fig6_gamma_real` Fig. 5,
`fig7_gains_real` Fig. 6, `fig8_oracle_gap` Fig. 7, `fig2_regime_map` Fig. 8, `fig9_mouse` Fig. 9, `figS3_gibbs`
Fig. S1, `figS1_loo_ablation` Fig. S2 and `figS2_runtime` Fig. S3.
