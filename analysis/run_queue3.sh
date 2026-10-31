#!/usr/bin/env bash
# Third queue (2026-10-29): checks added after the REML correction and the cage-out epistasis result.
#   1. E1 rejection rates with the REML test
#   2. mice KRR under cage-out CV (are the k-fold KRR gains shared cage environment?)
#   3. Bayesian baselines with long chains (the 3000-iteration chains failed the Geweke screen); written with tag _long
cd "$(dirname "$0")"
L=results
echo "start $(date)" > $L/queue3.log
E1_PROCS=4 python e1_reml_test.py > $L/queue3_e1_reml.log 2>&1;                                         echo "e1 reml $(date)" >> $L/queue3.log
E4_METHODS=KRR_RBF python e4_real.py mice cage kernel 3 4 > $L/queue3_mice_cage_krr.log 2>&1;           echo "mice cage krr $(date)" >> $L/queue3.log
E4_TAG=_long GIBBS_ITER=20000 GIBBS_BURN=5000 python e4_real.py rice kfold bayes 5 4 > $L/queue3_rice_bayes_long.log 2>&1;   echo "rice bayes long $(date)" >> $L/queue3.log
E4_TAG=_long GIBBS_ITER=20000 GIBBS_BURN=5000 python e4_real.py wheat kfold bayes 5 4 > $L/queue3_wheat_bayes_long.log 2>&1; echo "wheat bayes long $(date)" >> $L/queue3.log
E4_TAG=_long GIBBS_ITER=12000 GIBBS_BURN=3000 python e4_real.py mice kfold bayes 1 4 > $L/queue3_mice_bayes_long.log 2>&1;   echo "mice bayes long $(date)" >> $L/queue3.log
echo "QUEUE3 DONE $(date)" >> $L/queue3.log
