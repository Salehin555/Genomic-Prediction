#!/usr/bin/env bash
# Revision experiment queue (mouse first; E1 resumes near the end). Each line logs to results/queue_*.log
cd "$(dirname "$0")"
L=results
echo "start $(date)" > $L/queue.log
python e4_real.py mice kfold spectral 5 11            > $L/queue_mice_spectral.log 2>&1;  echo "mice spectral $(date)" >> $L/queue.log
python e4_real.py mice cage spectral 3 11             > $L/queue_mice_cage.log 2>&1;      echo "mice cage $(date)" >> $L/queue.log
python e4_real.py mice_auto kfold spectral 3 11       > $L/queue_mice_auto.log 2>&1;      echo "mice auto $(date)" >> $L/queue.log
python e4_real.py mice kfold kernel 2 11              > $L/queue_mice_kernel.log 2>&1;    echo "mice kernel $(date)" >> $L/queue.log
GIBBS_ITER=3000 GIBBS_BURN=1000 python e4_real.py mice kfold bayes 1 11 > $L/queue_mice_bayes.log 2>&1; echo "mice bayes $(date)" >> $L/queue.log
(python e4_real.py rice kfold spectral 5 4 > $L/queue_rice_spectral.log 2>&1 &
 python e4_real.py wheat kfold spectral 5 4 > $L/queue_wheat_spectral.log 2>&1 &
 python e4_real.py rice subpop spectral 1 3 > $L/queue_rice_subpop.log 2>&1 & wait); echo "rice/wheat spectral $(date)" >> $L/queue.log
(python e4_real.py rice kfold kernel 5 4 > $L/queue_rice_kernel.log 2>&1 &
 python e4_real.py wheat kfold kernel 5 4 > $L/queue_wheat_kernel.log 2>&1 &
 python e4_real.py rice kfold bayes 5 3 > $L/queue_rice_bayes.log 2>&1 & wait); echo "rice/wheat kernel, rice bayes $(date)" >> $L/queue.log
(python e4_real.py wheat kfold bayes 5 4 > $L/queue_wheat_bayes.log 2>&1 &
 python e3_semisynth.py size 200 > $L/queue_e2_size.log 2>&1 & wait); echo "wheat bayes, e2 size $(date)" >> $L/queue.log
python e3_semisynth.py regimes 10                     > $L/queue_e3_regimes.log 2>&1;     echo "e3 regimes $(date)" >> $L/queue.log
python e7_e8_e10.py stability                         > $L/queue_e7.log 2>&1;             echo "e7 $(date)" >> $L/queue.log
python e7_e8_e10.py ablation                          > $L/queue_e10.log 2>&1;            echo "e10 $(date)" >> $L/queue.log
python e1_regimes.py 25 10                           > $L/e1_regimes.log 2>&1;           echo "e1 $(date)" >> $L/queue.log
python e7_e8_e10.py scaling                           > $L/queue_e8.log 2>&1;             echo "e8 $(date)" >> $L/queue.log
echo "QUEUE DONE $(date)" >> $L/queue.log
