#!/usr/bin/env bash
# Second queue (E11 validations); starts when the main queue has finished.
cd "$(dirname "$0")"
L=results
until grep -q "QUEUE DONE" $L/queue.log 2>/dev/null; do sleep 120; done
echo "start $(date)" > $L/queue2.log
for d in mice rice wheat; do
  python e11_extra.py foldiag $d > $L/queue2_foldiag_$d.log 2>&1; echo "foldiag $d $(date)" >> $L/queue2.log
  python e11_extra.py ldak $d    > $L/queue2_ldak_$d.log 2>&1;    echo "ldak $d $(date)" >> $L/queue2.log
  python e11_extra.py epi $d     > $L/queue2_epi_$d.log 2>&1;     echo "epi $d $(date)" >> $L/queue2.log
done
echo "QUEUE2 DONE $(date)" >> $L/queue2.log
