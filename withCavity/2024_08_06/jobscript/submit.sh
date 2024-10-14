#!/bin/bash
for f in {5..6}
do
  cd cavity*_$f
    qsub jobarray.sh
    qsub -hold_jid my_job_array_$f final_aggregation.sh
  cd ..
done
