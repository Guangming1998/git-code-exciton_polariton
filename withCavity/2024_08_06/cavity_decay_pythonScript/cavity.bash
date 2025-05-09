#!/bin/bash

##$ -M gliu8@nd.edu   # Email address for job notification
##$ -m abe            # Send mail when job begins, ends and aborts
##$ -pe mpi-1 1     # Specify parallel environment and legal core size
#$ -q long@@theta_lab           # Specify queue
#$ -N disorder      # Specify job name
#$ -o job.%j.out

##module load xyz      # Required modules

##mpirun -np $NSLOTS ./app # Application to execute

source ~/.bashrc
echo "starting run at" `date`
python cavity.py 
echo "Finished run at" `date`
