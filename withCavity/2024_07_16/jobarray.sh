#!/bin/bash
#$ -N my_job_array
#$ -t 1-500

cp -r  tr$SGE_TASK_ID $TMPDIR
# Change directory to the appropriate folder
cd $TMPDIR/tr$SGE_TASK_ID

# Execute the job script
chmod +x cavity.bash
./cavity.bash

cp $TMPDIR/tr$SGE_TASK_ID/* $SGE_O_WORKDIR/tr$SGE_TASK_ID

