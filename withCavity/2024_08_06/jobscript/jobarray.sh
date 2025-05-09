#!/bin/bash
#$ -N jobname_1
#$ -t 1-1000
##$ cwd
#$ -o output.log
#$ -e error.log

cp -r  tr$SGE_TASK_ID $TMPDIR
# Change directory to the appropriate folder
cd $TMPDIR/tr$SGE_TASK_ID

# Execute the job script
chmod +x cavity.bash
./cavity.bash

# Copy the files from temporary directory to original directory
cp -r $TMPDIR/tr$SGE_TASK_ID/* $SGE_O_WORKDIR/tr$SGE_TASK_ID


