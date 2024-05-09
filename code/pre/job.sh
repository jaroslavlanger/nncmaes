#!/bin/bash
#PBS -l select=1:ncpus=1:mem=1gb:scratch_local=1gb
#PBS -l walltime=24:00:00

echo "python version: $(python3 --version)"

# define a DATADIR variable: directory where the input files are taken from and where output will be copied to
DATADIR="/storage/brno2/home/langera/test05/d2_f${f_id}_i${i_id}" # substitute username and path to to your real username and path
export PYTHONUSERBASE="/storage/brno2/home/langera/.local"
export PATH="$PYTHONUSERBASE/bin:$PATH"
export PYTHONPATH="$PYTHONUSERBASE/lib/python3.9/site-packages:$PYTHONPATH"

mkdir -p $DATADIR
# append a line to a file "jobs_info.txt" containing the ID of the job, the hostname of node it is run on and the path to a scratch directory
# this information helps to find a scratch directory in case the job fails and you need to remove the scratch directory manually
echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs_info.txt

# test if scratch directory is set
# if scratch directory is not set, issue error message and exit
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

# move into scratch directory
cd $SCRATCHDIR

cp "/storage/brno2/home/langera/experiment.py" .
cp "/storage/brno2/home/langera/cmaes_surrogate.py" .

{ time -p python3 experiment.py $f_id $i_id 2> $DATADIR/exp.err > $DATADIR/exp.out ; } 2>&1 | tr '\n' ',' | tr -d ' ' | xargs printf "$(hostname -f),%s,$PBS_JOBID\n" >> $DATADIR/time-perf.txt

# move the output to user's DATADIR or exit in case of failure
cp -r exdata/* $DATADIR/ || { echo >&2 "Result file(s) copying failed (with a code $?) !!"; exit 4; }

# clean the SCRATCH directory
# clean_scratch
