#!/bin/bash
#PBS -l select=1:ncpus=1:mem=1gb:scratch_local=1gb
#PBS -l walltime=24:00:00

# path to the python executable
if [ -z ${PYTHON+x} ]; then
    PYTHON=python3
    export PYTHONUSERBASE="/storage/brno2/home/langera/.local"
    export PATH="$PYTHONUSERBASE/bin:$PATH"
    export PYTHONPATH="$PYTHONUSERBASE/lib/python3.9/site-packages:$PYTHONPATH"
fi
# storage root for the scripts and outputs
[ -z ${STORAGE+x} ] && STORAGE="/storage/brno2"
# directory for outputs to be copied to
DATADIR="${STORAGE}/home/${USER}/test07/${SURROGATE}"
exp_name="exp_${SURROGATE}_${dim}d_${fun}f_${inst}i"
experiment="${STORAGE}/home/${USER}/experiment.py"

mkdir -p $DATADIR || { echo >&2 "Creating ${DATADIR} failed!"; exit 1; }

echo \
    "${PBS_JOBID}-job" \
    "$(hostname -f)-node" \
    "${SCRATCHDIR}-scratch" \
    "${DATADIR}-data" \
    "$($PYTHON --version)-version $(which $PYTHON)-which" \
    "${exp_name}-experiment" \
    | tee $DATADIR/jobs_info.txt

test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 2; }

# go to scratch directory for __pycache__ and exdata
cd ${SCRATCHDIR}

if [ -z ${inst+x} ]; then
    INST=""
else
    INST="--inst ${inst}"
fi

OUT="${exp_name}.out"
# keep std err
# TODO: if changed to false, handle the cp below
if true; then
    ERR="${exp_name}.err"
else
    ERR=/dev/null
fi

{ time -p ${PYTHON} ${experiment} --dim ${dim} --fun ${fun} ${INST} --surr ${SURROGATE} 2> ${ERR} > ${OUT} ; } 2>&1 \
    | tr '\n' ',' \
    | tr -d ' ' \
    | xargs printf "$(hostname -f),%s,$PBS_JOBID,${exp_name}\n" \
    >> ${DATADIR}/time-perf.txt \
    || { echo >&2 "error from python script (code $?) !!"; exit 3; }

# move the output to user's DATADIR or exit in case of failure
cp -r exdata/* ${OUT} ${ERR} "${DATADIR}/" || { echo >&2 "Result file(s) copying failed (with a code $?) !!"; exit 4; }

# clean the SCRATCH directory
clean_scratch
