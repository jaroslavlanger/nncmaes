#!/bin/bash
#PBS -l select=1:ncpus=1:mem=1gb:scratch_local=1gb
#PBS -l walltime=24:00:00
N_TEST="001"

if [ -z ${crit+x} ]; then
    crit="mean"
fi

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
DATADIR="${STORAGE}/home/${USER}/test${N_TEST}/${crit}"
exp_name="exp_${crit}_${dim}d_${fun}f_${inst}i"
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
    INST_FULL=""
else
    INST_FULL="--inst ${inst}"
fi

OUT="${exp_name}.out"
KEEP_ERR=false
if ${KEEP_ERR}; then
    ERR="${exp_name}.err"
else
    ERR=/dev/null
fi

{ time -p ${PYTHON} ${experiment} --dim ${dim} --fun ${fun} ${INST_FULL} --crit ${crit} 2> ${ERR} > ${OUT} ; } 2>&1 \
    | tr '\n' ',' \
    | tr -d ' ' \
    | xargs printf "$(hostname -f),%s,$PBS_JOBID,${exp_name}\n" \
    >> ${DATADIR}/time-perf.txt \
    || { echo >&2 "error from python script (code $?) !!"; exit 3; }

# move the output to user's DATADIR or exit in case of failure
cp -r exdata/* ${OUT} ${DATADIR} || { echo >&2 "Result file(s) copying failed (with a code $?) !!"; exit 4; }
if ${KEEP_ERR}; then
    cp ${ERR} ${DATADIR}
fi
# clean the SCRATCH directory
clean_scratch
