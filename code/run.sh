#!/bin/bash
#PBS -l select=1:ncpus=1:mem=1gb:scratch_local=1gb
#PBS -l walltime=24:00:00
N_TEST="003-a" # Kendall threshold 0.95

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

if [ -z ${inst+x} ]; then
    inst_name=""
    inst_param=""
else
    inst_name=$(printf "_i%02d" $inst)
    inst_param="--inst ${inst}"
fi

# storage root for the scripts and outputs
[ -z ${STORAGE+x} ] && STORAGE="/storage/brno2"
# directory for outputs to be copied to
DATADIR="${STORAGE}/home/${USER}/test${N_TEST}/${crit}"
exp_name=$(printf "d%02d_f%02d%s_%s" "$dim" "$fun" "$inst_name" "$crit")
experiment="${STORAGE}/home/${USER}/experiment.py"

mkdir -p $DATADIR || { echo >&2 "Creating ${DATADIR} failed!"; exit 1; }

echo \
    "${PBS_JOBID}-job" \
    "$(hostname -f)-node" \
    "${SCRATCHDIR}-scratch" \
    "${DATADIR}-data" \
    "$($PYTHON --version)-version $(which $PYTHON)-which" \
    "${exp_name}-experiment" \
    | tee -a $DATADIR/jobs_info.txt

test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 2; }

# go to scratch directory for __pycache__ and exdata
cd ${SCRATCHDIR}

OUT="${exp_name}.out"
if [ -z ${KEEP_ERR+x} ]; then
    KEEP_ERR=false
fi
if ${KEEP_ERR}; then
    ERR="${exp_name}.err"
else
    ERR=/dev/null
fi

{ time -p ${PYTHON} -O ${experiment} --dim ${dim} --fun ${fun} ${inst_param} --crit ${crit} 2> ${ERR} > ${OUT} ; } 2>&1 \
    | tr '\n' ',' \
    | tr -d ' ' \
    | xargs printf "$(hostname -f),%s,$PBS_JOBID,${exp_name}\n" \
    >> ${DATADIR}/time-perf.txt \
    || { echo >&2 "error from python script (code: $?)!"; exit 3; }

if ${KEEP_ERR}; then
    cp ${ERR} ${DATADIR}
fi
cp -r exdata/* ${OUT} ${DATADIR} || { echo >&2 "Result file(s) copying failed (with a code $?)!"; exit 4; }
# clean the SCRATCH directory
clean_scratch
