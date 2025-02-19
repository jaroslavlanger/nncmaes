#!/usr/bin/env python
"""A short and simple example experiment with restarts.

The code is fully functional but mainly emphasises on readability.
Hence produces only rudimentary progress messages and does not provide
batch distribution or timing prints, as `example_experiment2.py` does.

To apply the code to a different solver, `fmin` must be re-assigned or
re-defined accordingly. For example, using `cma.fmin` instead of
`scipy.optimize.fmin` can be done like::

>>> import cma  # doctest:+SKIP
>>> def fmin(fun, x0):
...     return cma.fmin(fun, x0, 2, {'verbose':-9})

"""
from __future__ import division, print_function
import os
os.environ["OMP_NUM_THREADS"] =        "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] =   "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] =        "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] =    "1" # export NUMEXPR_NUM_THREADS=6
import torch
torch.set_num_threads(1)
import cocoex, cocopp  # experimentation and post-processing modules
import scipy.optimize  # to define the solver to be benchmarked
from numpy.random import rand  # for randomised restarts
import webbrowser  # to show post-processed results in the browser
from cmaes_surrogate import ProblemCocoex, cmaes_safe, eaf, raf
import sys
import argparse

### input
parser = argparse.ArgumentParser(description='CMAES with surrogate experiment')
parser.add_argument('--dim', type=int, choices=[2, 3, 5, 10, 20, 40], required=True,
                    help='Input dimensionality')
parser.add_argument('--fun', type=int, choices=list(range(1, 24 + 1)), required=True,
                    help='Function id')
parser.add_argument('--inst', nargs='+', type=int, choices=list(range(1, 15 + 1)),
                    help='Instance suite numbers')
surrogates = {mod.__name__: mod for mod in [eaf, raf]}
parser.add_argument('--surr', choices=list(surrogates.keys()), default=eaf.__name__,
                    help='surrogate model')
args = parser.parse_args()
print(f"{args = }")
DIMENSION = args.dim
FUNCTION = args.fun
INSTANCE = ','.join([str(i) for i in sorted(set(args.inst))]) if args.inst is not None else None
model = surrogates[args.surr]
suite_name = "bbob"
budget_multiplier = 250
suite_options = " ".join(
    [f"dimensions: {DIMENSION}",
     f"function_indices: {FUNCTION}",
    ] + (
    [f"instance_indices: {INSTANCE}"] if INSTANCE is not None else [])
)
print(f"{suite_options = }")

output_folder = "_".join(["cmaes_safe",
                         f"{DIMENSION:0>2}d",
                         f"{FUNCTION:0>2}f"] + (
                        [f"{INSTANCE:0>3}i"] if INSTANCE is not None else []
                    ) + [f"{budget_multiplier}"])
# fmin = scipy.optimize.fmin

### prepare
suite = cocoex.Suite(suite_name, "", suite_options)
observer = cocoex.Observer(suite_name, "result_folder: " + output_folder)
minimal_print = cocoex.utilities.MiniPrint()

### go
for problem in suite:  # this loop will take several minutes or longer
    problem.observe_with(observer)  # generates the data for cocopp post-processing
    x0 = problem.initial_solution
    # apply restarts while neither the problem is solved nor the budget is exhausted
    budget = problem.dimension * budget_multiplier
    while (problem.evaluations < budget and not problem.final_target_hit):
        cmaes_safe(ProblemCocoex(problem), budget=budget, model=model, log=True)
        # fmin(problem, x0, disp=False)  # here we assume that `fmin` evaluates the final/returned solution
        x0 = problem.lower_bounds + ((rand(problem.dimension) + rand(problem.dimension)) *
                    (problem.upper_bounds - problem.lower_bounds) / 2)
    minimal_print(problem, final=problem.index == len(suite) - 1)

### post-process data
# cocopp.main(observer.result_folder)  # re-run folders look like "...-001" etc
# webbrowser.open("file://" + os.getcwd() + "/ppdata/index.html")
