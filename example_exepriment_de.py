#!/usr/bin/env python
from __future__ import division, print_function
import cocoex, cocopp  # experimentation and post-processing modules
import scipy.optimize  # to define the solver to be benchmarked
import os, webbrowser  # to show post-processed results in the browser

### input
from algorithms.minimize import minimize

suite_name = "bbob"
output_folder = "de-optimize-fmin"
fmin = scipy.optimize.fmin
budget_multiplier = 1  # increase to 10, 100, ...

### prepare
suite = cocoex.Suite(suite_name, "", "")
observer = cocoex.Observer(suite_name, "result_folder: " + output_folder)
minimal_print = cocoex.utilities.MiniPrint()

### go
for problem in suite:  # this loop will take several minutes or longer
    problem.observe_with(observer)  # generates the data for cocopp post-processing
    # apply restarts while neither the problem is solved nor the budget is exhausted
    while (problem.evaluations < problem.dimension * budget_multiplier
           and not problem.final_target_hit):
        minimize(problem)
    minimal_print(problem, final=problem.index == len(suite) - 1)

### post-process data
cocopp.main(observer.result_folder)  # re-run folders look like "...-001" etc
webbrowser.open("file://" + os.getcwd() + "/ppdata/index.html")

