#!/usr/bin/env python
from __future__ import division, print_function

import argparse
import multiprocessing
import os
import webbrowser  # to show post-processed results in the browser

# experimentation and post-processing modules
import cocoex
import cocopp

from algorithms.minimize import minimize_de, minimize_dgde

# input
suite_name = "bbob"
budget_multiplier = 1  # increase to 10, 100, ...

# prepare
suite = cocoex.Suite(suite_name, "", "")
minimal_print = cocoex.utilities.MiniPrint()


def create_observer(output_folder):
	return cocoex.Observer(suite_name, "result_folder: " + output_folder)

parser = argparse.ArgumentParser(description='Benchmark dgde.')
group = parser.add_mutually_exclusive_group()
group.add_argument('--dgde-only', action = 'store_true')
group.add_argument('--de-only', action = 'store_true')
parser.add_argument('-v', '--verbose', action = 'store_true')

args = parser.parse_args()
if args.dgde_only:
	alg_data = [(minimize_dgde, 'DGDE')]
elif args.de_only:
	alg_data = [(minimize_de, 'DE')]
else:
	alg_data = [
		(minimize_de, 'DE'),
		(minimize_dgde, 'DGDE')
	]

algorithms = [(minimize, create_observer(output_folder)) for minimize, output_folder in alg_data]

# go
pp_args = '--include-single'
for minimize, observer in algorithms:
	for problem in suite: #type: cocoex.Problem  # this loop will take several minutes or longer
		problem.observe_with(observer)  # generates the data for cocopp post-processing
		# apply restarts while neither the problem is solved nor the budget is exhausted
		while problem.evaluations < problem.dimension * budget_multiplier and not problem.final_target_hit:
			minimize(problem, gen_limit = 50 * problem.dimension, pop_size = 100)
		if args.verbose:
			minimal_print(problem, final = problem.index == len(suite) - 1)

	pp_args += f' {observer.result_folder}'

# post-process data
if not args.dgde_only and not args.de_only:
	cocopp.main(pp_args)  # re-run folders look like "...-001" etc
	webbrowser.open("file://" + os.getcwd() + "/ppdata/index.html")
