import cocoex
import numpy as np
import tkinter
from algorithms.de import DE
from algorithms.population_init import sampling_lhs


def minimize(problem: cocoex.Problem, x0: np.ndarray | None = None, pop_size: int = 0, gen_limit = 20):
	if x0 is None:
		xl = problem.lower_bounds
		xu = problem.upper_bounds
		x0 = sampling_lhs(pop_size, problem.dimension, xl, xu)

	pop_size = len(x0)
	de = DE(problem, pop_size, initial_population = x0)

	gen_count = 0
	while gen_count < gen_limit:
		infills = de.infill()
		de.advance(infills)
		gen_count += 1
