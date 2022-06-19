import cocoex
import numpy as np
from algorithms.de import DE
from algorithms.dg import DGController
from algorithms.population_init import sampling_lhs


def minimize_de(problem: cocoex.Problem, x0: np.ndarray | None = None, pop_size: int = 0, gen_limit = 20):
	x0 = prepare_x0(problem, x0, pop_size)
	de = DE(problem, pop_size = len(x0), initial_population = x0)
	run_de(de, gen_limit)


def minimize_dgde(
		div_low: float, div_high: float, problem: cocoex.Problem, x0: np.ndarray | None = None,
		pop_size: int = 0, gen_limit = 20
):
	x0 = prepare_x0(problem, x0, pop_size)
	de = DE(problem, pop_size = len(x0), initial_population = x0, dg_controller = DGController(div_low, div_high))
	run_de(de, gen_limit)


def prepare_x0(problem: cocoex.Problem, x0: np.ndarray | None = None, pop_size: int = 0):
	if x0:
		return x0

	xl = problem.lower_bounds
	xu = problem.upper_bounds
	return sampling_lhs(pop_size, problem.dimension, xl, xu)


def run_de(de: DE, gen_limit = 20):
	gen_count = 0
	while gen_count < gen_limit:
		infills = de.infill()
		de.advance(infills)
		gen_count += 1
