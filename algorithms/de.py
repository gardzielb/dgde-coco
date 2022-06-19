# This file contains modified code from Pymoo repository,
# which is distributed according to license included in pymoo_license.txt file

from typing import Callable, Optional

import cocoex
import numpy as np

from algorithms.des import differential_selection
from algorithms.dex import differential_crossover_and_mutation
from algorithms.dg import DGController


# noinspection PyPep8Naming
class DE:
	def __init__(self,
				 problem: cocoex.Problem | Callable[[np.ndarray], float],
				 pop_size = 100,
				 CR = 0.5,
				 F = 0.9,
				 initial_population: np.ndarray | None = None,
				 dg_controller: Optional[DGController] = None
				 ):
		"""
		Parameters
		----------
		pop_size : {pop_size}
		F : float
		 The F to be used during the crossover.
		CR : float
		 The probability the individual exchanges variable values from the donor vector.
		"""
		# parse the information from the string

		self.population = initial_population
		self.problem = problem
		self.F = F
		self.CR = CR

		self.pop_size = pop_size
		self.prev_pop_f = np.full((pop_size,), np.inf, dtype = float)

		for i in range(len(self.population)):
			self.prev_pop_f[i] = self.problem(self.population[i])

		self.dg_controller = dg_controller

	def infill(self):
		# how many parents need to be select for the mating - depending on number of offsprings remaining
		n_select = len(self.population)

		# this return an array of 3-tuples of indexes j, k, l that will later take part in differential mutation
		# p_j + F*(p_k-p_l)
		parents = differential_selection(self.population, n_select)

		# do the mutation and crossover using the parents index and the population array
		infills = differential_crossover_and_mutation(self.population, parents, self.F, self.CR, self.dg_controller,
													  self.problem.lower_bounds, self.problem.upper_bounds,
													  at_least_once = False)

		return infills

	def advance(self, infills = None):
		assert infills is not None, "This algorithms uses the AskAndTell interface thus infills must to be provided."

		# replace the individuals with the corresponding parents from the mating
		self.population = self.__succession(self.problem, self.population, infills)

	def __succession(self, problem: cocoex.Problem | Callable[[np.ndarray], float], population: np.ndarray,
					 infills: np.ndarray):
		pop_size = len(population)
		off_selection = np.full((pop_size, 1), False)
		off_f = np.full((pop_size,), np.inf, dtype = float)

		for i in range(len(population)):
			off_f[i] = problem(infills[i])

		off_selection[off_f < self.prev_pop_f] = True
		off_selection = off_selection[:, 0]
		population = population.copy()
		population[off_selection] = infills[off_selection]
		self.prev_pop_f[off_selection] = off_f[off_selection]

		return population
