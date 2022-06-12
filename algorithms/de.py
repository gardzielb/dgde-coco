import random
from typing import Callable

import cocoex
import numpy as np

from algorithms.des import DES
from algorithms.dex import DEX


class DE:

	def __init__(self,
				 problem: cocoex.Problem | Callable[[np.ndarray], float],
				 pop_size = 100,
				 n_offsprings = None,
				 CR = 0.5,
				 F = 0.9,
				 initial_population: np.ndarray | None = None,
				 seed = None
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

		self.pop = initial_population
		self.problem = problem
		self.selection = DES()

		self.crossover = DEX(F = F,
							 CR = CR,
							 at_least_once = False)

		self.seed = seed
		if self.seed is None:
			self.seed = np.random.randint(0, 10000000)
		# set the random seed for Python and Numpy methods
		random.seed(self.seed)
		np.random.seed(self.seed)

		self.pop_size = pop_size
		self.n_offsprings = n_offsprings

		self.prev_pop_f = np.full((pop_size,), np.inf, dtype = float)

		for i in range(len(self.pop)):
			self.prev_pop_f[i] = self.problem(self.pop[i])

	def infill(self):
		# how many parents need to be select for the mating - depending on number of offsprings remaining
		n_select = len(self.pop)

		# this return an array of 3-tuples of indexes j, k, l that will take part in differential mutation
		# p_j + F*(p_k-p_l)
		parents = self.selection.do(self.pop, n_select)

		# do the mutation and crossover using the parents index and the population array
		infills = self.crossover.do(self.pop, parents)

		return infills

	def advance(self, infills = None):
		assert infills is not None, "This algorithms uses the AskAndTell interface thus infills must to be provided."

		# replace the individuals with the corresponding parents from the mating
		self.pop = self.__succession(self.problem, self.pop, infills)

	def __succession(self, problem: cocoex.Problem | Callable[[np.ndarray], float], pop: np.ndarray,
					 infills: np.ndarray):
		n = len(pop)
		ret = np.full((n, 1), False)
		off_f = np.full((n,), np.inf, dtype = float)

		for i in range(len(pop)):
			off_f[i] = problem(infills[i])

		ret[off_f < self.prev_pop_f] = True
		I = ret[:, 0]
		pop = pop.copy()
		pop[I] = infills[I]
		self.prev_pop_f[I] = off_f[I]

		return pop
