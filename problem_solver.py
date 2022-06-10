from abc import ABC, abstractmethod

import numpy
from cocoex import Problem


class ProblemSolver(ABC):
	@abstractmethod
	def solve(self, objective_function: Problem, x0: numpy.ndarray):
		pass
