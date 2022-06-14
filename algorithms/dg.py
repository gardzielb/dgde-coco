from enum import Enum

import numpy as np


class DGMode(Enum):
	NONE = 0
	EXPLORE = 1
	EXPLOIT = 2


class DGController:
	def __init__(self, div_low: float, div_high: float):
		assert div_low <= div_high, "div_low must not be greater than div_high"
		self.div_low = div_low
		self.div_high = div_high
		self.mode = DGMode.EXPLOIT
		self.pop_mean = None

	def calculate_mode(self, population: np.ndarray, lower_bound, upper_bound) -> DGMode:
		self.pop_mean = np.mean(population, axis = 0)  # average population point
		diversity = __calculate_diversity__(population, self.pop_mean, lower_bound, upper_bound)
		if diversity < self.div_low:
			self.mode = DGMode.EXPLORE
		elif diversity > self.div_high:
			self.mode = DGMode.EXPLOIT
		return self.mode


def __calculate_diversity__(population: np.ndarray, pop_mean: np.ndarray, lower_bound, upper_bound) -> float:
	pop_size = len(population)
	diag = np.linalg.norm(lower_bound - upper_bound)
	coefficient = 1 / (pop_size + diag)

	diffs = (population - pop_mean) ** 2

	sqrts = np.sqrt(np.sum(diffs))
	total = np.sum(sqrts)
	result = coefficient * total
	return result
