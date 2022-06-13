from enum import Enum

import numpy as np

class DGMode(Enum):
	NONE = 0
	EXPLORE = 1
	EXPLOIT = 2

def calculate_diversity(pop: np.ndarray, lower_bound, upper_bound) -> float:
	pop_size = len(pop)
	diag = np.linalg.norm(lower_bound - upper_bound)
	coefficient = 1 / (pop_size + diag)

	s_mean = np.mean(pop, axis = 0) # average population point
	diffs = (pop - s_mean) ** 2

	sqrts = np.sqrt(np.sum(diffs))
	total = np.sum(sqrts)
	result = coefficient * total
	return result
