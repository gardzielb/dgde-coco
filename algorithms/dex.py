import numpy as np

from algorithms.dg import DGMode, DGController


def _differential_mutation(X, F):
	n_parents, n_matings, n_var = X.shape
	assert n_parents % 2 == 1, "For the differential an odd number of values need to be provided"

	# make sure F is a one-dimensional vector
	F = np.ones(n_matings) * F

	# perform mutation according to DE eqaution
	return X[0] + F[:, None] * (X[1] - X[2])


def _dg_mutation(X, F, pop_mean: np.ndarray, dg_mode: DGMode):
	n_parents, n_matings, n_var = X.shape
	assert n_parents % 2 == 1, "For the differential an odd number of values need to be provided"

	# make sure F is a one-dimensional vector
	F = np.ones(n_matings) * F

	# calculate results of differential mutation for both diff options
	result1 = X[0] + F[:, None] * (X[1] - X[2])
	result2 = X[1] + F[:, None] * (X[2] - X[1])

	dist_to_mean_1 = np.sum((result1 - pop_mean) ** 2, axis = 1)[:, None]
	dist_to_mean_2 = np.sum((result2 - pop_mean) ** 2, axis = 1)[:, None]

	# select mutation result according to current DG mode
	choose1 = dist_to_mean_1 < dist_to_mean_2 if dg_mode == DGMode.EXPLOIT else dist_to_mean_1 < dist_to_mean_2
	return np.select([choose1, choose1 == False], [result1, result2])


def differential_crossover_and_mutation(population, parents, F, CR, dg_controller: DGController, lower_bound,
										upper_bound, at_least_once = True):
	X = population[parents.T].copy()
	assert len(X.shape) == 3, "Please provide a three-dimensional matrix n_parents x pop_size x n_vars."

	if dg_controller:
		dg_mode = dg_controller.calculate_mode(population, upper_bound, lower_bound)
		mutants = _dg_mutation(X, F, dg_controller.pop_mean, dg_mode)
	else:
		# prepare the out to be set
		mutants = _differential_mutation(X, F)

	_, offspring_size, problem_dim = X.shape
	cross_indices = _binomial_crossover(offspring_size, problem_dim, CR, at_least_once = at_least_once)

	# take the first parents (this is already a copy)
	X = X[0]

	# set the corresponding values from the donor vector
	X[cross_indices] = mutants[cross_indices]

	return X


def _row_at_least_once_true(M):
	_, d = M.shape
	for k in np.where(~np.any(M, axis = 1))[0]:
		M[k, np.random.randint(d)] = True
	return M


def _binomial_crossover(n, m, prob, at_least_once = True):
	M = np.random.random((n, m)) < prob

	if at_least_once:
		M = _row_at_least_once_true(M)

	return M
