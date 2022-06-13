import numpy as np

from algorithms.dg import calculate_diversity, DGMode


def _differential_mutation(X, F):
	n_parents, n_matings, n_var = X.shape
	assert n_parents % 2 == 1, "For the differential an odd number of values need to be provided"

	# make sure F is a one-dimensional vector
	F = np.ones(n_matings) * F

	# the differentials from substraction the selected each pair
	diffs = np.zeros((n_matings, n_var))
	diffs += F[:, None] * (X[1] - X[2])

	# now add the differentials to the first parent
	Xp = X[0] + diffs

	return Xp


def _dg_mutation(X, F, mode: DGMode, diversity):
	n_parents, n_matings, n_var = X.shape
	assert n_parents % 2 == 1, "For the differential an odd number of values need to be provided"

	# make sure F is a one-dimensional vector
	F = np.ones(n_matings) * F

	# the differentials from substraction the selected each pair
	diffs1 = np.zeros((n_matings, n_var))
	diffs2 = np.zeros((n_matings, n_var))

	diffs1 += F[:, None] * (X[1] - X[2])
	diffs2 += F[:, None] * (X[2] - X[1])

	# TODO: do something with those diffs

	# now add the differentials to the first parent
	# Xp = X[0] + diffs1

	# return Xp


def differential_crossover_and_mutation(pop, parents, F, CR, dg_mode: DGMode, lower_bound, upper_bound,
										at_least_once = True):
	X = pop[parents.T].copy()
	assert len(X.shape) == 3, "Please provide a three-dimensional matrix n_parents x pop_size x n_vars."

	_, n_matings, n_var = X.shape

	if dg_mode is not DGMode.NONE:
		diversity = calculate_diversity(pop, lower_bound, upper_bound)
		Xp = _dg_mutation(X, F, dg_mode, diversity)
	else:
		# prepare the out to be set
		Xp = _differential_mutation(X, F)

	M = _binomial_crossover(n_matings, n_var, CR, at_least_once = at_least_once)

	# take the first parents (this is already a copy)
	X = X[0]

	# set the corresponding values from the donor vector
	X[M] = Xp[M]

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
