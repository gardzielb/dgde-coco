import numpy as np

from algorithms.binx import binomial_crossover


def de_differential(X, F):
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

class DEX:
	def __init__(self,
				 F = 0.8,
				 CR = 0.9,
				 n_iter = 1,
				 at_least_once = True):
		self.F = F
		self.CR = CR
		self.at_least_once = at_least_once
		self.n_iter = n_iter

	def do(self, pop, parents):
		X = pop[parents.T].copy()
		assert len(X.shape) == 3, "Please provide a three-dimensional matrix n_parents x pop_size x n_vars."

		_, n_matings, n_var = X.shape

		# prepare the out to be set
		Xp = de_differential(X, self.F)

		M = binomial_crossover(n_matings, n_var, self.CR, at_least_once = self.at_least_once)

		# take the first parents (this is already a copy)
		X = X[0]

		# set the corresponding values from the donor vector
		X[M] = Xp[M]

		return X
