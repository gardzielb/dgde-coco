# This file contains modified code from Pymoo repository,
# which is distributed according to license included in pymoo_license.txt file

import numpy as np
import scipy


def cdist(A, B, **kwargs):
	return scipy.spatial.distance.cdist(A.astype(float), B.astype(float), **kwargs)


def criterion_maxmin(X):
	D = cdist(X, X)
	np.fill_diagonal(D, np.inf)
	return np.min(D)


def criterion_corr(X):
	M = np.corrcoef(X.T, rowvar = True)
	return -np.sum(np.tril(M, -1) ** 2)


def sampling_lhs(n_samples, n_var, xl = 0, xu = 1, smooth = True, criterion = criterion_corr, n_iter = 50):
	X = sampling_lhs_unit(n_samples, n_var, smooth = smooth)

	# if a criterion is selected to further improve the sampling
	if criterion is not None:

		# current best score is stored here
		score = criterion(X)

		for j in range(1, n_iter):

			# create new random sample and check the score again
			_X = sampling_lhs_unit(n_samples, n_var, smooth = smooth)
			_score = criterion(_X)

			if _score > score:
				X, score = _X, _score

	return xl + X * (xu - xl)


def sampling_lhs_unit(n_samples, n_var, smooth = True):
	X = np.random.random(size = (n_samples, n_var))
	Xp = X.argsort(axis = 0) + 1

	if smooth:
		Xp = Xp - np.random.random(Xp.shape)
	else:
		Xp = Xp - 0.5
	Xp /= n_samples
	return Xp
