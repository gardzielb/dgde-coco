# This file contains modified code from Pymoo repository,
# which is distributed according to license included in pymoo_license.txt file

import math

import numpy as np


def differential_selection(population: np.ndarray, n_select):
	# number of random individuals needed
	n_random = n_select * 3

	# number of permutations needed
	n_perms = math.ceil(n_random / len(population))

	# get random permutations and reshape them
	P = _random_permuations(n_perms, len(population))[:n_random]

	return np.reshape(P, (n_select, 3))


def _random_permuations(perm_count, perm_len):
	perms = []
	for i in range(perm_count):
		perms.append(np.random.permutation(perm_len))
	P = np.concatenate(perms)
	return P
