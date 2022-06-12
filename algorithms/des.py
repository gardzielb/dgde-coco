import math

import numpy as np


class DES:
	def do(self, pop: np.ndarray, n_select):
		# number of random individuals needed
		n_random = n_select * 3

		# number of permutations needed
		n_perms = math.ceil(n_random / len(pop))

		# get random permutations and reshape them
		P = random_permuations(n_perms, len(pop))[:n_random]

		return np.reshape(P, (n_select, 3))


def random_permuations(n, l):
	perms = []
	for i in range(n):
		perms.append(np.random.permutation(l))
	P = np.concatenate(perms)
	return P
