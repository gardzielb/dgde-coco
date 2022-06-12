import numpy as np

def row_at_least_once_true(M):
    _, d = M.shape
    for k in np.where(~np.any(M, axis=1))[0]:
        M[k, np.random.randint(d)] = True
    return M

def binomial_crossover(n, m, prob, at_least_once=True):
    M = np.random.random((n, m)) < prob

    if at_least_once:
        M = row_at_least_once_true(M)

    return M
