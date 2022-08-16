import numpy as np


def generate_ijk_indices(n):
    """
    Generate the (i,j,k) for a higher order mesh element.
    :param n:
    :return:
    """
    ijk_indices = []
    for r in range(n + 1):
        for j in range(r + 1):
            i = r - j
            k = n - r
            ijk_indices.append((i, j, k))
    return np.array(ijk_indices)