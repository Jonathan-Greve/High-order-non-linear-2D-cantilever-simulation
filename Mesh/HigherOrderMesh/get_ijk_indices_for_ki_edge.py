import numpy as np


def get_ijk_indices_for_ki_edge(ijk_indices):
    ki_edge_pattern = []
    for (i, j, k) in ijk_indices:
        if (k > 0 or i > 0) and j == 0:
            ki_edge_pattern.append((i, j, k))

    return np.array(ki_edge_pattern)