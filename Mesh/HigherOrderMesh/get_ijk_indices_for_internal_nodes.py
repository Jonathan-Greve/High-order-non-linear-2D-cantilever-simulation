import numpy as np


def get_ijk_indices_for_internal_nodes(ijk_indices):
    ki_edge_pattern = []
    for (i,j,k) in ijk_indices:
        if i > 0 and j > 0 and k > 0:
            ki_edge_pattern.append((i,j,k))

    return np.array(ki_edge_pattern)