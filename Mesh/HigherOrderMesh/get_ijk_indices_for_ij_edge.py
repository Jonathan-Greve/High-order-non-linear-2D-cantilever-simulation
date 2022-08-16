import numpy as np


def get_ijk_indices_for_ij_edge(ijk_indices):
    ij_edge_pattern = []
    for (i,j,k) in ijk_indices:
        if (i > 0 or j > 0) and k == 0:
            ij_edge_pattern.append((i,j,k))

    return np.array(ij_edge_pattern)