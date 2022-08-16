import numpy as np


def get_ijk_indices_for_jk_edge(ijk_indices):
    jk_edge_pattern = []
    for (i,j,k) in ijk_indices:
        if (j > 0 or k > 0) and i == 0:
            jk_edge_pattern.append((i,j,k))

    return np.array(list(reversed(jk_edge_pattern)))