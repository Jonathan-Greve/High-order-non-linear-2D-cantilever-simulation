import numpy as np

from Mesh.HigherOrderMesh.generate_ijk_indices import generate_ijk_indices
from Mesh.HigherOrderMesh.get_ijk_indices_for_internal_nodes import \
    get_ijk_indices_for_internal_nodes


def decode_triangle_indices(encoding, n):
    """
    Decodes the triangle indices from the encoding.
    :param encoding:
    :param n:
    :return:
    """

    m = (n + 1) * (n + 2) / 2

    global_indices = []
    ijk_indices = [[n,0,0], [0,n,0], [0,0,n]]

    # Corner node indices
    i = encoding[0]
    j = encoding[1]
    k = encoding[2]
    global_indices.extend([i, j, k])

    # Internal node indices
    num_internal_nodes = int((n - 2) * (n-1) / 2)
    for l in range(num_internal_nodes):
        global_indices.append(encoding[3]+l)
    ijk_indices.extend(get_ijk_indices_for_internal_nodes(generate_ijk_indices(n)))

    # ij-edge indices (if any)
    num_edge_nodes = n - 1
    if n >= 2:
        # Indices for ij edge
        ij_edge_indices = []
        ij_edge_ijk_indices = []
        for l in range(num_edge_nodes):
            ij_edge_indices.append(encoding[4] + l)
            ij_edge_ijk_indices.append((n-l-1, l+1, 0))
        if encoding[5] == -1:
            ij_edge_indices = reversed(ij_edge_indices)
            ij_edge_ijk_indices = reversed(ij_edge_ijk_indices)
        global_indices.extend(ij_edge_indices)
        ijk_indices.extend(ij_edge_ijk_indices)

        # Indices for jk edge
        jk_edge_indices = []
        jk_edge_ijk_indices = []
        for l in range(num_edge_nodes):
            jk_edge_indices.append(encoding[6] + l)
            jk_edge_ijk_indices.append((0, n-l-1, l+1))
        if encoding[7] == -1:
            jk_edge_indices = reversed(jk_edge_indices)
            jk_edge_ijk_indices = reversed(jk_edge_ijk_indices)
        global_indices.extend(jk_edge_indices)
        ijk_indices.extend(jk_edge_ijk_indices)

        # Indices for jk edge
        ki_edge_indices = []
        ki_edge_ijk_indices = []
        for l in range(num_edge_nodes):
            ki_edge_indices.append(encoding[8] + l)
            ki_edge_ijk_indices.append((l+1, 0, n-l-1))
        if encoding[7] == -1:
            ki_edge_indices = reversed(ki_edge_indices)
            ki_edge_ijk_indices = reversed(ki_edge_ijk_indices)
        global_indices.extend(ki_edge_indices)
        ijk_indices.extend(ki_edge_ijk_indices)

    assert(m == len(global_indices))

    return np.array(global_indices), np.array(ijk_indices)


