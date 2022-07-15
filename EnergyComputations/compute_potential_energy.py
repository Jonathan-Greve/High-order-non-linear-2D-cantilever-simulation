import numpy as np


def compute_potential_energy(m, vertices, gravity):
    """
    Computes the potential energy of the mesh.
    """

    ys = vertices[np.arange(1, len(vertices), 2)]
    sum_ys = np.sum(ys)
    return -gravity[1] * m * sum_ys