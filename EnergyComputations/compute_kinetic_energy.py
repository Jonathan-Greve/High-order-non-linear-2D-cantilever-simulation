import numpy as np


def compute_kinetic_energy(m, v):
    """
    Computes the kinetic energy of the mesh.
    """
    vx = v[np.arange(0, len(v) - 1, 2)]
    vy = v[np.arange(1, len(v), 2)]

    return 0.5 * m * (np.dot(vx, vx) + np.dot(vy, vy))