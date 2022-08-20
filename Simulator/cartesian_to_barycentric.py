import numpy as np


def cartesian_to_barycentric(x, V_triangle):
    """
    Convert the cartesian coordinates of a point to barycentric coordinates.
    :param x:
    :param V_triangle:
    :return barycentric_coordinates:
    """

    V_ix, V_iy = V_triangle[0]
    V_jx, V_jy = V_triangle[1]
    V_kx, V_ky = V_triangle[2]

    # T
    T = np.array([
        [V_ix - V_kx, V_jx - V_kx],
        [V_iy - V_ky, V_jy - V_ky]
    ])

    # Get the first 2 barycentric coordinates
    xi_1, xi_2 = np.linalg.inv(T) @ np.array([x[0] - V_kx, x[1] - V_ky])
    xi_3 = 1.0 - xi_1 - xi_2

    return np.array([xi_1, xi_2, xi_3])



