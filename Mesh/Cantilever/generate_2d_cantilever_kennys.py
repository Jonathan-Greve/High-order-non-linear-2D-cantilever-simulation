import numpy as np
import scipy.spatial as spatial

def generate_2d_cantilever_kennys(beam_length, beam_width, num_vertices_x_dir, num_vertices_y_dir):
    x0 = -beam_length / 2.0
    y0 = -beam_width / 2.0

    shape = (num_vertices_x_dir-1, num_vertices_y_dir-1)
    I = shape[0]
    J = shape[1]
    dx = beam_length / float(I)
    dy = beam_width / float(J)
    V = np.zeros(((I + 1) * (J + 1), 2), dtype=np.float64)
    for j in range(J + 1):
        for i in range(I + 1):
            k = i + j * (I + 1)
            V[k, 0] = x0 + i * dx
            V[k, 1] = y0 + j * dy
    T = np.zeros((2 * I * J, 3), dtype=np.int32)
    for j in range(J):
        for i in range(I):
            k00 = (i) + (j) * (I + 1)
            k01 = (i + 1) + (j) * (I + 1)
            k10 = (i) + (j + 1) * (I + 1)
            k11 = (i + 1) + (j + 1) * (I + 1)
            e = 2 * (i + j * I)
            if (i + j) % 2:
                T[e, :] = (k00, k01, k11)
                T[e + 1, :] = (k00, k11, k10)
            else:
                T[e, :] = (k10, k00, k01)
                T[e + 1, :] = (k10, k01, k11)
    return V, T