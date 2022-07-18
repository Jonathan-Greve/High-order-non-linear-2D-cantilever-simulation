import numpy as np
import quadpy

from Mesh.Cantilever.area_computations import compute_triangle_element_area
from Simulator.triangle_shape_functions import triangle_shape_function_i_helper


def compute_shape_function_volume(points, face):
    """
    Compute the integral of the shape function value for node i.
    :param points: Is a nx2 numpy array containing the x and y coordinates of the nodes.
    :param face: Is a 3x1 numpy array containing the indices of the nodes of the triangle.
    :return:
    """

    # Compute matrix using quadpy (quadpy is a quadrature package)
    triangle = points[face]

    # get a "good" scheme of degree 10. (Even 2 should be enough since be use linear elements
    # and we multiply them to get a second order polynomial)
    scheme = quadpy.t2.get_good_scheme(10)

    def N_i(x):
        return triangle_shape_function_i_helper(points, face, x)


    n_i = scheme.integrate(lambda x: N_i(x.astype(np.float64)), triangle)

    return n_i