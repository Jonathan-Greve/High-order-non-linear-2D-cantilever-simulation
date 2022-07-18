import numpy as np
import scipy.spatial as spatial

def generate_2d_cantilever_delaunay(beam_length, beam_width, num_vertices_x_dir, num_vertices_y_dir):
    """
    Generate a 2D mesh using Delaunay triangulation.

    :param beam_length: Length of the beam.
    :param beam_width: Width of the beam.
    :param num_vertices_x_dir: Number of vertices in x direction.
    :param num_vertices_y_dir: Number of vertices in y direction.

    :return: A list of vertices and a list of faces.
    """

    # Generate a 2D mesh using Delaunay triangulation.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html
    x = np.linspace(0, beam_length, num_vertices_x_dir)  # x-coordinates
    y = np.linspace(0, beam_width, num_vertices_y_dir)  # y-coordinates
    xv, yv = np.meshgrid(x, y)  # x-y coordinates
    xv = xv.flatten()
    yv = yv.flatten()
    points = np.vstack((xv, yv)).T
    tri = spatial.Delaunay(points)

    tri.points[:, 0] = tri.points[:, 0] - beam_length / 2.0
    tri.points[:, 1] = tri.points[:, 1] - beam_width / 2.0

    return tri.points, tri.simplices