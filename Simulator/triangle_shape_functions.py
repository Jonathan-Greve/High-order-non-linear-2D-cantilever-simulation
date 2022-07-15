import numpy as np

from Mesh.Cantilever.area_computations import compute_triangle_element_area

def triangle_shape_function_i_helper(points, face, x, face_index=0):
    """
    Compute the shape function value for node i.
    :param x: Is a 2x1 numpy array containing the x and y coordinates of the point.
    :return:
    """

    # Get the triangle element
    triangle_face = face
    triangle_element = points[triangle_face]

    # Compute the area of the triangle
    A_e = compute_triangle_element_area(points, triangle_face)

    # Get element points
    p_j = triangle_element[1]
    p_k = triangle_element[2]

    # Compute vectors between nodes
    p_jk = p_k - p_j
    p_jp = [x[0] - p_j[0], x[1] - p_j[1]]

    val = (p_jk[0] * p_jp[1] - p_jk[1] * p_jp[0]) / (2.0 * A_e)

    return val

def triangle_shape_function_j_helper(points, face, x):
    """
    Compute the shape function value for node j.
    :param x: Is a 2x1 numpy array containing the x and y coordinates of the point.
    :return:
    """

    # Get the triangle element
    triangle_face = face
    triangle_element = points[triangle_face]

    # Compute the area of the triangle
    A_e = compute_triangle_element_area(points, triangle_face)

    # Get element points
    p_i = triangle_element[0]
    p_k = triangle_element[2]

    # Compute vectors between nodes
    p_ki = p_i - p_k
    p_kp = [x[0] - p_k[0], x[1] - p_k[1]]

    val = (p_ki[0] * p_kp[1] - p_ki[1] * p_kp[0]) / (2.0 * A_e)

    return val

def triangle_shape_function_k_helper(points, face, x, face_index=0):
    """
    Compute the shape function value for node k.
    :param x: Is a 2x1 numpy array containing the x and y coordinates of the point.
    :return:
    """

    # Get the triangle element
    triangle_face = face
    triangle_element = points[triangle_face]

    # Compute the area of the triangle
    A_e = compute_triangle_element_area(points, triangle_face)

    # Get element points
    p_i = triangle_element[0]
    p_j = triangle_element[1]

    # Compute vectors between nodes
    p_ij = p_j - p_i
    p_ip = [x[0] - p_i[0], x[1] - p_i[1]]

    val = (p_ij[0] * p_ip[1] - p_ij[1] * p_ip[0]) / (2.0 * A_e)

    return val
