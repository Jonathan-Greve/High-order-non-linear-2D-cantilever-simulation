from Mesh.Cantilever.area_computations import compute_triangle_element_area


def iint_N_i_squared(points, faces, x, face_index=0):
    """
    Compute the integral of the squared shape function N_i.
    :param x: Is a 2x1 numpy array containing the x and y coordinates of the point.
    :return:
    """

    # Get the triangle element
    triangle_face = faces[face_index]
    triangle_element = points[triangle_face]

    # Compute the area of the triangle
    A_e = compute_triangle_element_area(points, triangle_face)

    # Get element points
    p_jx = triangle_element[1][0]
    p_jy = triangle_element[1][1]
    p_kx = triangle_element[2][0]
    p_ky = triangle_element[2][1]

    # Compute double integral over triangle
    t1 = h ** 2
    t2 = p_jx**2
    t3 = (-p_jy-2*p_kx+p_ky)*p_jx
    t4 = p_jy**2
    t5 = (p_kx -2*p_ky)*p_jy
    t6 = p_kx**2
    t7 = -p_kx*p_ky
    t8 = p_ky**2
    t9 = p_jx - p_jy - p_kx + p_ky
    result = t1 * ((t2 + t3 + t4 + t5 + t6 + t7 + t8) * t1 - 4*())

    return val