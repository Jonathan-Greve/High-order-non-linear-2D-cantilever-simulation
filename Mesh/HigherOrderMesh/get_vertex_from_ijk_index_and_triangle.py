def get_vertex_from_ijk_index_and_triangle(V_triangle, ijk_index, n):
    """
    Get the vertex from the ijk index and the triangle.
    The ijk_index encodes the barycentric coordinates of the vertex.
    More specifically, the ijk_index is a tuple of three integers (i,j,k)
    and when we divide by n we get the barycentric coordinates.

    Then we just use:
        (x,y) = V_i * (i/n) + V_j * (j/n) + V_k * (k/n)
    where (i/n), (j/n) and (k/n) are the barycentric coordinates.

    :param V_triangle:
    :param ijk_index:
    :param n:
    :return vertex:
    """

    # Get the barycentric coordinates
    bary_i, bary_j, bary_k = ijk_index / n

    vertex = V_triangle[0] * bary_i + V_triangle[1] * bary_j + V_triangle[2] * bary_k

    return vertex