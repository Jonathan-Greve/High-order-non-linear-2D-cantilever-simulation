import numpy as np

from Mesh.HigherOrderMesh.decode_triangle_indices import decode_triangle_indices
from Simulator.cartesian_to_barycentric import cartesian_to_barycentric


def f_i(i, xi, n):
    return ((n*xi - i + 1) / i)

def P(z, xi, n):
    if z == 0:
        return 1
    return f_i(z, xi, n) * P(z - 1, xi, n)

def silvester_shape_function(ijk_index, xi, n):
    """
    Returns a the value of a shape function corresponding to the ijk_index.
    :param ijk_index:
    :param n:
    :return shape_function:
    """

    P_i = P(ijk_index[0], xi[0], n)
    P_j = P(ijk_index[1], xi[1], n)
    P_k = P(ijk_index[2], xi[2], n)

    shape_function = P_i * P_j * P_k

    return shape_function


def dP_dxi(z, xi, n):
    result = 0
    for i in range(2, z + 1):
        result += (n / i) / f_i(i, xi, n)

    f_1 = f_i(1, xi, n)
    if f_1 != 0:
        result += (n / 1) / f_1

    derivative = result * P(z, xi, n)

    return derivative

def shape_function_barycentric_derivative(ijk_index, xi, xi_index, n):
    """
    Computes the barycentric derivative of the shape function at the ijk_index with
    order n.

    The parameter {xi_index} is either 0, 1 or 2 representing the barycentric coordinates
    we want to take the derivative with respect to: 0 for xi_1, 1 for xi_2, 2  for xi_3.

    :param ijk_index:
    :param xi:
    :param xi_index:
    :param n:
    :return result:
    """

    i = ijk_index[0]
    j = ijk_index[1]
    k = ijk_index[2]
    zs = [i, j, k]

    P_i = P(i, xi[0], n)
    P_j = P(j, xi[1], n)
    P_k = P(k, xi[2], n)
    Ps = [P_i, P_j, P_k]

    z = zs[xi_index]

    slow_result = 0
    fast_result = 0
    for l in range(1, z + 1):
        f_l = f_i(l, xi[xi_index], n)
        if f_l != 0:
            df_l = n / l
            fast_result += (df_l / f_l)
        else:
            inner_result = n / l
            for h in range(l + 1, z + 1):
                inner_result *= f_i(h, xi[xi_index], n)
            slow_result += inner_result

    result = (slow_result + fast_result * Ps[xi_index]) * Ps[xi_index - 1] * Ps[xi_index - 2]

    return result


def shape_function_spatial_derivative(V_e, triangle_encoding, x, n):
    """
    Computes the spatial derivatives of the all the shape function with order n at the position
    x within the triangle.
    The V_e parameter contains the (x,y) pairs of each vertex on the triangle.
    The triangle_encoding parameter contains the global_index encoding of the triangle.

    The triangle_vertices are sorted as: [corner vertices, internal nodes, ij edge, jk edge, ki edge]

    In the code an e subscript denotes that it is for the element only. E.g. F_e, V_e.

    :param V_e:
    :param triangle_encoding:
    :param x:
    :param n:
    :return:
    """

    # Number of vertices in the triangle
    m = int((n+1)*(n+2)/2)

    # Get the global indices and ijk_indices of the triangle
    F_e, ijk_indices_e = decode_triangle_indices(triangle_encoding, n)

    # Get the barycentric coordinates of the vertices for the triangle
    xi = cartesian_to_barycentric(x, V_e)

    xs = V_e[:,0]
    ys = V_e[:,1]

    # V_mat matrix
    V_mat = np.ones((3, m))
    V_mat[1,:] = xs
    V_mat[2,:] = ys

    # Barycentric derivatives matrix
    dN_dxi = np.zeros((m, 3))
    for i in range(len(ijk_indices_e)):
        for xi_index in range(3):
            ijk_index = ijk_indices_e[i] # The ijk index of the shape function
            dN_dxi[i, xi_index] = shape_function_barycentric_derivative(ijk_index, xi, xi_index, n)

    B = np.dot(V_mat, dN_dxi)
    BInv = np.linalg.inv(B)

    dxi_dx = np.dot(BInv, np.array([[0, 0], [1, 0], [0, 1]]))

    dN_dx = np.dot(dN_dxi, dxi_dx)

    return dN_dx

def binomial(x,y,i,k):
    return (y**k) * (x**(i-k))

def binomial_theorem(x,y,n):
    result = []
    for i in range (n+1):
        for k in range (i+1):
            result.append(binomial(x,y,i,k))

    return np.array(result)

def derivative_binomial_x(x,y,i,k):
    if i-k-1 < 0 and x == 0:
        return 0
    return (i-k) * x**(i-k-1) * y**k

def derivative_binomial_y(x,y,i,k):
    if k-1 < 0 and y == 0:
        return 0
    return k*x**(i-k) * y**(k-1)

def derivative_binomial_theorem(x,y,n):
    result_x = []
    result_y = []
    for i in range (n+1):
        for k in range (i+1):
            result_x.append(derivative_binomial_x(x,y,i,k))
            result_y.append(derivative_binomial_y(x,y,i,k))

    return np.array([result_x, result_y])

def vandermonde_matrix(V_e, n):
    x = V_e[:,0]
    y = V_e[:,1]

    num_cols = int((n+1)*(n+2)/2)

    V = np.zeros((len(x), num_cols))

    for i in range(num_cols):
            V[i,:] = binomial_theorem(x[i], y[i], n)

    return V

def vandermonde_shape_function(V_e, x, n):
    """
    Returns the value of the vandermonde shape function at x.
    :param triangle:
    :param x:
    :param n:
    :return shape_function_val:
    """

    V = vandermonde_matrix(V_e, n)
    p = binomial_theorem(x[0], x[1], n)

    results = []
    for i in range(len(x[0])):
        p_i = p[:,i]
        results.append(p_i@np.linalg.inv(V))

    return np.array(results)

def vandermonde_spatial_derivative(V_e, x, n):
    """
    Returns the spatial derivatives of the vandermonde shape function at x.
    :param triangle:
    :param x:
    :param n:
    :return shape_function_val:
    """

    V = vandermonde_matrix(V_e, n)
    dp = derivative_binomial_theorem(x[0], x[1], n)

    results = []
    if not np.isscalar(x[0]):
        for i in range(len(x[0])):
            dp_i = dp[:,i]
            results.append(dp_i@np.linalg.inv(V))
    else:
        return dp@np.linalg.inv(V)

    return np.array(results)