import numpy as np

def compute_element_potential_energy(density, vertices_y_e, gravity, A_e):
    """
    Computes the potential energy of the element.
    """

    mass = density * A_e

    center_of_mass = np.sum(vertices_y_e) / len(vertices_y_e)

    return -gravity[1] * mass * center_of_mass

def compute_potential_energy(density, vertices, faces, gravity, A):
    """
    Computes the potential energy of the mesh.
    """

    ys = vertices[np.arange(1, len(vertices), 2)]
    return -gravity[1] * (density*np.sum(A)) * np.sum(ys) / len(ys)

    # vertices_y = vertices[np.arange(1, len(vertices), 2)]
    #
    # E_potential = 0
    # for i in range(len(faces)):
    #     face = faces[i]
    #     A_e = A[i]
    #     E_potential_element = compute_element_potential_energy(density,
    #                                                            vertices_y[face],
    #                                                            gravity,
    #                                                            A_e)
    #     E_potential += E_potential_element
    #
    #
    # return E_potential