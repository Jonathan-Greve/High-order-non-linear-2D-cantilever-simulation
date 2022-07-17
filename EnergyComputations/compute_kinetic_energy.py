import numpy as np


# def compute_kinetic_energy(density, v, A):
#     """
#     Computes the kinetic energy of the mesh.
#     """
#     mass = density * np.sum(A)
#
#     return 0.5 * mass * np.dot(v.T, v)

def compute_element_kinetic_energy(density, velocities_x_e, velocities_y_e, A_e):
    """
    Computes the potential energy of the element.
    """

    mass = density * A_e

    average_element_velocity_x = np.sum(velocities_x_e) / len(velocities_x_e)
    average_element_velocity_y = np.sum(velocities_y_e) / len(velocities_y_e)

    vx_squared = average_element_velocity_x ** 2
    vy_squared = average_element_velocity_y ** 2

    v_squared = vx_squared + vy_squared

    return 0.5 * mass * v_squared

def compute_kinetic_energy(density, velocities, faces, A):
    """
    Computes the potential energy of the mesh.
    """

    vertices_x = velocities[np.arange(0, len(velocities) - 1, 2)]
    vertices_y = velocities[np.arange(1, len(velocities), 2)]

    E_kinetic = 0
    for i in range(len(faces)):
        face = faces[i]
        A_e = A[i]
        E_kinetic_element = compute_element_kinetic_energy(density,
                                                           vertices_x[face],
                                                           vertices_y[face],
                                                           A_e)
        E_kinetic += E_kinetic_element


    return E_kinetic

def compute_kinetic_energy_from_M_and_v(M, v):
    """
    Computes the kinetic energy of the mesh.
    """

    return 0.5 * np.dot(v.T, np.dot(M, v))
