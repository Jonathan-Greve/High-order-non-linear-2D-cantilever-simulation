import numpy as np

# def compute_element_potential_energy(density, vertices_y_e, gravity, A_e):
#     """
#     Computes the potential energy of the element.
#     """
#
#     mass = density * A_e
#
#     center_of_mass = np.sum(vertices_y_e) / len(vertices_y_e)
#
#     return -gravity[1] * mass * center_of_mass

global_damping_energy_lost = 0

def compute_lost_damping_energy(displacements, damping_forces):
    """
    Computes the total lost damping energy of the mesh over time.
    """

    global global_damping_energy_lost
    global_damping_energy_lost += np.dot(np.transpose(displacements), damping_forces)

    return global_damping_energy_lost
