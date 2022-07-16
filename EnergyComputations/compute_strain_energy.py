import numpy as np

def compute_element_strain_energy(A_e, E_e, lambda_, mu):
    """
    Computes the strain energy of the element.
    """
    E_e_squared = np.dot(np.transpose(E_e), E_e)
    element_strain_energy = (lambda_ / 2) * np.trace(E_e)**2 + mu * np.trace(E_e_squared)
    # element_strain_energy = (lambda_ / 2) * (np.trace(E_e)**2) + mu * (np.sum(np.multiply(E_e,E_e)))

    return element_strain_energy * A_e

def compute_strain_energy(index, faces, result, areas, lambda_, mu):
    """
    Computes the strain energy of the mesh.
    """


    strain_energy = 0
    for i in range(len(faces)):
        A_e = areas[i]
        E_e = result.Es[index][i]
        strain_energy += compute_element_strain_energy(A_e, E_e, lambda_, mu)

    return strain_energy