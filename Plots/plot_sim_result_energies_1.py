import matplotlib.pyplot as plt
import numpy as np

from EnergyComputations.compute_kinetic_energy import compute_kinetic_energy
from EnergyComputations.compute_potential_energy import compute_potential_energy
from EnergyComputations.compute_strain_energy import compute_strain_energy
from Simulator.result import Result


def plot_sim_result_energies_1(vertices, faces, density, result: Result, gravity, areas, lambda_, mu):
    kinetic_energies = np.array([compute_kinetic_energy(density, velocities) for velocities in result.nodal_velocities])
    potential_energies = np.array([compute_potential_energy(density, vertices.reshape([len(displacements)]) + displacements, gravity) for displacements in result.nodal_displacements])
    strain_energies = np.array([compute_strain_energy(i, faces, result, areas, lambda_, mu) for i in range(len(result.Es))])

    kinetic_energy_plot = plt.plot(result.time_steps, kinetic_energies, label='Kinetic energy')
    potential_energy_plot = plt.plot(result.time_steps, potential_energies, label='Potential energy')
    strain_energy_plot = plt.plot(result.time_steps, strain_energies, label='Strain energy')
    total_energy = kinetic_energies + potential_energies + strain_energies
    total_energy_plot = plt.plot(result.time_steps, total_energy, label='Total energy')
    plt.legend()
    plt.xlabel(r'$t$ (s)')
    plt.ylabel(r'$Energy$ (J)')
    plt.title(f'Showing kinetic and potential energy of the deformed cantilever mesh.')
    plt.show()