import matplotlib.pyplot as plt
import numpy as np

from EnergyComputations.compute_kinetic_energy import compute_kinetic_energy, \
    compute_kinetic_energy_from_M_and_v
from EnergyComputations.compute_potential_energy import compute_potential_energy
from EnergyComputations.compute_strain_energy import compute_strain_energy
from Simulator.result import Result


def plot_sim_result_energies_1(vertices, faces, density, result: Result, gravity, areas, lambda_, mu):
    velocities = result.nodal_velocities
    displacements = result.nodal_displacements

    kinetic_energies = np.array([compute_kinetic_energy(density, velocities[i], faces, areas) for i in range(len(velocities))])
    kinetic_energies_Mv = np.array([compute_kinetic_energy_from_M_and_v(result.Ms[i], velocities[i]) for i in range(len(velocities))])
    potential_energies = np.array([compute_potential_energy(density, vertices.reshape([len(displacements[i])]) + displacements[i], faces, gravity, areas) for i in range(len(displacements))])
    strain_energies = np.array([compute_strain_energy(i, faces, result, areas, lambda_, mu) for i in range(len(result.Es))])

    kinetic_energy_plot = plt.plot(result.time_steps, kinetic_energies, label='Kinetic energy', color='cyan', linestyle='-.')
    kinetic_energy_plot_Mv = plt.plot(result.time_steps, kinetic_energies_Mv, label='Kinetic energy (Mv)', color='red', linestyle='solid', alpha=0.5)
    potential_energy_plot = plt.plot(result.time_steps, potential_energies, label='Potential energy', color='green', linestyle='solid')
    strain_energy_plot = plt.plot(result.time_steps, strain_energies, label='Strain energy', color='blue', linestyle='solid')
    total_energy = kinetic_energies_Mv + potential_energies + strain_energies
    total_energy_plot = plt.plot(result.time_steps, total_energy, label='Total energy', color='black', linestyle='solid')
    plt.legend()
    plt.xlabel(r'$t$ (s)')
    plt.ylabel(r'$Energy$ (J)')
    plt.title(f'Showing kinetic and potential energy of the deformed cantilever mesh.')
    plt.show()