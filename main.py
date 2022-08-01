import math
import os
import pickle
import sys

import Materials.MaterialProperties as mat_prop
from Plots.plot_sim_result_1 import plot_sim_result_1
from Plots.plot_sim_result_energies_1 import plot_sim_result_energies_1
from Plots.plot_sim_result_gif_1 import make_sim_result_gif_1
from Simulator.simulator import Simulator

import time


def main():
    # Setup material properties query
    material_properties_query = mat_prop.MaterialPropertiesQuery()

    # Select material
    material_name = "Test 1"
    material_properties = material_properties_query.get_material_properties(material_name)

    # Print material properties
    print("----------------------------------------------------")
    print("Material properties:")
    print("  Material name: {}".format(material_name))
    print("  Young's modulus: {}".format(material_properties.youngs_modulus))
    print("  Poisson ratio: {}".format(material_properties.poisson_ratio))
    print("  Density: {}".format(material_properties.density))
    print("  Damping coefficient: {}".format(material_properties.damping_coefficient))
    print("----------------------------------------------------")

    # Setup simulation settings
    time_to_simulate = 1.8 * 5  # Seconds
    time_step = 0.0001  # Seconds
    # time_step = 1 / 30
    number_of_time_steps = math.ceil(time_to_simulate / time_step)

    # Cantilever settings
    length = 6.0  # Meters
    height = 2.0  # Meters
    number_of_nodes_x = 60 # Number of nodes in x direction
    number_of_nodes_y = 20 # Number of nodes in y direction
    traction_force = [0, -10000.0]  # Newtons
    gravity = [0, -0]  # m/s^2

    # Print simulation settings
    print("----------------------------------------------------")
    print("Simulation settings:")
    print("  Time to simulate: {}".format(time_to_simulate))
    print("  Time step: {}".format(time_step))
    print("  Number of time steps: {}".format(number_of_time_steps))
    print("----------------------------------------------------")

    simulator = Simulator(number_of_time_steps, time_step, material_properties,
                          length, height, number_of_nodes_x, number_of_nodes_y, traction_force,
                          gravity)
    sim_file_name = f'result_{length}l_{height}h_{number_of_nodes_x}xn_{number_of_nodes_y}yn_{traction_force}tf_{time_to_simulate}t_{time_step}ts_{material_name}mn_{gravity}g_{simulator.material_properties.damping_coefficient}dc'
    try:
        f = open(sim_file_name, 'rb')
        result = pickle.load(f)
        f.close()
    except:
        # Start simulation
        result = simulator.simulate()
        f = open(sim_file_name, 'wb')
        pickle.dump(result, f)
        f.close()

    # Plot the final simulation result
    plot_sim_result_1(simulator.mesh_points, simulator.mesh_faces, result.nodal_displacements[-1],
                      simulator.number_of_nodes_x, simulator.number_of_nodes_y, simulator.traction_force,
                      result.time_steps[-1])

    # Plot the various energies as a function of time
    plot_sim_result_energies_1(simulator.mesh_points, simulator.mesh_faces,
                               simulator.material_properties.density, result,
                               simulator.gravity, simulator.all_A_e, simulator.lambda_, simulator.mu)

    # make a gif of the simulation
    make_sim_result_gif_1(simulator.mesh_points, simulator.mesh_faces,
                          result, simulator.number_of_nodes_x, simulator.number_of_nodes_y,
                          simulator.traction_force, result.time_steps[-1], simulator.time_step,
                          sim_file_name)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
