import matplotlib.pyplot as plt
import numpy as np

from Mesh.HigherOrderMesh.decode_triangle_indices import decode_triangle_indices
from Mesh.HigherOrderMesh.generate_ijk_indices import generate_ijk_indices
from Simulator.HigherOrderElements.shape_functions import silvester_shape_function


def plot_sim_result_1(FEM_V, FEM_encodings, u, num_nodes_x, num_nodes_y, traction, time, element_order):
    n = num_nodes_x * num_nodes_y
    print(f'nodes x: {num_nodes_x}')
    print(f'nodes y: {num_nodes_y}')
    print(f'Vertices: {FEM_V.shape}')
    print(f'Faces: {FEM_encodings.shape}')
    print(f'uuu: {u.shape}')
    print(f'n: {n}')

    sample_points = generate_ijk_indices(20) / 20

    deformed_V = FEM_V + u.reshape((len(FEM_V), 2))

    for i, encoding in enumerate(FEM_encodings):
        reference_points = []
        interpolated_points = []
        global_indices, ijk_indices = decode_triangle_indices(encoding, element_order)
        for sample_point in sample_points:
            N_vals = []
            for i, ijk_index in enumerate(ijk_indices):
                N_val = silvester_shape_function(ijk_index, sample_point, element_order)
                N_vals.append(N_val)
            N_vals = np.array(N_vals)

            interpolated_point = N_vals @ deformed_V[global_indices]
            interpolated_points.append(interpolated_point)
            interpolated_reference_point = N_vals @ FEM_V[global_indices]
            reference_points.append(interpolated_reference_point)
        interpolated_points = np.array(interpolated_points)
        reference_points = np.array(reference_points)
        plt.scatter(reference_points[:, 0], reference_points[:, 1], zorder=0)
        plt.scatter(interpolated_points[:, 0], interpolated_points[:, 1], zorder=10)

    plt.show()