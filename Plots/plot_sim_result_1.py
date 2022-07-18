import matplotlib.pyplot as plt
import numpy as np

def plot_sim_result_1(vertices, faces, u, num_nodes_x, num_nodes_y, traction, time):
    n = num_nodes_x * num_nodes_y
    print(f'nodes x: {num_nodes_x}')
    print(f'nodes y: {num_nodes_y}')
    print(f'Vertices: {vertices.shape}')
    print(f'Faces: {faces.shape}')
    print(f'uuu: {u.shape}')
    print(f'n: {n}')
    reference, _ = plt.triplot(vertices[:,0], vertices[:,1], faces, label='Reference', alpha=0.3)
    deformed, _ = plt.triplot(vertices[:,0] + u[np.arange(0, 2*n-1, 2)], vertices[:,1] + u[np.arange(0, 2*n-1, 2)+1], faces, label='Deformed')
    plt.legend(handles=[reference, deformed], loc='upper left')
    plt.xlabel(r'$X_1$')
    plt.ylabel(r'$X_2$')
    plt.title(f'Showing the deformed cantilever mesh compared with the reference mesh.' +
              '\n' + f'Horizontal nodes: {num_nodes_x}. Vertical nodes: {num_nodes_y}. Total nodes: {num_nodes_x * num_nodes_y}. Elements: {faces.shape[0]}' +
              '\n' + f'Traction force: {traction}'
              '\n' + f'Time elapsed: {time}'
              )
    plt.show()