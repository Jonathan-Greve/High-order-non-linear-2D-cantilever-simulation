# Simulator class
# Containts the main loop of the simulator called simulate
import numpy as np
import quadpy
from tqdm import tqdm

from Mesh.Cantilever.area_computations import compute_triangle_element_area, \
    compute_all_element_areas
from Mesh.Cantilever.generate_2d_cantilever_delaunay import generate_2d_cantilever_delaunay
from Mesh.Cantilever.generate_2d_cantilever_kennys import generate_2d_cantilever_kennys
from Simulator.integral_computations import compute_shape_function_volume
from Simulator.result import Result
from Simulator.triangle_shape_functions import triangle_shape_function_i_helper, \
    triangle_shape_function_j_helper, triangle_shape_function_k_helper


class Simulator:
    def __init__(self, number_of_time_steps, time_step, material_properties,
                 length, height, number_of_nodes_x, number_of_nodes_y, traction_force, gravity):
        # Simulation settings
        self.number_of_time_steps = number_of_time_steps
        self.time_step = time_step
        self.gravity = gravity

        # Material settings
        self.material_properties = material_properties
        # self.lambda_ = (self.material_properties.youngs_modulus * self.material_properties.poisson_ratio
        #         / ((1 + self.material_properties.poisson_ratio) *
        #            (1 - 2 * self.material_properties.poisson_ratio)))
        # self.mu = (self.material_properties.youngs_modulus /
        #       (2 * (1 + self.material_properties.poisson_ratio)))

        self.lambda_ = (
                (self.material_properties.youngs_modulus * self.material_properties.poisson_ratio) /
                ((1+self.material_properties.poisson_ratio)*(1-2*self.material_properties.poisson_ratio))
        )
        self.mu = (
                self.material_properties.youngs_modulus /
                (2 * (1+self.material_properties.poisson_ratio))
        )

        # Cantilever settings
        self.length = length
        self.height = height
        self.number_of_nodes_x = number_of_nodes_x
        self.number_of_nodes_y = number_of_nodes_y
        self.traction_force = traction_force
        self.total_number_of_nodes = self.number_of_nodes_x * self.number_of_nodes_y

        # Initialize the cantilever mesh
        points, faces = generate_2d_cantilever_delaunay(self.length, self.height,
                                                     self.number_of_nodes_x, self.number_of_nodes_y)
        # points, faces = generate_2d_cantilever_kennys(self.length, self.height,
        #                                                 self.number_of_nodes_x, self.number_of_nodes_y)
        self.mesh_points = points
        self.mesh_faces = faces
        self.all_A_e = compute_all_element_areas(self.mesh_points, self.mesh_faces)

        # All volume under shape functions
        self.all_V_e = np.array([compute_shape_function_volume(self.mesh_points, face) for face in self.mesh_faces])

        # Boundary node indices
        self.dirichlet_boundary_indices_x = \
            np.arange(0, 2 * self.total_number_of_nodes - 1, self.number_of_nodes_x * 2)
        self.dirichlet_boundary_indices_y = self.dirichlet_boundary_indices_x + 1



    def simulate(self):
        # Initialize variables
        time = 0.0

        print("Simulation started...")
        print("----------------------------------------------------")
        print("Simulation settings:")
        print("  Time to simulate: {}".format(self.time_step * self.number_of_time_steps))
        print("  Time step: {}".format(self.time_step))
        print("  Number of time steps: {}".format(self.number_of_time_steps))
        print("----------------------------------------------------")

        # Precompute some variables
        M = self.compute_mass_matrix()
        C = self.compute_damping_matrix()
        f_t = self.compute_traction_forces()
        f_g = self.compute_body_forces(include_gravity=True)
        f = - f_t - f_g

        # Add boundary conditions
        # f[self.dirichlet_boundary_indices_x] = 0
        # f[self.dirichlet_boundary_indices_y] = 0

        u_n = np.zeros(self.total_number_of_nodes * 2)
        v_n = np.zeros(self.total_number_of_nodes * 2)
        a_n = np.zeros(self.total_number_of_nodes * 2)
        a_n[np.arange(1, self.total_number_of_nodes * 2, 2)] = self.gravity[1]
        x_n = self.mesh_points.reshape([self.total_number_of_nodes * 2])

        times = [0]
        displacements = [u_n]
        velocities = [v_n]
        accelerations = [a_n]
        Es = [np.zeros([len(self.mesh_faces), 2,2])]
        Ms = [M]

        # Main loop
        for i in tqdm(range(self.number_of_time_steps), desc="Running simulation"):
            # Compute stiffness matrix
            k, E = self.compute_stiffness_matrix(u_n)

            # Do simulation step
            Minv = np.linalg.inv(M)
            damping_term = np.dot(C, v_n)

            # # Remove all forces after 1 sec.
            # if (i * self.time_step > 1):
            #     f = f * 0

            forces = f - damping_term - k
            a_n_1 = np.dot(Minv,  forces)

            # M_condition_num = np.linalg.cond(M)
            # Minv_condition_num = np.linalg.cond(Minv)
            # C_condition_num = np.linalg.cond(C)

            # Dirichlet boundary conditions (set acceleration to 0 on the fixed boundary nodes)
            # a_n_1[self.dirichlet_boundary_indices_x] = 0
            # a_n_1[self.dirichlet_boundary_indices_y] = 0

            v_n_1 = v_n + self.time_step * a_n_1
            v_n_1[self.dirichlet_boundary_indices_x] = 0
            v_n_1[self.dirichlet_boundary_indices_y] = 0

            x_n_1 = x_n + self.time_step * v_n_1

            # New displacements
            u_n = x_n_1 - self.mesh_points.reshape([self.total_number_of_nodes * 2])

            # New velocities
            v_n = v_n_1

            # New positions
            x_n = x_n_1


            # Update time
            time += self.time_step
            times.append(time)
            displacements.append(u_n)
            velocities.append(v_n)
            accelerations.append(a_n_1)
            Es.append(E)
            Ms.append(M)

            # Print time
            # print(f"i: {i}. Time: {time}")

        return Result(times, displacements, velocities, accelerations, Es, Ms)

    def compute_integral_N_squared(self, triangle_face):
        # Compute matrix using quadpy (quadpy is a quadrature package)
        triangle = self.mesh_points[triangle_face]

        # get a "good" scheme of degree 10. (Even 2 should be enough since be use linear elements
        # and we multiply them to get a second order polynomial)
        scheme = quadpy.t2.get_good_scheme(10)

        def N_i(x):
            return triangle_shape_function_i_helper(self.mesh_points, triangle_face, x)

        def N_j(x):
            return triangle_shape_function_j_helper(self.mesh_points, triangle_face, x)

        def N_k(x):
            return triangle_shape_function_k_helper(self.mesh_points, triangle_face, x)

        test = scheme.integrate(lambda x: N_i(x), triangle)

        n_i_squared = scheme.integrate(lambda x: N_i(x) ** 2, triangle)
        n_j_squared = scheme.integrate(lambda x: N_j(x) ** 2, triangle)
        n_k_squared = scheme.integrate(lambda x: N_k(x) ** 2, triangle)

        n_ij = scheme.integrate(lambda x: N_i(x) * N_j(x), triangle)
        n_ik = scheme.integrate(lambda x: N_i(x) * N_k(x), triangle)
        n_jk = scheme.integrate(lambda x: N_j(x) * N_k(x), triangle)

        integral_N_square = np.array([
            [n_i_squared, 0, n_ij, 0, n_ik, 0],
            [0, n_i_squared, 0, n_ij, 0, n_ik],
            [n_ij, 0, n_j_squared, 0, n_jk, 0],
            [0, n_ij, 0, n_j_squared, 0, n_jk],
            [n_ik, 0, n_jk, 0, n_k_squared, 0],
            [0, n_ik, 0, n_jk, 0, n_k_squared]
        ]
        )

        return integral_N_square

    def compute_mass_matrix(self):
        def compute_element_mass_matrix(face_index):
            triangle_face = self.mesh_faces[face_index]
            integral_N_square = self.compute_integral_N_squared(triangle_face)
            return integral_N_square * self.material_properties.density

        # Compute all element mass matrices
        all_M_e = np.array([compute_element_mass_matrix(i) for i in range(len(self.mesh_faces))])

        # Assemble the mass matrix
        M = self.assemble_square_matrix(all_M_e)

        return M


    def compute_damping_matrix(self):
        def compute_element_damping_matrix(face_index):
            triangle_face = self.mesh_faces[face_index]
            integral_N_square = self.compute_integral_N_squared(triangle_face)
            return integral_N_square * self.material_properties.density

        # Compute all element mass matrices
        all_C_e = np.array([compute_element_damping_matrix(i) for i in range(len(self.mesh_faces))])

        # Assemble the mass matrix
        C = self.assemble_square_matrix(all_C_e)


        return C * self.material_properties.damping_coefficient

    def compute_stiffness_matrix(self, u_n):
        def compute_element_stiffness_matrix(triangle_face, A_e):
            node_i_idx = triangle_face[0]
            node_j_idx = triangle_face[1]
            node_k_idx = triangle_face[2]

            # Material coordinates
            X_i, Y_i = self.mesh_points[node_i_idx]
            X_j, Y_j = self.mesh_points[node_j_idx]
            X_k, Y_k = self.mesh_points[node_k_idx]

            # Spatial coordinates
            x_i, y_i = (X_i + u_n[node_i_idx*2]), (Y_i + u_n[node_i_idx*2 + 1])
            x_j, y_j = (X_j + u_n[node_j_idx*2]), (Y_j + u_n[node_j_idx*2 + 1])
            x_k, y_k = (X_k + u_n[node_k_idx*2]), (Y_k + u_n[node_k_idx*2 + 1])

            # Triangle material vectors
            dY_kj = Y_k - Y_j
            dY_ik = Y_i - Y_k
            dY_ji = Y_j - Y_i
            dX_kj = X_k - X_j
            dX_ik = X_i - X_k
            dX_ji = X_j - X_i

            # Triangle shape function derivatives
            dN_iX = dY_kj / (2 * A_e)
            dN_jX = dY_ik / (2 * A_e)
            dN_kX = dY_ji / (2 * A_e)
            dN_iY = -dX_kj / (2 * A_e)
            dN_jY = -dX_ik / (2 * A_e)
            dN_kY = -dX_ji / (2 * A_e)

            # Compute F
            F_i = np.array([
                [dN_iX * x_i, dN_iY * x_i],
                [dN_iX * y_i, dN_iY * y_i],
            ])
            F_j = np.array([
                [dN_jX * x_j, dN_jY * x_j],
                [dN_jX * y_j, dN_jY * y_j],
            ])
            F_k = np.array([
                [dN_kX * x_k, dN_kY * x_k],
                [dN_kX * y_k, dN_kY * y_k],
            ])
            F = F_i + F_j + F_k
            # print(f'Triangle {triangle_face}')
            # F = self.compute_F(
            #     np.array([x_i, y_i]),
            #     np.array([x_j, y_j]),
            #     np.array([x_k, y_k]),
            #     np.array([X_i, Y_i]),
            #     np.array([X_j, Y_j]),
            #     np.array([X_k, Y_k]),
            # )

            # Compute E
            I = np.eye(2)
            C = np.dot(np.transpose(F), F)
            E = (C - I) / 2.0

            # Compute S
            S = self.lambda_ * np.trace(E) * I + 2 * self.mu * E

            # Compute vector field gradients
            grad_i = np.array([dN_iX, dN_iY])
            grad_j = np.array([dN_jX, dN_jY])
            grad_k = np.array([dN_kX, dN_kY])

            k_e = np.array([
                (A_e * np.dot(np.dot(F, S), grad_i))[0],
                (A_e * np.dot(np.dot(F, S), grad_i))[1],
                (A_e * np.dot(np.dot(F, S), grad_j))[0],
                (A_e * np.dot(np.dot(F, S), grad_j))[1],
                (A_e * np.dot(np.dot(F, S), grad_k))[0],
                (A_e * np.dot(np.dot(F, S), grad_k))[1]
            ])

            return k_e, E, F

        # Computes all element stiffness matrices
        all_k_e_matrices = np.zeros([len(self.mesh_faces), 6])
        all_Es = np.zeros([len(self.mesh_faces), 2, 2])
        all_Fs = np.zeros([len(self.mesh_faces), 2, 2])
        for i in range(len(self.mesh_faces)):
            triangle_face = self.mesh_faces[i]
            A_e = self.all_A_e[i]
            k_e, E, F = compute_element_stiffness_matrix(triangle_face, A_e)
            all_k_e_matrices[i] = k_e
            all_Es[i] = E
            all_Fs[i] = F

        # Assemble the stiffness matrix
        k = np.zeros([2 * self.total_number_of_nodes])
        for i in range(len(self.mesh_faces)):
            triangle_face = self.mesh_faces[i]

            node_i_idx = triangle_face[0]
            node_j_idx = triangle_face[1]
            node_k_idx = triangle_face[2]

            # The indices of i_x, i_y, j_x, j_y, k_x, k_y
            global_indices_list = np.array([
                node_i_idx * 2,
                node_i_idx * 2 + 1,
                node_j_idx * 2,
                node_j_idx * 2 + 1,
                node_k_idx * 2,
                node_k_idx * 2 + 1
            ])
            #         print(global_indices_list)

            k[global_indices_list] += all_k_e_matrices[i]


        # print("F: {}".format(all_Fs[0]))
        # print("CE: {}".format(all_Es[0]))
        # print("F: {}".format(all_Fs[1]))
        # print("CE: {}".format(all_Es[1]))
        # print('-------------------------------------')
        return k, all_Es

    def assemble_square_matrix(self, all_M_e):
        matrix = np.zeros([2 * self.total_number_of_nodes, 2 * self.total_number_of_nodes])
        for i in range(len(self.mesh_faces)):
            triangle_face = self.mesh_faces[i]

            node_i_idx = triangle_face[0]
            node_j_idx = triangle_face[1]
            node_k_idx = triangle_face[2]

            # The indices of i_x, i_y, j_x, j_y, k_x, k_y
            global_indices_list = np.array([
                node_i_idx * 2,
                node_i_idx * 2 + 1,
                node_j_idx * 2,
                node_j_idx * 2 + 1,
                node_k_idx * 2,
                node_k_idx * 2 + 1
            ])

            matrix[global_indices_list.reshape([6, 1]), global_indices_list] += all_M_e[i]

        return matrix

    def compute_body_forces(self, include_gravity=True):
        f_b = np.zeros([2 * self.total_number_of_nodes])

        # Add gravity force to all nodes
        if include_gravity:
            def compute_element_gravity_term(face_index):
                V_e = self.all_V_e[face_index]

                value = V_e * self.material_properties.density * self.gravity[1]
                f_g_e = np.array([
                    0,
                    value,
                    0,
                    value,
                    0,
                    value
                ])

                return f_g_e

            all_gravity_terms = np.zeros([len(self.mesh_faces), 6])
            for i in range(len(self.mesh_faces)):
                f_g_e = compute_element_gravity_term(i)
                all_gravity_terms[i] = f_g_e

            # Assemble the stiffness matrix
            f_g = np.zeros([2 * self.total_number_of_nodes])
            for i in range(len(self.mesh_faces)):
                triangle_face = self.mesh_faces[i]

                node_i_idx = triangle_face[0]
                node_j_idx = triangle_face[1]
                node_k_idx = triangle_face[2]

                # The indices of i_x, i_y, j_x, j_y, k_x, k_y
                global_indices_list = np.array([
                    node_i_idx * 2,
                    node_i_idx * 2 + 1,
                    node_j_idx * 2,
                    node_j_idx * 2 + 1,
                    node_k_idx * 2,
                    node_k_idx * 2 + 1
                ])
                #         print(global_indices_list)

                f_g[global_indices_list] += all_gravity_terms[i]

            P_0g = - f_g

            f_b += P_0g

        return f_b

    def compute_traction_forces(self):
        """
            Compute the traction vector acting on the object. It takes as input a
            traction vector with dimension: 2x1. Then returns the (2n)x1 vector
            with all the nodes on the traction edge set to the traction of this parameter.

            :param traction: The traction vector applied to all the nodes on the traction edge.

            :return: A (2n)x1 vector.
            """

        # x-indices
        traction_edge_indices_x = np.arange((self.number_of_nodes_x - 1) * 2,
                                            2 * self.total_number_of_nodes - 1,
                                            self.number_of_nodes_x * 2)
        # y-indices
        traction_edge_indices_y = traction_edge_indices_x + 1

        line_element_length = self.height / (len(traction_edge_indices_y) - 1)

        f_t = np.zeros([2 * self.total_number_of_nodes])

        scaled_traction = line_element_length * np.array(self.traction_force)

        f_t[traction_edge_indices_x] = scaled_traction[0]
        f_t[traction_edge_indices_y] = scaled_traction[1]

        # Remember end nodes only have half traction values.
        f_t[traction_edge_indices_x[0]] = scaled_traction[0] / 2
        f_t[traction_edge_indices_x[-1]] = scaled_traction[0] / 2
        f_t[traction_edge_indices_y[0]] = scaled_traction[1] / 2
        f_t[traction_edge_indices_y[-1]] = scaled_traction[1] / 2

        P_t = -f_t

        return P_t

    def compute_F(self, x_i, x_j, x_k, X_i, X_j, X_k):
        x_ij = (x_j - x_i)
        x_ik = (x_k - x_i)

        X_ij = (X_j - X_i)
        X_ik = (X_k - X_i)

        D = np.array([
            [x_ij[0], x_ik[0]],
            [x_ij[1], x_ik[1]],
        ])

        D_0 = np.array([
            [X_ij[0], X_ik[0]],
            [X_ij[1], X_ik[1]],
        ])

        F = D @ np.linalg.inv(D_0)
        return F

