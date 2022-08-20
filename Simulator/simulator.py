# Simulator class
# Containts the main loop of the simulator called simulate
import numpy as np
import quadpy
from tqdm import tqdm

from Mesh.Cantilever.area_computations import compute_triangle_element_area, \
    compute_all_element_areas
from Mesh.Cantilever.generate_2d_cantilever_delaunay import generate_2d_cantilever_delaunay
from Mesh.Cantilever.generate_2d_cantilever_kennys import generate_2d_cantilever_kennys
from Mesh.HigherOrderMesh.decode_triangle_indices import decode_triangle_indices
from Mesh.HigherOrderMesh.generate_FEM_mesh import generate_FEM_mesh
from Simulator.HigherOrderElements.shape_functions import silvester_shape_function
from Simulator.cartesian_to_barycentric import cartesian_to_barycentric
from Simulator.integral_computations import compute_shape_function_volume
from Simulator.result import Result
from Simulator.triangle_shape_functions import triangle_shape_function_i_helper, \
    triangle_shape_function_j_helper, triangle_shape_function_k_helper


class Simulator:
    def __init__(self, number_of_time_steps, time_step, material_properties,
                 length, height, number_of_nodes_x, number_of_nodes_y, traction_force, gravity,
                 element_order=1):
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

        # Element settings
        self.element_order = element_order

        # Initialize the cantilever mesh
        points, faces = generate_2d_cantilever_delaunay(self.length, self.height,
                                                     self.number_of_nodes_x, self.number_of_nodes_y)
        # points, faces = generate_2d_cantilever_kennys(self.length, self.height,
        #                                                 self.number_of_nodes_x, self.number_of_nodes_y)
        self.mesh_points = points.astype(np.float64)
        self.mesh_faces = faces
        self.all_A_e = compute_all_element_areas(self.mesh_points, self.mesh_faces)

        # All volume under shape functions
        self.all_V_e = np.array([compute_shape_function_volume(self.mesh_points, face) for face in self.mesh_faces], dtype=np.float64)


        # FEM mesh vertices, ijk_index for every V in FEM_V, global indice encoding for every V in FEM_V
        self.FEM_V, self.FEM_encoding = generate_FEM_mesh(self.mesh_points, self.mesh_faces, self.element_order)
        self.total_number_of_nodes = len(self.FEM_V)

        # Boundary node indices
        self.boundary_len = 0.0001
        self.dirichlet_boundary_indices_x = []
        for i, vertex in enumerate(self.FEM_V):
            if vertex[0] < 0 - (self.length / 2) + self.boundary_len:
                self.dirichlet_boundary_indices_x.append(2*i)

        self.dirichlet_boundary_indices_x = np.array(self.dirichlet_boundary_indices_x, dtype=np.int32)
        self.dirichlet_boundary_indices_y = self.dirichlet_boundary_indices_x + 1

        def check_if_traction_node(vertex):
            if vertex[0] > 0 + (self.length / 2) - self.boundary_len:
                return True
            else:
                return False

        # list of (encoding_index, edge_index). Edge index: 0 for ij, 1 for jk, 2 for ki
        self.traction_encodings = []
        for i, encoding in enumerate(self.FEM_encoding):
            global_indices, ijk_indices = decode_triangle_indices(encoding, self.element_order)

            is_i_traction_node = check_if_traction_node(self.FEM_V[global_indices[0]])
            is_j_traction_node = check_if_traction_node(self.FEM_V[global_indices[1]])
            is_k_traction_node = check_if_traction_node(self.FEM_V[global_indices[2]])

            # Check ij-edge
            if is_i_traction_node and is_j_traction_node:
                self.traction_encodings.append((i, 0))

            # Check jk-edge
            if is_j_traction_node and is_k_traction_node:
                self.traction_encodings.append((i, 1))
            # Check ki-edge
            if is_k_traction_node and is_i_traction_node:
                self.traction_encodings.append((i, 2))

        print("Simulator initialized")

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
        Minv = np.linalg.inv(M)
        C = self.compute_damping_matrix()
        f_t = self.compute_traction_forces()
        f_g = self.compute_body_forces(include_gravity=True)
        f = - f_t - f_g

        # Add boundary conditions
        # f[self.dirichlet_boundary_indices_x] = 0
        # f[self.dirichlet_boundary_indices_y] = 0

        u_n = np.zeros(self.total_number_of_nodes * 2, dtype=np.float64)
        v_n = np.zeros(self.total_number_of_nodes * 2, dtype=np.float64)
        a_n = np.zeros(self.total_number_of_nodes * 2, dtype=np.float64)
        a_n[np.arange(1, self.total_number_of_nodes * 2, 2)] = self.gravity[1]
        x_n = self.mesh_points.reshape([self.total_number_of_nodes * 2])

        times = [0]
        displacements = [u_n]
        velocities = [v_n]
        accelerations = [a_n]
        Es = [np.zeros([len(self.mesh_faces), 2,2], dtype=np.float64)]
        Ms = [M]
        damping_forces = [C@v_n]

        # Main loop
        for i in tqdm(range(self.number_of_time_steps), desc="Running simulation"):
            time_step_size = np.min([self.time_step, self.number_of_time_steps*self.time_step - time])
            # Compute stiffness matrix
            k, E = self.compute_stiffness_matrix(u_n)

            # Do simulation step
            damping_term = np.dot(C, v_n)

            # # Remove all forces after 1 sec.
            # if (i * self.time_step > 1):
            #     f = f * 0

            forces = f - damping_term - k
            # forces[np.abs(forces) < 1e-10] = 0

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
            v_n_1[np.abs(v_n_1) < 1e-10] = 0

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
            damping_forces.append(damping_term)


            # Print time
            # print(f"i: {i}. Time: {time}")

        return Result(times, displacements, velocities, accelerations, Es, Ms, damping_forces, f_g)

    def compute_integral_N_squared(self, triangle_encoding):
        # Compute matrix using quadpy (quadpy is a quadrature package)
        global_indices, ijk_indices = decode_triangle_indices(triangle_encoding, self.element_order)
        V_e = self.FEM_V[global_indices]

        # Number of nodes
        m = (self.element_order + 1) * (self.element_order + 2) // 2
        if len(global_indices) != m:
            raise Exception("Number of nodes in element is not correct")

        i,j,k = global_indices[0:3]

        # Corner vertices of triangle
        triangle = self.mesh_points[[i,j,k]]

        scheme = quadpy.t2.get_good_scheme(self.element_order+1)

        # N is 2xm so the "square" matrix given by the outer product with itself is 2m x 2m
        integral_N_square = np.zeros((m*2, m*2))
        for i in range(2 * m):
            for j in range(2 * m):
                if i % 2 == 0 and j % 2 != 0:
                    continue
                if i % 2 != 0 and j % 2 == 0:
                    continue
                if i == j:
                    def f(x):
                        xi = cartesian_to_barycentric(x, triangle)
                        shape_function = silvester_shape_function(ijk_indices[int(i / 2)], xi, self.element_order)
                        return shape_function ** 2
                else:
                    def f(x):
                        xi = cartesian_to_barycentric(x, triangle)
                        shape_function_1 = silvester_shape_function(ijk_indices[int(i / 2)], xi, self.element_order)
                        shape_function_2 = silvester_shape_function(ijk_indices[int(j / 2)], xi, self.element_order)

                        return shape_function_1 * shape_function_2

                integral_N_square[i,j] = scheme.integrate(f, triangle)

        return integral_N_square

    def compute_mass_matrix(self):
        def compute_element_mass_matrix(face_index):
            triangle_encoding = self.FEM_encoding[face_index]
            integral_N_square = self.compute_integral_N_squared(triangle_encoding)
            return integral_N_square * self.material_properties.density

        # Compute all element mass matrices
        all_M_e = np.array([compute_element_mass_matrix(i) for i in range(len(self.mesh_faces))], dtype=np.float64)

        # Assemble the mass matrix
        M = self.assemble_square_matrix(all_M_e)

        return M


    def compute_damping_matrix(self):
        def compute_element_damping_matrix(face_index):
            triangle_encoding = self.FEM_encoding[face_index]
            integral_N_square = self.compute_integral_N_squared(triangle_encoding)
            return integral_N_square * self.material_properties.density

        # Compute all element mass matrices
        all_C_e = np.array([compute_element_damping_matrix(i) for i in range(len(self.mesh_faces))], dtype=np.float64)

        # Assemble the mass matrix
        C = self.assemble_square_matrix(all_C_e)


        return C * self.material_properties.damping_coefficient

    def compute_stiffness_matrix(self, u_n):
        def compute_element_stiffness_matrix(triangle_face, A_e):
            node_i_idx = triangle_face[0]
            node_j_idx = triangle_face[1]
            node_k_idx = triangle_face[2]
            
            node_i_global_index_x = node_i_idx * 2
            node_i_global_index_y = node_i_global_index_x + 1
            node_j_global_index_x = node_j_idx * 2
            node_j_global_index_y = node_j_global_index_x + 1
            node_k_global_index_x = node_k_idx * 2
            node_k_global_index_y = node_k_global_index_x + 1

            # Material coordinates
            X_i, Y_i = self.mesh_points[node_i_idx]
            X_j, Y_j = self.mesh_points[node_j_idx]
            X_k, Y_k = self.mesh_points[node_k_idx]


            # Spatial coordinates
            x_i, y_i = (X_i + u_n[node_i_global_index_x]), (Y_i + u_n[node_i_global_index_y])
            x_j, y_j = (X_j + u_n[node_j_global_index_x]), (Y_j + u_n[node_j_global_index_y])
            x_k, y_k = (X_k + u_n[node_k_global_index_x]), (Y_k + u_n[node_k_global_index_y])

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
                [dN_iX * y_i, dN_iY * y_i]
            ], dtype=np.float64)
            F_j = np.array([
                [dN_jX * x_j, dN_jY * x_j],
                [dN_jX * y_j, dN_jY * y_j]
            ], dtype=np.float64)
            F_k = np.array([
                [dN_kX * x_k, dN_kY * x_k],
                [dN_kX * y_k, dN_kY * y_k]
            ], dtype=np.float64)
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
            I = np.eye(2, dtype=np.float64)
            C = np.dot(np.transpose(F), F)
            E = (C - I) / 2.0

            # Compute S
            S = self.lambda_ * np.trace(E) * I + 2 * self.mu * E

            # Compute vector field gradients
            grad_i = np.array([dN_iX, dN_iY])
            grad_j = np.array([dN_jX, dN_jY])
            grad_k = np.array([dN_kX, dN_kY])


            res_i = A_e * np.dot(np.dot(F, S), grad_i)
            res_j = A_e * np.dot(np.dot(F, S), grad_j)
            res_k = A_e * np.dot(np.dot(F, S), grad_k)
            k_e = np.array([
                res_i[0],
                res_i[1],
                res_j[0],
                res_j[1],
                res_k[0],
                res_k[1]
            ], dtype=np.float64)

            return k_e, E, F

        # Computes all element stiffness matrices
        all_k_e_matrices = np.zeros([len(self.mesh_faces), 6], dtype=np.float64)
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
        k = np.zeros([2 * self.total_number_of_nodes], dtype=np.float64)
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
        matrix = np.zeros([2 * self.total_number_of_nodes, 2 * self.total_number_of_nodes], dtype=np.float64)
        assert(len(self.mesh_faces) == len(self.FEM_encoding))
        for i in range(len(self.FEM_encoding)):
            triangle_encoding = self.FEM_encoding[i]

            global_indices, ijk_indices = decode_triangle_indices(triangle_encoding, self.element_order)

            global_indices_list = []
            for j in range(len(global_indices)):
                global_indices_list.append(global_indices[j] * 2)
                global_indices_list.append(global_indices[j] * 2 + 1)
            global_indices_list = np.array(global_indices_list)

            matrix[global_indices_list.reshape([2*len(global_indices), 1]), global_indices_list] += all_M_e[i]

        return matrix

    def compute_body_forces(self, include_gravity=True):
        f_b = np.zeros([2 * self.total_number_of_nodes])

        # Number of nodes in the element
        m = int((self.element_order + 1) * (self.element_order + 2) / 2)

        # Add gravity force to all nodes
        if include_gravity:
            scheme = quadpy.t2.get_good_scheme(self.element_order + 1)

            def compute_element_gravity_term(face_index):
                triangle = self.mesh_points[self.mesh_faces[face_index]]

                triangle_encoding = self.FEM_encoding[face_index]
                global_indices, ijk_indices = decode_triangle_indices(triangle_encoding, self.element_order)

                N_int_values = np.zeros([2, 2*m])
                for i in range(len(global_indices)):
                    def f(x):
                        xi = cartesian_to_barycentric(x, triangle)
                        shape_function_val = silvester_shape_function(ijk_indices[i], xi,
                                                                  self.element_order)
                        return shape_function_val

                    int_val = scheme.integrate(f, triangle)
                    N_int_values[0, i*2] = int_val
                    N_int_values[1, 1 + i*2] = int_val

                f_g_e = self.material_properties.density * N_int_values.T@self.gravity

                return f_g_e

            all_gravity_terms = np.zeros([len(self.mesh_faces), 2*m], dtype=np.float64)
            for i in range(len(self.mesh_faces)):
                f_g_e = compute_element_gravity_term(i)
                all_gravity_terms[i] = f_g_e

            # Assemble the stiffness matrix
            f_g = np.zeros([2 * self.total_number_of_nodes], dtype=np.float64)
            for i in range(len(self.mesh_faces)):
                triangle_encoding = self.FEM_encoding[i]

                global_indices, ijk_indices = decode_triangle_indices(triangle_encoding,
                                                                      self.element_order)
                global_indices_list = []
                for j in range(len(global_indices)):
                    global_indices_list.append(global_indices[j] * 2)
                    global_indices_list.append(global_indices[j] * 2 + 1)
                global_indices_list = np.array(global_indices_list)

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

        num_internal_nodes = (self.element_order -1) * (self.element_order - 2) / 2

        def compute_element_traction(traction_encoding_index):
            traction_encoding = self.traction_encodings[traction_encoding_index]
            triangle_encoding = self.FEM_encoding[traction_encoding[0]]

            global_indices, ijk_indices = decode_triangle_indices(triangle_encoding, self.element_order)
            num_edge_nodes = self.element_order - 1

            i,j,k = global_indices[0:3]

            # If traction edge is ij-edge
            local_edge_indices = []
            if traction_encoding[1] == 0:
                local_edge_indices.append(0)
                edge_start_index = 4 + num_internal_nodes
                local_edge_indices.extend(np.arange(edge_start_index, edge_start_index + num_edge_nodes))
                local_edge_indices.append(1)
            # Elif traction edge is jk-edge
            elif traction_encoding[1] == 1:
                local_edge_indices.append(1)
                edge_start_index = 4 + num_internal_nodes + num_edge_nodes
                local_edge_indices.extend(np.arange(edge_start_index, edge_start_index + num_edge_nodes))
                local_edge_indices.append(2)
            # Elif traction edge is ki-edge
            elif traction_encoding[1] == 2:
                local_edge_indices.append(2)
                edge_start_index = 4 + num_internal_nodes + num_edge_nodes * 2
                local_edge_indices.extend(np.arange(edge_start_index, edge_start_index + num_edge_nodes))
                local_edge_indices.append(0)

            local_edge_indices = np.array(local_edge_indices)

            edge_indices = global_indices[local_edge_indices]
            # Assumes edge is fully vertical
            element_length = np.abs(self.FEM_V[edge_indices[0]][1] - self.FEM_V[edge_indices[-1]][1])

            scheme = quadpy.c1.gauss_patterson(self.element_order + 1)

            N_int_vals = np.zeros([2, len(local_edge_indices)*2])
            for l in range(len(local_edge_indices)):
                def f(x):
                    # Compute barycentric coordinates
                    xi = None
                    if traction_encoding[1] == 0:
                        xi_3 = x * 0
                        xi_1 = 1 - x/element_length
                        xi_2 = 1 - xi_1
                        xi = np.array([xi_1, xi_2, xi_3])
                    elif traction_encoding[1] == 1:
                        xi_1 = x * 0
                        xi_2 = 1 - x/element_length
                        xi_3 = 1 - xi_2
                        xi = np.array([xi_1, xi_2, xi_3])
                    elif traction_encoding[1] == 2:
                        xi_2 = x * 0
                        xi_3 = 1 - x/element_length
                        xi_1 = 1 - xi_3
                        xi = np.array([xi_1, xi_2, xi_3])

                    shape_function_val = silvester_shape_function(
                        ijk_indices[local_edge_indices[l]], xi, self.element_order)

                    return shape_function_val

                val = scheme.integrate(f, [0.0, element_length])

                N_int_vals[0, l*2] = val
                N_int_vals[1, 1 + l*2] = val

            f_t_e = N_int_vals.T@self.traction_force

            f_t_e_global_indices = global_indices[local_edge_indices]

            return f_t_e, f_t_e_global_indices

        m = int((self.element_order + 1) * (self.element_order + 2) / 2)
        all_traction_terms = []
        all_traction_indices = []
        for i in range(len(self.traction_encodings)):
            f_t_e, f_t_e_indices = compute_element_traction(i)
            all_traction_terms.append(f_t_e)
            all_traction_indices.append(f_t_e_indices)

        f_t = np.zeros([2 * self.total_number_of_nodes], dtype=np.float64)
        for i in range(len(all_traction_terms)):
            f_t_e = all_traction_terms[i]
            global_indices = all_traction_indices[i]

            global_indices_list = []
            for j in range(len(global_indices)):
                global_indices_list.append(global_indices[j] * 2)
                global_indices_list.append(global_indices[j] * 2 + 1)
            global_indices_list = np.array(global_indices_list)

            f_t[global_indices_list] += f_t_e

        return -f_t

    def compute_F(self, x_i, x_j, x_k, X_i, X_j, X_k):
        x_ij = (x_j - x_i)
        x_ik = (x_k - x_i)

        X_ij = (X_j - X_i)
        X_ik = (X_k - X_i)

        D = np.array([
            [x_ij[0], x_ik[0]],
            [x_ij[1], x_ik[1]],
        ], dtype=np.float64)

        D_0 = np.array([
            [X_ij[0], X_ik[0]],
            [X_ij[1], X_ik[1]],
        ], dtype=np.float64)

        F = D @ np.linalg.inv(D_0)
        return F

