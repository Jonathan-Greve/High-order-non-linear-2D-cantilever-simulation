# Simulator class
# Containts the main loop of the simulator called simulate
import numpy as np
import quadpy
from scipy.spatial import Delaunay
from tqdm import tqdm

from scipy import optimize

from Mesh.Cantilever.area_computations import compute_triangle_element_area, \
    compute_all_element_areas
from Mesh.Cantilever.generate_2d_cantilever_delaunay import generate_2d_cantilever_delaunay
from Mesh.Cantilever.generate_2d_cantilever_kennys import generate_2d_cantilever_kennys
from Mesh.HigherOrderMesh.decode_triangle_indices import decode_triangle_indices
from Mesh.HigherOrderMesh.generate_FEM_mesh import generate_FEM_mesh
from Simulator.HigherOrderElements.shape_functions import silvester_shape_function, \
    shape_function_spatial_derivative, vandermonde_shape_function, vandermonde_spatial_derivative, \
    vandermonde_shape_function_1D
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
        # points, faces = generate_2d_cantilever_delaunay(self.length, self.height,
        #                                              self.number_of_nodes_x, self.number_of_nodes_y)
        points, faces = generate_2d_cantilever_kennys(self.length, self.height,
                                                        self.number_of_nodes_x, self.number_of_nodes_y)
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
        self.boundary_indices = np.append(self.dirichlet_boundary_indices_x,
                                     self.dirichlet_boundary_indices_y)

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
        # M[M < 0] = 0
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
        # a_n[np.arange(1, self.total_number_of_nodes * 2, 2)] = self.gravity[1]
        X_0 = self.FEM_V.reshape([self.total_number_of_nodes * 2])
        x_n = X_0

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
            k, E = self.compute_stiffness_matrix(x_n)
            # k[self.boundary_indices] = 0

            # Do simulation step
            damping_term = np.dot(C, v_n)

            # # Remove all forces after 1 sec.
            # if (i * self.time_step > 1):
            #     f = f * 0

            forces = f - damping_term - k
            # forces[self.boundary_indices] = 0
            # forces[np.abs(forces) < 1e-30] = 0

            # def fun(x_vec):
            #     x_next = x_vec[0:len(x_n)]
            #     v_next = x_vec[len(x_n):len(x_n)*2]
            #
            #     # k, _ = self.compute_stiffness_matrix(x_next)
            #
            #     x_res = x_next - x_n - time_step_size * v_next
            #
            #     v_res = (M + time_step_size * C)@v_next - M@v_n - time_step_size*forces + time_step_size* k *(x_next - X_0)
            #     result = np.append(x_res, v_res)
            #
            #     return result
            #
            # input_guess = np.append(x_n, v_n)
            # sol = optimize.root(fun, input_guess, method='hybr')
            # x_n_1 = sol.x[0:len(x_n)]
            # x_n_1[self.boundary_indices] = X_0[self.boundary_indices]
            # v_n_1 = sol.x[len(x_n):len(x_n)*2]
            # # v_n_1 = (x_n_1 - x_n) / time_step_size
            # v_n_1[self.boundary_indices] = 0
            # v_n_1[np.abs(v_n_1) < 1e-10] = 0
            #
            # a_n_1 = (v_n_1 - v_n) / time_step_size


            a_n_1 = np.dot(Minv,  forces)
            v_n_1 = v_n + self.time_step * a_n_1 + 1e-10
            v_n_1[self.dirichlet_boundary_indices_x] = 0
            v_n_1[self.dirichlet_boundary_indices_y] = 0
            v_n_1[np.abs(v_n_1) < 1e-10] = 0

            x_n_1 = x_n + self.time_step * v_n_1

            # New displacements
            u_n = x_n_1 - X_0

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

        return Result(times, np.array(displacements), velocities, accelerations, Es, Ms, damping_forces, f_g)

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

        scheme = quadpy.t2.get_good_scheme(self.element_order*2+1)

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
                        # xi = cartesian_to_barycentric(x, triangle)
                        # shape_function = silvester_shape_function(ijk_indices[int(i / 2)], xi, self.element_order)
                        shape_function1 = vandermonde_shape_function(V_e, x, self.element_order)[:,int(i / 2)]
                        # assert(np.allclose(shape_function, shape_function1))

                        return shape_function1 ** 2
                else:
                    def f(x):
                        # xi = cartesian_to_barycentric(x, triangle)
                        # shape_function_1 = silvester_shape_function(ijk_indices[int(i / 2)], xi, self.element_order)
                        # shape_function_2 = silvester_shape_function(ijk_indices[int(j / 2)], xi, self.element_order)

                        shape_function_v1 = vandermonde_shape_function(self.FEM_V[global_indices], x, self.element_order)[:,int(i / 2)].T
                        shape_function_v2 = vandermonde_shape_function(self.FEM_V[global_indices], x, self.element_order)[:,int(j / 2)].T

                        # assert(np.allclose(shape_function_1, shape_function_v1))
                        # assert(np.allclose(shape_function_2, shape_function_v2))

                        return shape_function_v1 * shape_function_v2

                # if i*2 in self.boundary_indices or j*2 in self.boundary_indices:
                #     integral_N_square[i, j] = 0.1
                # else:
                #     integral_N_square[i,j] = scheme.integrate(f, triangle)
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

    def compute_stiffness_matrix(self, x_n):
        m = int((self.element_order + 1) * (self.element_order + 2) / 2)
        def compute_element_stiffness_matrix(face_index):
            triangle_encoding = self.FEM_encoding[face_index]
            global_indices, ijk_indices = decode_triangle_indices(triangle_encoding, self.element_order)
            triangle = self.FEM_V[global_indices[0:3]]

            # Reference vertices (non-deformed vertices)
            V_e = self.FEM_V[global_indices]
            v_e = []
            for i in range(m):
                x = x_n[global_indices[i]*2]
                y = x_n[global_indices[i]*2+1]
                v_e.append(np.array([x,y]))
            v_e = np.array(v_e)


            I = np.eye(2, dtype=np.float64)
            # C = np.dot(np.transpose(F), F)
            # E = (C - I) / 2.0
            #
            # # Compute S
            # S_v = self.lambda_ * np.trace(E) * I + 2 * self.mu * E # Saint Venant–Kirchhoff model
            #
            # # Compressible neo-hookean model (page 163 [Bonet and Wood,2008], Eq. 6.28)
            # J = np.linalg.det(F)
            # S_comp_neo = self.mu * (np.eye(2) - np.linalg.inv(C)) + self.lambda_ * np.log(J) * np.linalg.inv(C)
            #
            # # Incompressible neo-hookean model (page 169 [Bonet and Wood,2008], Eq. 6.52)
            # I_C = np.trace(C)
            # III_C = np.linalg.det(C)
            # III_C2 = J ** 2
            # k = 10**3
            # p = k * (J - 1)
            # S_incomp_neo = self.mu * III_C ** (-1/3) * (np.eye(2) - (1/3) * I_C * np.linalg.inv(C)) * p * J * np.linalg.inv(C)
            #
            # S = S_v




            scheme = quadpy.t2.get_good_scheme(self.element_order+1)
            quad_points = scheme.points
            quad_weights = scheme.weights

            A_e = self.all_A_e[face_index]

            u_e = (v_e - V_e)

            k_e = np.zeros(2*m)
            E = np.zeros((2,2))
            F = np.zeros((2,2))
            for i in range(m):
                def f(x):
                    # dN = vandermonde_spatial_derivative(V_e, x, self.element_order).T
                    dN = shape_function_spatial_derivative(V_e, triangle_encoding,x, self.element_order)

                    # F = np.zeros((2,2))
                    # for j in range(m):
                    #     F_j = np.array([
                    #         [dN[j,0]*v_e[j,0], dN[j,1]*v_e[j,0]],
                    #         [dN[j,0]*v_e[j,1], dN[j,1]*v_e[j,1]]
                    #     ])
                    #     F += F_j

                    F = np.eye(2)
                    for j in range(m):
                        F_j = np.array([
                            [dN[j,0]*u_e[j,0], dN[j,1]*u_e[j,0]],
                            [dN[j,0]*u_e[j,1], dN[j,1]*u_e[j,1]]
                        ])
                        F += F_j

                    C = np.dot(np.transpose(F), F)
                    E = (C - I) / 2.0
                    S = self.lambda_ * np.trace(E) * I + 2 * self.mu * E

                    # J = np.linalg.det(F)
                    # S = self.mu * (np.eye(2) - np.linalg.inv(C)) + self.lambda_ * np.log(
                    #     J) * np.linalg.inv(C)

                    # dN = shape_function_spatial_derivative(V_e, triangle_encoding, x,
                    #                                    self.element_order)
                    # dN1 = vandermonde_spatial_derivative(V_e, x, self.element_order).T
                    # assert (np.allclose(dN, dN1))
                    P = F @ S
                    result = (P @ dN[i])

                    return result

                val = 0
                for j in range(len(quad_points[0])):
                    point_bary = quad_points[:,j]
                    point = triangle[0] * point_bary[0] + \
                            triangle[1] * point_bary[1] + \
                            triangle[2] * point_bary[2]
                    f_val = f(point)
                    val += f_val * quad_weights[j]

                val *= A_e

                # val_x = scheme.integrate(f_x, triangle)
                # val_y = scheme.integrate(f_y, triangle)

                k_e[2*i] = val[0]
                k_e[2*i+1] = val[1]

            return k_e, E, F

        # Computes all element stiffness matrices
        all_k_e_matrices = np.zeros([len(self.mesh_faces), 2*m], dtype=np.float64)
        all_Es = np.zeros([len(self.mesh_faces), 2, 2])
        all_Fs = np.zeros([len(self.mesh_faces), 2, 2])
        for i in range(len(self.mesh_faces)):
            k_e, E, F = compute_element_stiffness_matrix(i)
            all_k_e_matrices[i] = k_e
            all_Es[i] = E
            all_Fs[i] = F

        # Assemble the stiffness matrix
        k = np.zeros([2 * self.total_number_of_nodes], dtype=np.float64)
        for i in range(len(all_k_e_matrices)):
            triangle_encoding = self.FEM_encoding[i]

            global_indices, ijk_indices = decode_triangle_indices(triangle_encoding,
                                                                  self.element_order)
            global_indices_list = []
            for j in range(len(global_indices)):
                global_indices_list.append(global_indices[j] * 2)
                global_indices_list.append(global_indices[j] * 2 + 1)
            global_indices_list = np.array(global_indices_list)

            k[global_indices_list] += all_k_e_matrices[i]


        # print("F: {}".format(all_Fs[0]))
        # print("CE: {}".format(all_Es[0]))
        # print("F: {}".format(all_Fs[1]))
        # print("CE: {}".format(all_Es[1]))
        # print('-------------------------------------')
        # if not np.isclose(np.sum(k), 0, atol=1e-5):
        #     print('here')
        # assert(np.isclose(np.sum(k), 0, atol=1e-5))
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
                        # xi = cartesian_to_barycentric(x, triangle)
                        # shape_function_val = silvester_shape_function(ijk_indices[i], xi,
                        #                                           self.element_order)
                        shape_function_val = vandermonde_shape_function(self.FEM_V[global_indices], x, self.element_order)[:,i]
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
                edge_start_index = 3 + num_internal_nodes
                local_edge_indices.extend(np.arange(edge_start_index, edge_start_index + num_edge_nodes))
                local_edge_indices.append(1)
            # Elif traction edge is jk-edge
            elif traction_encoding[1] == 1:
                local_edge_indices.append(1)
                edge_start_index = 3 + num_internal_nodes + num_edge_nodes
                local_edge_indices.extend(np.arange(edge_start_index, edge_start_index + num_edge_nodes))
                local_edge_indices.append(2)
            # Elif traction edge is ki-edge
            elif traction_encoding[1] == 2:
                local_edge_indices.append(2)
                edge_start_index = 3 + num_internal_nodes + num_edge_nodes * 2
                local_edge_indices.extend(np.arange(edge_start_index, edge_start_index + num_edge_nodes))
                local_edge_indices.append(0)

            local_edge_indices = np.array(local_edge_indices, dtype=np.int32)

            edge_indices = global_indices[local_edge_indices]
            # Assumes edge is fully vertical
            element_length = np.abs(self.FEM_V[edge_indices[0]][1] - self.FEM_V[edge_indices[-1]][1])


            sub_element_length = element_length / self.element_order
            edge_points = []
            for i in range(self.element_order + 1):
                edge_points.append(0 + sub_element_length * i)
            edge_points = np.array(edge_points)

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

                    # shape_function_val = silvester_shape_function(
                    #     ijk_indices[local_edge_indices[l]], xi, self.element_order)

                    shape_function_val1 = vandermonde_shape_function_1D(edge_points, x, self.element_order)[l]
                    # assert(np.allclose(shape_function_val, shape_function_val1))

                    return shape_function_val1

                scheme = quadpy.c1.gauss_legendre(self.element_order + 1)
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