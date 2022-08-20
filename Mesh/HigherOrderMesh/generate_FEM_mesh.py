import numpy as np

from Mesh.HigherOrderMesh.generate_ijk_indices import generate_ijk_indices
from Mesh.HigherOrderMesh.get_ijk_indices_for_ij_edge import get_ijk_indices_for_ij_edge
from Mesh.HigherOrderMesh.get_ijk_indices_for_internal_nodes import \
    get_ijk_indices_for_internal_nodes
from Mesh.HigherOrderMesh.get_ijk_indices_for_jk_edge import get_ijk_indices_for_jk_edge
from Mesh.HigherOrderMesh.get_ijk_indices_for_ki_edge import get_ijk_indices_for_ki_edge
from Mesh.HigherOrderMesh.get_vertex_from_ijk_index_and_triangle import \
    get_vertex_from_ijk_index_and_triangle


def generate_FEM_mesh(V, faces, n=1):
    """
    Generate a higher order triangle mesh.
    :param V:
    :param F:
    :param n:
    :return V_new, faces_encoding:
    """

    # The (i,j,k) indices for an element.
    ijk_indices = generate_ijk_indices(n)

    ij_edge_indices = get_ijk_indices_for_ij_edge(ijk_indices)
    jk_edge_indices = get_ijk_indices_for_jk_edge(ijk_indices)
    ki_edge_indices = get_ijk_indices_for_ki_edge(ijk_indices)
    internal_node_indices = get_ijk_indices_for_internal_nodes(ijk_indices)

    V_new = V.copy()
    faces_encoding = []

    # Edge look up table
    edge_LUT = dict()

    # Loop over triangle faces
    for face in faces:
        encoding = []

        i = face[0]
        j = face[1]
        k = face[2]
        encoding.extend([i,j,k])

        # Add internal nodes (if any) to the FEM mesh
        if len(internal_node_indices) > 0:
            encoding.append(len(V_new))
            for ijk_index in internal_node_indices:
                vertex = get_vertex_from_ijk_index_and_triangle(V[face], ijk_index, n)
                V_new = np.concatenate((V_new,[vertex]),axis=0)
        else:
            encoding.append(0)

        if n <= 2:
            # No edge nodes
            encoding.extend([-1, 0]*3)
        else:
            # Add the edge nodes (if any) to the FEM mesh
            ij_key = sorted((i,j))
            jk_key = sorted((j,k))
            ki_key = sorted((k,i))

            key_edges = [((ij_key[0],ij_key[1]), (i,j), ij_edge_indices[1:n]),
                         ((jk_key[0],jk_key[1]), (j,k), jk_edge_indices[1:n]),
                         ((ki_key[0],ki_key[1]), (k,i), ki_edge_indices[1:n])]
            for key_edge in key_edges:
                key, edge, edge_ijk_indices = key_edge

                orientation =1
                edge_offset = len(V_new)

                if key in edge_LUT:
                    existing_edge, edge_offset = edge_LUT[key]
                    if edge != existing_edge:
                        orientation = -1
                else:
                    # Add edge to LUT so other triangle that share the same edge won't duplicate the new
                    # vertices.
                    edge_LUT[key] = (edge, edge_offset)
                    for ijk_index in edge_ijk_indices:
                        vertex = get_vertex_from_ijk_index_and_triangle(V[face], ijk_index, n)
                        V_new = np.concatenate((V_new,[vertex]),axis=0)
                encoding.append(edge_offset)
                encoding.append(orientation)
        if len(encoding) < 10:
            raise Exception("Encoding is too short")
        faces_encoding.append(encoding)

    return V_new, faces_encoding
