import math

import numpy as np


def cross_product_z(vec1, vec2):
    return vec1[0]*vec2[1] - vec1[1]*vec2[0]


def compute_triangle_element_area(vertices, triangle_face):
    p_i = vertices[triangle_face[0]]
    p_j = vertices[triangle_face[1]]
    p_k = vertices[triangle_face[2]]

    vec_i_to_j = p_j - p_i
    vec_j_to_k = p_k - p_j

    area = 0.5 * cross_product_z(vec_i_to_j, vec_j_to_k)

    return np.absolute(area)


def compute_all_element_areas(vertices, triangle_faces):
    element_areas = np.zeros(triangle_faces.shape[0], dtype=np.float64)

    for i in range(triangle_faces.shape[0]):
        triangle_face_i = triangle_faces[i]
        element_areas[i] = compute_triangle_element_area(vertices, triangle_face_i)

    return element_areas


def validate_area(beam_length, beam_height, computed_areas):
    true_area = beam_length * beam_height
    computed_area = np.sum(computed_areas, dtype=np.float64)

    diff = true_area - computed_area
    if math.isclose(true_area, computed_area):
        return True, diff
    return False, diff