import numpy as np
from atm_tree_builder.utils import math_ops

def test_l2_normalize():
    vec = np.array([3, 4])
    normalized_vec = math_ops.l2_normalize(vec)
    assert np.allclose(normalized_vec, np.array([0.6, 0.8]))
    assert np.isclose(np.linalg.norm(normalized_vec), 1.0)

def test_compute_centroid_direction():
    embeddings = [np.array([1, 1]), np.array([1, 3])]
    centroid = math_ops.compute_centroid_direction(embeddings)
    assert np.allclose(centroid, np.array([0.4472136, 0.89442719]))
