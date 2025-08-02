import numpy as np
from typing import List
from atm_tree_builder.data_structures import ATMNode
from atm_tree_builder.utils.math_ops import compute_centroid_direction

def create_parent_from_cluster(cluster: List[ATMNode], sigma: float) -> ATMNode:
    """Creates a parent node from a cluster of child nodes."""
    child_embeddings = [node.embedding for node in cluster]
    centroid = compute_centroid_direction(child_embeddings)

    weighted_sum = np.zeros_like(centroid)
    for child in cluster:
        angular_distance = np.arccos(np.clip(np.dot(child.embedding, centroid), -1.0, 1.0))
        weight = np.exp(-(angular_distance**2) / (2 * sigma**2))
        weighted_sum += weight * child.embedding

    parent_embedding = weighted_sum / np.linalg.norm(weighted_sum)

    # The ID of the parent node will be assigned later in the tree building process
    parent_node = ATMNode(id=-1, embedding=parent_embedding)

    return parent_node