from typing import List
from atm_tree_builder.data_structures import ATMNode
from atm_tree_builder.config import ATMBuilderParameters
from sklearn.cluster import KMeans
import numpy as np

class Refiner:
    def __init__(self, params: ATMBuilderParameters):
        self.params = params

    def refine_bucket(self, bucket: List[ATMNode]) -> List[ATMNode]:
        """Recursively refines a bucket of nodes using bisecting K-Means."""
        return self._refine_cluster(bucket, 0)

    def _refine_cluster(self, cluster: List[ATMNode], depth: int) -> List[ATMNode]:
        if depth >= self.params.max_depth or len(cluster) <= self.params.min_cluster_size:
            return cluster

        # Bisect the cluster using K-Means
        embeddings = np.array([node.embedding for node in cluster])
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(embeddings)
        labels = kmeans.labels_

        left_cluster = [node for i, node in enumerate(cluster) if labels[i] == 0]
        right_cluster = [node for i, node in enumerate(cluster) if labels[i] == 1]

        # Recursively refine the sub-clusters
        refined_nodes = []
        if left_cluster:
            refined_nodes.extend(self._refine_cluster(left_cluster, depth + 1))
        if right_cluster:
            refined_nodes.extend(self._refine_cluster(right_cluster, depth + 1))

        return refined_nodes