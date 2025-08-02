import numpy as np
from typing import List
from lshashpy3 import LSHash

class LSHClusterer:
    def __init__(self, n_projections: int, n_permutations: int, input_dim: int):
        self.lsh = LSHash(hash_size=n_projections, 
                          input_dim=input_dim, 
                          num_hashtables=n_permutations)

    def cluster(self, embeddings: np.ndarray) -> List[List[int]]:
        """
        Clusters embeddings using LSH by querying for neighbors for each point.
        This implementation ensures each point belongs to exactly one cluster.
        """
        # Index all embeddings, storing their original index in extra_data
        for i, emb in enumerate(embeddings):
            self.lsh.index(emb, extra_data=i)

        clusters = []
        processed_indices = set()

        for i in range(len(embeddings)):
            # If this point has already been added to a cluster, skip it
            if i in processed_indices:
                continue

            # Query for neighbors. The first result is usually the point itself.
            nn_results = self.lsh.query(embeddings[i], num_results=None, distance_func="cosine")
            
            # Create a new cluster. Start it with the current point.
            new_cluster = set([i])

            if nn_results:
                for result in nn_results:
                    # result is a tuple: ((vector, extra_data), distance)
                    if result and isinstance(result, tuple) and len(result) > 0:
                        extra_data = result[0][1]
                        if extra_data is not None:
                            neighbor_index = extra_data
                            # Add the neighbor to the cluster if it hasn't been processed yet
                            if neighbor_index not in processed_indices:
                                new_cluster.add(neighbor_index)
            
            # Mark all indices in the newly formed cluster as processed
            for index in new_cluster:
                processed_indices.add(index)
            
            # Add the new cluster to our list of clusters
            clusters.append(list(new_cluster))

        return [cluster for cluster in clusters if cluster]
