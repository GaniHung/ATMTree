from typing import List
import numpy as np
from sklearn.cluster import KMeans
from atm_tree_builder.data_structures import ATMTree, ATMNode
from atm_tree_builder.config import ATMBuilderParameters
from atm_tree_builder.clustering.lsh import LSHClusterer
from atm_tree_builder.utils.math_ops import synthesize_parent_embedding

class ATMTreeBuilder:
    def __init__(self, params: ATMBuilderParameters, input_dim: int):
        self.params = params
        self.lsh_clusterer = LSHClusterer(params.lsh_n_projections, params.lsh_n_permutations, input_dim)

    def build(self, retrieval_embeddings: np.ndarray, content_embeddings: np.ndarray) -> ATMTree:
        """
        Builds the ATMTree from retrieval and content embeddings.
        """
        # Create a single root node to represent the entire dataset
        root_retrieval_embedding = synthesize_parent_embedding(retrieval_embeddings, sigma=self.params.sigma)
        root_content_embedding = np.mean(content_embeddings, axis=0)
        root = ATMNode(retrieval_embedding=root_retrieval_embedding, 
                       content_embedding=root_content_embedding, 
                       generation_method="Root")
        root.num_embeddings = len(retrieval_embeddings)

        # Phase 1: Coarse-grained partitioning with LSH
        lsh_buckets = self.lsh_clusterer.cluster(retrieval_embeddings)

        # Phase 2: For each bucket, create a Level 1 node and build a binary subtree under it
        for bucket in lsh_buckets:
            if not bucket:
                continue

            if len(bucket) == 1:
                # If the bucket has only one embedding, create a leaf node directly
                leaf_retrieval_embedding = retrieval_embeddings[bucket[0]]
                leaf_content_embedding = content_embeddings[bucket[0]]
                leaf_node = ATMNode(retrieval_embedding=leaf_retrieval_embedding, 
                                  content_embedding=leaf_content_embedding, 
                                  is_leaf=True)
                root.children.append(leaf_node)
                leaf_node.parent = root
                continue

            bucket_retrieval_embeddings = np.array([retrieval_embeddings[i] for i in bucket])
            bucket_content_embeddings = np.array([content_embeddings[i] for i in bucket])
            
            # Create a parent for the LSH bucket (Level 1 Node)
            level1_retrieval_embedding = synthesize_parent_embedding(bucket_retrieval_embeddings, sigma=self.params.sigma)
            level1_content_embedding = np.mean(bucket_content_embeddings, axis=0)
            level1_node = ATMNode(retrieval_embedding=level1_retrieval_embedding, 
                                  content_embedding=level1_content_embedding, 
                                  generation_method="LSH")
            level1_node.num_embeddings = len(bucket_retrieval_embeddings)
            
            root.children.append(level1_node)
            level1_node.parent = root

            # Build the binary subtree under this Level 1 node
            self._build_binary_subtree(level1_node, bucket_retrieval_embeddings, bucket_content_embeddings, 1)
        
        # Assign IDs using a pre-order traversal
        self._assign_ids_pre_order(root)

        return ATMTree(root)

    def _assign_ids_pre_order(self, node: ATMNode, current_id: int = 0) -> int:
        """
        Assigns IDs to nodes in a pre-order traversal.
        """
        node.id = current_id
        next_id = current_id + 1
        for child in node.children:
            next_id = self._assign_ids_pre_order(child, next_id)
        return next_id

    def _build_binary_subtree(self, parent_node: ATMNode, retrieval_data_partition: np.ndarray, content_data_partition: np.ndarray, depth: int):
        """
        Recursively builds a binary subtree using Bisecting K-Means.
        """
        # Base case: if the partition is too small or max depth is reached, create leaf nodes.
        if depth >= self.params.max_depth or len(retrieval_data_partition) < self.params.min_cluster_size:
            for i in range(len(retrieval_data_partition)):
                leaf_node = ATMNode(retrieval_embedding=retrieval_data_partition[i], 
                                  content_embedding=content_data_partition[i], 
                                  is_leaf=True)
                parent_node.children.append(leaf_node)
                leaf_node.parent = parent_node
            return

        # Perform K-Means clustering on the retrieval embeddings
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(retrieval_data_partition)
        
        left_cluster_indices = np.where(kmeans.labels_ == 0)[0]
        right_cluster_indices = np.where(kmeans.labels_ == 1)[0]

        left_retrieval_cluster = retrieval_data_partition[left_cluster_indices]
        right_retrieval_cluster = retrieval_data_partition[right_cluster_indices]
        left_content_cluster = content_data_partition[left_cluster_indices]
        right_content_cluster = content_data_partition[right_cluster_indices]

        # If K-Means fails to create a valid split, treat the current partition as a leaf node.
        if len(left_retrieval_cluster) == 0 or len(right_retrieval_cluster) == 0:
            for i in range(len(retrieval_data_partition)):
                leaf_node = ATMNode(retrieval_embedding=retrieval_data_partition[i], 
                                  content_embedding=content_data_partition[i], 
                                  is_leaf=True)
                parent_node.children.append(leaf_node)
                leaf_node.parent = parent_node
            return

        # Create and recurse on the left child
        left_child_retrieval_embedding = synthesize_parent_embedding(left_retrieval_cluster, sigma=self.params.sigma)
        left_child_content_embedding = np.mean(left_content_cluster, axis=0)
        left_child = ATMNode(retrieval_embedding=left_child_retrieval_embedding, 
                               content_embedding=left_child_content_embedding, 
                               generation_method="K-Means")
        left_child.num_embeddings = len(left_retrieval_cluster)
        parent_node.children.append(left_child)
        left_child.parent = parent_node
        self._build_binary_subtree(left_child, left_retrieval_cluster, left_content_cluster, depth + 1)

        # Create and recurse on the right child
        right_child_retrieval_embedding = synthesize_parent_embedding(right_retrieval_cluster, sigma=self.params.sigma)
        right_child_content_embedding = np.mean(right_content_cluster, axis=0)
        right_child = ATMNode(retrieval_embedding=right_child_retrieval_embedding, 
                                content_embedding=right_child_content_embedding, 
                                generation_method="K-Means")
        right_child.num_embeddings = len(right_retrieval_cluster)
        parent_node.children.append(right_child)
        right_child.parent = parent_node
        self._build_binary_subtree(right_child, right_retrieval_cluster, right_content_cluster, depth + 1)
