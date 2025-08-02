import numpy as np
from .data_structures import ATMTree, ATMNode
from .utils.math_ops import cosine_similarity

def search_tree(tree: ATMTree, query_embedding: np.ndarray, traversal_threshold: float, cutoff_threshold: float):
    """
    Performs a threshold-guided search on the ATM-Tree.

    Args:
        tree (ATMTree): The ATM-Tree to search in.
        query_embedding (np.ndarray): The embedding of the query.
        traversal_threshold (float): The similarity threshold to traverse down the tree.
        cutoff_threshold (float): The final similarity threshold for candidate selection.

    Returns:
        list[ATMNode]: A list of result nodes, sorted by similarity.
    """
    candidate_nodes = set()
    
    # Phase 1: Threshold-Guided Tree Traversal
    traversal_stack = [tree.root]
    while traversal_stack:
        current_node = traversal_stack.pop()
        candidate_nodes.add(current_node)
        
        if not current_node.is_leaf:
            for child in current_node.children:
                if child.retrieval_embedding is not None:
                    similarity = cosine_similarity(query_embedding, child.retrieval_embedding)
                    if similarity > traversal_threshold:
                        traversal_stack.append(child)

    # Phase 2: Final Candidate Selection
    result_nodes = []
    for node in candidate_nodes:
        similarity = cosine_similarity(query_embedding, node.retrieval_embedding)
        if similarity > cutoff_threshold:
            result_nodes.append(node)
            
    # Sort results by similarity
    result_nodes.sort(key=lambda n: cosine_similarity(query_embedding, n.retrieval_embedding), reverse=True)
    
    return result_nodes
