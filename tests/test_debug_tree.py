import numpy as np
import pickle
from atm_tree_builder.tree_builder import ATMTreeBuilder
from atm_tree_builder.config import ATMBuilderParameters

def test_debug_tree():
    # Load the real embeddings
    all_embeddings = np.load("data/corpus_embeddings.npy")

    # Use a small, diverse subset of the real embeddings
    indices = np.linspace(0, len(all_embeddings) - 1, 100, dtype=int)
    embeddings = all_embeddings[indices]

    # Build the tree
    params = ATMBuilderParameters()
    builder = ATMTreeBuilder(params, input_dim=embeddings.shape[1])
    tree = builder.build(embeddings)

    # Save the test tree
    with open("data/test_atm_tree.pkl", "wb") as f:
        pickle.dump(tree, f)

    # Print the tree structure
    print_tree(tree.root)

def print_tree(node, level=0):
    indent = "    " * level
    print(f"{indent}Node ID: {node.id}, Is Leaf: {node.is_leaf}, Embeddings: {node.num_embeddings}, Method: {node.generation_method}")
    if not node.is_leaf:
        print(f"{indent}Children: {len(node.children)}")
        for child in node.children:
            print_tree(child, level + 1)

if __name__ == "__main__":
    test_debug_tree()
