import numpy as np
from atm_tree_builder.tree_builder import ATMTreeBuilder
from atm_tree_builder.config import ATMBuilderParameters

def test_tree_builder():
    # Create a small, synthetic dataset
    embeddings = np.array([
        [1, 1],
        [1, 2],
        [10, 10],
        [10, 11],
    ])

    # Build the tree
    params = ATMBuilderParameters()
    builder = ATMTreeBuilder(params)
    tree = builder.build(embeddings)

    # Add assertions to verify the tree structure
    assert tree.root is not None
    assert len(tree.root.children) > 0
