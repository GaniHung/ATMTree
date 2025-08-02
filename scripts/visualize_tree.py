import pickle
import argparse
from graphviz import Digraph
from atm_tree_builder.data_structures import ATMTree, ATMNode

def visualize_tree(tree: ATMTree, output_filename: str):
    """Visualizes the ATMTree using graphviz."""
    dot = Digraph(comment='ATMTree')

    def add_nodes_edges(node: ATMNode):
        dot.node(str(node.id), f"ID: {node.id}\nLeaf: {node.is_leaf}\nEmbeddings: {node.num_embeddings}\nMethod: {node.generation_method}")
        for child in node.children:
            dot.edge(str(node.id), str(child.id))
            add_nodes_edges(child)

    add_nodes_edges(tree.root)
    dot.save(output_filename)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree_path", type=str, default="./data/atm_tree.pkl", help="Path to the ATMTree pkl file")
    parser.add_argument("--output_path", type=str, default="./data/atm_tree.gv", help="Path to save the graphviz file")
    args = parser.parse_args()

    # Load the ATMTree
    with open(args.tree_path, "rb") as f:
        atm_tree = pickle.load(f)

    visualize_tree(atm_tree, args.output_path)

if __name__ == "__main__":
    main()

