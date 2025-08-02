import argparse
import pickle
import numpy as np
from atm_tree_builder.data_structures import ATMTree, ATMNode

def analyze_lsh_step(root: ATMNode):
    """Analyzes the first level of the tree (LSH buckets)."""
    print("--- LSH Step Analysis ---")
    if not root.children:
        print("The root node has no children.")
        return

    lsh_node_sizes = [child.num_embeddings for child in root.children]
    
    max_size = np.max(lsh_node_sizes)
    min_size = np.min(lsh_node_sizes)
    avg_size = np.mean(lsh_node_sizes)

    print(f"Number of LSH Buckets (Level 1 Nodes): {len(lsh_node_sizes)}")
    print(f"Max embeddings in a bucket: {max_size}")
    print(f"Min embeddings in a bucket: {min_size}")
    print(f"Avg embeddings in a bucket: {avg_size:.2f}")
    print("-------------------------")

def find_longest_path(root: ATMNode):
    """Finds and prints the longest path from root to leaf."""
    print("--- Longest Path Analysis ---")
    longest_path = []
    
    def dfs(node, current_path):
        nonlocal longest_path
        current_path.append(node)

        if node.is_leaf:
            if len(current_path) > len(longest_path):
                longest_path = list(current_path) # Make a copy
        else:
            for child in node.children:
                dfs(child, current_path)
        
        current_path.pop()

    dfs(root, [])

    if not longest_path:
        print("No paths found in the tree.")
        return

    print(f"Found longest path with depth: {len(longest_path) - 1}")
    for i, node in enumerate(longest_path):
        indent = "  " * i
        print(f"{indent}-> Node ID: {node.id}, Method: {node.generation_method}, Embeddings: {node.num_embeddings}")
    print("---------------------------")

def check_embedding_sample(retrieval_embeddings_path: str, content_embeddings_path: str):
    """Loads the corpus and checks the shape of a random sample."""
    print("--- Embedding Sample Check ---")
    try:
        retrieval_embeddings = np.load(retrieval_embeddings_path)
        content_embeddings = np.load(content_embeddings_path)
        print(f"Retrieval embeddings shape: {retrieval_embeddings.shape}")
        print(f"Content embeddings shape: {content_embeddings.shape}")
        
        if retrieval_embeddings.shape[0] >= 30:
            sample_indices = np.random.choice(retrieval_embeddings.shape[0], 30, replace=False)
            retrieval_sample = retrieval_embeddings[sample_indices]
            content_sample = content_embeddings[sample_indices]
            print(f"Random retrieval sample shape (n=30): {retrieval_sample.shape}")
            print(f"Random content sample shape (n=30): {content_sample.shape}")
        else:
            print("Corpus has fewer than 30 embeddings, cannot take sample.")
            
    except FileNotFoundError:
        print(f"Error: Embeddings file not found.")
    print("----------------------------")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree_path", type=str, default="./data/atm_tree.pkl")
    parser.add_argument("--retrieval_embeddings_path", type=str, default="./data/corpus_retrieval_embeddings.npy")
    parser.add_argument("--content_embeddings_path", type=str, default="./data/corpus_content_embeddings.npy")
    args = parser.parse_args()

    """Main function to run the analysis."""
    print("Analyzing the generated ATMTree...")
    try:
        with open(args.tree_path, "rb") as f:
            atm_tree = pickle.load(f)
        
        root = atm_tree.root
        analyze_lsh_step(root)
        find_longest_path(root)
        check_embedding_sample(args.retrieval_embeddings_path, args.content_embeddings_path)

    except FileNotFoundError:
        print(f"Error: {args.tree_path} not found. Please build the tree first.")

if __name__ == "__main__":
    main()
