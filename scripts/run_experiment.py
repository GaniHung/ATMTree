
import argparse
import os
import pickle
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from atm_tree_builder.config import (
    ATMBuilderParameters,
    ATMEvaluationParameters,
    RETRIEVAL_MODEL,
    DATASET,
    OUTPUT_DIR,
    SMOKE_TEST_RETRIEVAL_MODEL,
    SMOKE_TEST_DATASET
)
from atm_tree_builder.tree_builder import ATMTreeBuilder
from atm_tree_builder.search import search_tree

def prepare_data(model_name: str, dataset_name: str, output_dir: str):
    """
    Loads a dataset, generates embeddings for the corpus, and saves them.
    """
    print(f"--- Stage: Preparing Data ---")
    print(f"Model: {model_name}, Dataset: {dataset_name}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    corpus_embeddings_file = os.path.join(output_dir, "corpus_embeddings.npy")
    corpus_contexts_file = os.path.join(output_dir, "corpus_contexts.pkl")

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(dataset_name, split="train")
    corpus = [doc['text'] for doc in dataset]

    # Load tokenizer and model
    print("Loading retrieval model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Generate embeddings
    print("Generating embeddings...")
    with torch.no_grad():
        embeddings = []
        for text in tqdm(corpus):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
            outputs = model(**inputs)
            # Mean pooling
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            embeddings.append(embedding)
    
    embeddings = np.array(embeddings)

    # Save embeddings and contexts
    print(f"Saving embeddings to {corpus_embeddings_file}")
    np.save(corpus_embeddings_file, embeddings)

    print(f"Saving contexts to {corpus_contexts_file}")
    with open(corpus_contexts_file, "wb") as f:
        pickle.dump(corpus, f)
    
    print("--- Data Preparation Complete ---")


def build_tree(output_dir: str):
    """
    Loads corpus embeddings and builds the ATMTree.
    """
    print(f"--- Stage: Building Tree ---")
    embeddings_file = os.path.join(output_dir, "corpus_embeddings.npy")
    tree_file = os.path.join(output_dir, "atm_tree.pkl")

    print(f"Loading embeddings from {embeddings_file}...")
    embeddings = np.load(embeddings_file)

    print("Building ATMTree...")
    params = ATMBuilderParameters()
    builder = ATMTreeBuilder(params)
    atm_tree = builder.build_tree(embeddings)

    print(f"Saving tree to {tree_file}")
    with open(tree_file, "wb") as f:
        pickle.dump(atm_tree, f)

    print("--- Tree Building Complete ---")


def run_evaluation(output_dir: str):
    """
    Loads the ATMTree and runs an evaluation.
    """
    print(f"--- Stage: Running Evaluation ---")
    tree_file = os.path.join(output_dir, "atm_tree.pkl")
    embeddings_file = os.path.join(output_dir, "corpus_embeddings.npy")

    print(f"Loading tree from {tree_file}...")
    with open(tree_file, "rb") as f:
        atm_tree = pickle.load(f)

    print(f"Loading embeddings from {embeddings_file}...")
    embeddings = np.load(embeddings_file)
    
    params = ATMEvaluationParameters()

    # This is a placeholder for a real evaluation.
    # For now, we'll just search for the first 5 embeddings as queries.
    print("Running placeholder evaluation...")
    for i in range(5):
        query_embedding = embeddings[i]
        results = search_tree(atm_tree, query_embedding, params)
        print(f"Query {i}: Found {len(results)} results.")

    print("--- Evaluation Complete ---")


def main():
    """
    Main function to run the experiment stages.
    """
    parser = argparse.ArgumentParser(description="Run the ATMTree experiment pipeline.")
    parser.add_argument(
        "--stage",
        type=str,
        choices=["prepare", "build", "evaluate", "all"],
        required=True,
        help="The stage of the experiment to run."
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use smaller models and datasets for a quick smoke test."
    )
    args = parser.parse_args()

    if args.smoke_test:
        model = SMOKE_TEST_RETRIEVAL_MODEL
        dataset = SMOKE_TEST_DATASET
        output_dir = "data/smoke_test"
    else:
        model = RETRIEVAL_MODEL
        dataset = DATASET
        output_dir = OUTPUT_DIR

    if args.stage == "prepare" or args.stage == "all":
        prepare_data(model_name=model, dataset_name=dataset, output_dir=output_dir)
    
    if args.stage == "build" or args.stage == "all":
        build_tree(output_dir=output_dir)

    if args.stage == "evaluate" or args.stage == "all":
        run_evaluation(output_dir=output_dir)

if __name__ == "__main__":
    main()
