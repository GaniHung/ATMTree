import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModel
import torch
from atm_tree_builder.tree_builder import ATMTreeBuilder
from atm_tree_builder.config import ATMBuilderParameters
from atm_tree_builder.utils.math_ops import synthesize_parent_embedding

# --- Configuration ---
RETRIEVAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATIVE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# --- Dummy Data ---
DUMMY_PASSAGES = [
    "The capital of France is Paris.",
    "The currency of Japan is the Yen.",
    "The highest mountain in the world is Mount Everest.",
]

# --- Embedding Generation Functions ---
def get_retrieval_embeddings(texts, model, tokenizer, device='cpu'):
    model.to(device)
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()

def get_content_embeddings(texts, model, tokenizer, device='cpu'):
    model.to(device)
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()

def main():
    # --- Load Models ---
    retrieval_tokenizer = AutoTokenizer.from_pretrained(RETRIEVAL_MODEL)
    retrieval_model = AutoModel.from_pretrained(RETRIEVAL_MODEL)
    generative_tokenizer = AutoTokenizer.from_pretrained(GENERATIVE_MODEL)
    generative_model = AutoModel.from_pretrained(GENERATIVE_MODEL)

    # --- Generate Embeddings ---
    retrieval_embeddings = get_retrieval_embeddings(DUMMY_PASSAGES, retrieval_model, retrieval_tokenizer)
    content_embeddings = get_content_embeddings(DUMMY_PASSAGES, generative_model, generative_tokenizer)

    # --- Manual Calculation ---
    expected_root_w = synthesize_parent_embedding(retrieval_embeddings, sigma=0.5)
    expected_root_w_word = np.mean(content_embeddings, axis=0)

    with open("expected_w_and_w_content.txt", "w") as f:
        f.write("--- Expected Values ---\n")
        f.write(f"Root Retrieval (w):\n{expected_root_w}\n\n")
        f.write(f"Root Content (w_word):\n{expected_root_w_word}\n")

    # --- ATMTreeBuilder Calculation ---
    params = ATMBuilderParameters()
    builder = ATMTreeBuilder(params, input_dim=retrieval_embeddings.shape[1])
    tree = builder.build(retrieval_embeddings, content_embeddings)

    with open("real_w_and_w_content.txt", "w") as f:
        f.write("--- Real Values ---\n")
        f.write(f"Root Retrieval (w):\n{tree.root.retrieval_embedding}\n\n")
        f.write(f"Root Content (w_word):\n{tree.root.content_embedding}\n")

    print("Test complete. Please compare the contents of expected_w_and_w_content.txt and real_w_and_w_content.txt")

if __name__ == "__main__":
    main()
