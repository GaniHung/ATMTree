import os
import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
from atm_tree_builder import config
# --- Configuration ---

# --- Dummy Data ---
DUMMY_PASSAGES = [
    "The capital of France is Paris.",
    "The currency of Japan is the Yen.",
    "The highest mountain in the world is Mount Everest.",
    "The largest ocean in the world is the Pacific Ocean.",
    "The most populous country in the world is China.",
    "The longest river in the world is the Nile River.",
    "The largest desert in the world is the Antarctic Polar Desert.",
    "The smallest country in the world is Vatican City.",
    "The most spoken language in the world is Mandarin Chinese.",
    "The most visited city in the world is Bangkok."
]

# --- Embedding Generation Functions ---
def get_retrieval_embeddings(texts, model, tokenizer, device='cpu', batch_size=32):
    """Generates sentence-level retrieval embeddings."""
    model.to(device)
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating Retrieval Embeddings"):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        all_embeddings.append(embeddings.cpu().numpy())
    return np.vstack(all_embeddings)

def get_content_embeddings(texts, model, tokenizer, device='cpu', batch_size=32):
    """Generates word-level content embeddings."""
    model.to(device)
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating Content Embeddings"):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state
        all_embeddings.append(embeddings.cpu().numpy())
    return np.vstack(all_embeddings)

def main():
    """Main function to prepare the dummy data."""
    # --- Load Models ---
    print(f"--- Loading Models ---")
    print(f"Loading retrieval model: {config.RETRIEVAL_MODEL}...")
    retrieval_tokenizer = AutoTokenizer.from_pretrained(config.RETRIEVAL_MODEL)
    retrieval_model = AutoModel.from_pretrained(config.RETRIEVAL_MODEL)
    print("Retrieval model loaded.")

    print(f"Loading generative model for content: {config.SMOKE_TEST_GENERATIVE_MODEL}...")
    content_tokenizer = AutoTokenizer.from_pretrained(config.SMOKE_TEST_GENERATIVE_MODEL)
    content_model = AutoModel.from_pretrained(config.SMOKE_TEST_GENERATIVE_MODEL)
    print("Content model loaded.")
    print("----------------------")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    retrieval_model.to(device)
    content_model.to(device)

    # --- Generate Embeddings ---
    retrieval_embeddings = get_retrieval_embeddings(DUMMY_PASSAGES, retrieval_model, retrieval_tokenizer, device=device)
    content_embeddings = get_content_embeddings(DUMMY_PASSAGES, content_model, content_tokenizer, device=device)

    # --- Save Data ---
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    retrieval_embeddings_path = os.path.join(config.OUTPUT_DIR, 'test_corpus_retrieval_embeddings.npy')
    content_embeddings_path = os.path.join(config.OUTPUT_DIR, 'test_corpus_content_embeddings.npy')
    contexts_path = os.path.join(config.OUTPUT_DIR, 'test_corpus_contexts.pkl')

    np.save(retrieval_embeddings_path, retrieval_embeddings)
    np.save(content_embeddings_path, content_embeddings)
    with open(contexts_path, 'wb') as f:
        pickle.dump(DUMMY_PASSAGES, f)

    print(f"Test retrieval embeddings saved to {retrieval_embeddings_path}")
    print(f"Test content embeddings saved to {content_embeddings_path}")
    print(f"Test contexts saved to {contexts_path}")

if __name__ == "__main__":
    main()
