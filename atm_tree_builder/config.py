"""
Centralized configuration file for the ATMTree project.
"""
from dataclasses import dataclass

# --- Global Settings ---
RETRIEVAL_MODEL = "BAAI/bge-base-en-v1.5"
GENERATIVE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET = "nq_open"
OUTPUT_DIR = "data"

# --- Smoke Test Settings ---
SMOKE_TEST_RETRIEVAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SMOKE_TEST_GENERATIVE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SMOKE_TEST_DATASET = "scifact"


# --- Parameter Dataclasses ---

@dataclass
class ATMBuilderParameters:
    """Parameters for building the ATMTree."""
    lsh_n_projections: int = 16
    lsh_n_permutations: int = 128
    min_cluster_size: int = 2
    max_depth: int = 10
    sigma: float = 0.5

@dataclass
class ATMEvaluationParameters:
    """Parameters for evaluating the ATMTree."""
    traversal_threshold: float = 0.7
    cutoff_threshold: float = 0.8