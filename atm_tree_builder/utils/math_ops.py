import numpy as np

def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Calculates the L2 norm of a vector."""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

def compute_centroid_direction(embeddings: list[np.ndarray]) -> np.ndarray:
    """Computes the centroid of a list of embeddings."""
    centroid = np.mean(embeddings, axis=0)
    return l2_normalize(centroid)

def synthesize_parent_embedding(child_embeddings: np.ndarray, sigma: float) -> np.ndarray:
    """
    Synthesizes a parent embedding from a cluster of child embeddings using a weighted average
    based on angular distance from the centroid.
    """
    # Compute the normalized centroid of the cluster
    centroid = compute_centroid_direction(child_embeddings)

    # Calculate weights for each child embedding
    angular_distances = np.arccos(np.clip(np.dot(child_embeddings, centroid), -1.0, 1.0))
    weights = np.exp(-angular_distances**2 / (2 * sigma**2))

    # Compute the weighted sum of child embeddings
    weighted_sum = np.dot(weights, child_embeddings)

    # Normalize to get the final parent embedding
    return l2_normalize(weighted_sum)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Computes the cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
