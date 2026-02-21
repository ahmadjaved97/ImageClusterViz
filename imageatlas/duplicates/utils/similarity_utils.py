"""
Similarity computation utilities.
"""

import numpy as np

def cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two vectors.
    """

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)

def euclidean_distance(vec1, vec2):
    """
    Compute euclidean distance between two vectors.
    """

    return np.linalg.norm(vec1 - vec2)

def euclidean_similarity(vec1, vec2):
    """
    Convert euclidean distance to similarity score (0-1).
    """
    distance = euclidean_distance(vec1, vec2)
    return 1.0 / (1.0 + distance)

def pairwise_similarity(
    features,
    metric = 'cosine',
    batch_size=1000
):
    """
    Compute pairwise similarity matrix for all features.
    """

    n_samples = features.shape[0]
    similarity_matrix = np.zeros((n_samples, n_samples), dtype=np.float32)

    if metric == 'cosine':
        # Normalize features for cosine similarity
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1   # Avoid division by 0
        normalized_features = features / norms

        # Compute in batches
        for i in range(0, n_samples, batch_size):
            end_i = min(i + batch_size, n_samples)
            batch = normalized_features[i:end_i]

            # Cosine similarity
            similarity_matrix[i:end_i, :] = np.dot(batch, normalized_features.T)
    
    elif metric == 'euclidean':
        # Compute in batches
        for i in range(0, n_samples, batch_size):
            end_i = min(i + batch_size, n_samples)
            batch = features[i:end_i]


            # Euclidean Distance
            for j in range(n_samples):
                distances = np.linalg.norm(batch - features[j], axis=1)
                similarity_matrix[i:end_i, j] = 1.0 / (1.0 + distances)
    
    else:
        raise ValueError(f"Unknown metric: {metric}")
    

    return similarity_matrix

def find_pairs_above_threshold(
    similarity_matrix,
    threshold,
    filenames
):
    """
    Extract pairs above similarity threshold from the matrix.
    """

    pairs = {}
    n_samples = similarity_matrix.shape[0]

    for i in range(n_samples):
        for j in range(i+1, n_samples):
            similarity = similarity_matrix[i, j]

            if similarity >= threshold:
                img1 = filenames[i]
                img2 = filenames[j]

                # Add both directions (symmetric)
                if img1 not in pairs:
                    pairs[img1] = []
                pairs[img1].append((img2, float(similarity)))

                if img2 not in pairs:
                    pairs[img2] = []
                
                pairs[img2].append((img1, float(similarity)))
    
    return pairs

def compute_similarity_statistics(similarity_matrix):
    """
    Compute statistics about similarity distribution.
    """

    # Extract upper triangle.
    n = similarity_matrix.shape[0]
    upper_triangle = similarity_matrix[np.triu_indices(n, k=1)]

    return {
        'mean': float(np.mean(upper_triangle)),
        'median': float(np.median(upper_triangle)),
        'std': float(np.std(upper_triangle)),
        'min': float(np.min(upper_triangle)),
        'max': float(np.max(upper_triangle)),
        'percentile_95': float(np.percentile(upper_triangle, 95)),
        'percentile_99': float(np.percentile(upper_triangle, 99)),
    }