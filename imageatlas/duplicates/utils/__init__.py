"""
Utility function for duplicate detection.
"""

from .hash_utils import (
    compute_crypto_hash,
    compute_average_hash,
    compute_difference_hash,
    compute_perceptual_hash,
    compute_wavelet_hash,
    hamming_distance,
    normalized_hamming_distance,
    hamming_similarity
)

from .similarity_utils.py import (
    cosine_similarity,
    euclidean_distance,
    euclidean_similarity,
    pairwise_similarity,
    find_pairs_above_threshold,
    compute_similarity_statistics
)

from .validation import (
    validate_image_paths,
    validate_threshold,
    validate_detector_params,
    validate_similarity_metric
)

from .progress import ProgressTracker


__all__ [
    # Hash utilities
    'compute_crypto_hash',
    'compute_average_hash',
    'compute_difference_hash',
    'compute_perceptual_hash',
    'compute_wavelet_hash',
    'hamming_distance',
    'normalized_hamming_distance',
    'hamming_similarity',

    # Similarity utilities
    'cosine_similarity',
    'euclidean_distance',
    'euclidean_similarity',
    'pairwise_similarity',
    'find_pairs_above_threshold',
    'compute_similarity_statistics',

    # Validation
    'validate_image_paths',
    'validate_threshold',
    'validate_detector_params',
    'validate_similarity_metric',

    # Progress tracking
    'ProgressTracker'
]