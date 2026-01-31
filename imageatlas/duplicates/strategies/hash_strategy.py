"""
Hash based duplicate detection strategy.
"""

import numpy as np
from PIL import Image
import warnings


from ..base import DuplicateDetectionStrategy
from ..utils import (
    compute_crypto_hash,
    compute_average_hash,
    compute_difference_hash,
    compute_perceptual_hash,
    compute_wavelet_hash,
    hamming_similarity,
    ProgressTracker
)

class CryptographicHashStrategy(DuplicateDetectionStrategy):
    """
    Cryptographic hash based duplicate detection.
    """

    def __init__(self, algorithm='md5', **kwargs):
        """
        Initialize cryptographic hash strategy.
        """
        super().__init__(algorithm=algorithm)
        self.algorithm = algorithm
    
    def compute_signatures(
        self,
        image_paths,
        verbose=True,
        **kwargs
    ):
        """
        Compute cryptographic hashes for images.
        """

        hashes = []

        with ProgressTracker(
            len(image_paths),
            desc=f"Computing {self.algorithm.upper()} hashes",
            disable=not verbose
        ) as progress:

            for path in image_paths:
                try:
                    hash_value = compute_crypto_hash(path, self.algorithm)
                    hashes.append(hash_value)
                except for Exception as e:
                    warnings.warn(f"Failed to hash {path}: {e}")
                    hashes.append(None)
                
                progress.update(1)
        
        return np.array(hashes, dtype=object)
    

    def compute_similarity(self, sig1, sig2):
        """
        Compute two cryptographic hashes.

        Returns:
            1.0 if identical, 0.0 if different.
        """

        if sig1 is None or sig2 is None:
            return 0.0
        
        return 1.0 if sig1 == sig2 else 0.0
    
    def get_method_name(self):
        return f"crypto_hash_{self.algorithm}"
    

class PerceptualHashStrategy(DuplicateDetectionStrategy):
    """
    Perceptual hash based duplicate detection.
    Find near identical images.
    """

    def __init__(
        self,
        algorithm='phash',
        hash_size=8
    ):
        """
        Initialize perceptual hash strategy.
        """

        super().__init__(algorithm=algorithm, hash_size=hash_size, **kwargs)
        self.algorithm = algorithm
        self.hash_size = hash_size

        # Select hash function
        if algorithm == 'ahash':
            self.hash_func = compute_average_hash
        elif algorithm == 'dhash':
            self.hash_func = compute_difference_hash
        elif algorithm == 'phash':
            self.hash_func = compute_perceptual_hash
        elif algorithm == 'whash':
            self.hash_func = compute_wavelet_hash
        else:
            raise ValueError(f"Unknown hash algorithm: {algorithm}")
    
    def compute_signatures(
        self,
        image_paths,
        verbose=True,
        **kwargs
    ):
        """
        Compute perceptual hashes for images.
        """

        hashes = []

        with ProgressTracker(
            len(image_paths),
            desc=f"Computing {self.algorithm} hashes",
            disable=not verbose
        ) as progress:

            for path in image_paths:
                try:
                    image = Image.open(path)

                    # Compute hash based on algorithm
                    if self.algorithm == 'whash':
                        hash_bits = self.hash_func(image, self.hash_size, mode='haar')
                    else:
                        hash_bits = self.hash_func(image, self.hash_size)
                    
                    hashes.append(hash_bits)
                
                except Exception as e:
                    warnings.warn(f"Failed to hash {path}: {e}")
                
                progress.update(1)
        
        return np.array(hashes)
    
    def compute_similarity(self, sig1, sig2):
        """

        Compute similarity between two perceptual hashes.
        Uses hamming similarity.
        """

        return hamming_similarity(sig1, sig2)
    
    def get_method_name(self):
        return self.algorithm

