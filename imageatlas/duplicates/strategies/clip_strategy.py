"""
CLIP-based duplicate detection strategy.
"""

import numpy as np
from typing import List, Literal

from ..base import DuplicateDetectionStrategy
from ..adapters.feature_extractor_adapter import FeatureExtractorAdapter
from ..utils import cosine_similarity, ProgressTracker


class CLIPStrategy(DuplicateDetectionStrategy):
    """
    CLIP-based duplicate detection.
    Uses CLIP embeddings for multi-modal semantic similarity.
    """
    
    def __init__(
        self,
        variant: str = 'ViT-B/16',
        device: str = 'cpu',
        **kwargs
    ):
        """
        Initialize CLIP strategy.
        
        Args:
            variant: CLIP model variant
            device: Device to use ('cpu', 'cuda')
        """
        super().__init__(
            variant=variant,
            device=device,
            **kwargs
        )
        
        self.variant = variant
        self.device = device
        
        # Create adapter for CLIP
        self.adapter = FeatureExtractorAdapter(
            model_type='clip',
            variant=variant,
            device=device
        )
    
    def compute_signatures(
        self,
        image_paths: List[str],
        batch_size: int = 32,
        verbose: bool = True,
        **kwargs
    ) -> np.ndarray:
        """
        Compute CLIP embeddings for images.
        
        Returns:
            CLIP embedding matrix, shape (n_images, embedding_dim)
        """
        n_batches = (len(image_paths) + batch_size - 1) // batch_size
        
        all_embeddings = []
        
        with ProgressTracker(
            n_batches,
            desc=f"Extracting CLIP embeddings",
            disable=not verbose
        ) as progress:
            
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i + batch_size]
                
                # Extract CLIP features
                batch_embeddings = self.adapter.extract_features_batch(
                    batch_paths,
                    batch_size=len(batch_paths)
                )
                
                all_embeddings.append(batch_embeddings)
                progress.update(1, processed=len(all_embeddings) * batch_size)
        
        # Stack all embeddings
        embeddings = np.vstack(all_embeddings)
        
        # Ensure correct size
        if len(embeddings) > len(image_paths):
            embeddings = embeddings[:len(image_paths)]
        
        return embeddings
    
    def compute_similarity(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """
        Compute similarity between two CLIP embeddings.
        
        Uses cosine similarity (CLIP embeddings are normalized).
        
        Returns:
            Cosine similarity score
        """
        return cosine_similarity(sig1, sig2)
    
    def get_method_name(self) -> str:
        return f"clip_{self.variant.replace('/', '_')}"