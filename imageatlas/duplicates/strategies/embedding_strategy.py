"""
Deep learning embedding based duplicate detection methods.
"""

import numpy as np
from typing import List, Literal

from ..base import DuplicateDetectionStrategy
from ..adapters.feature_extractor_adapter import FeatureExtractorAdapter
from ..utils import cosine_similarity, euclidean_similarity, ProgressTracker

class EmbeddingStrategy(DuplicateDetectionStrategy):
    """
    Embedding based duplicate detection using deep learning models.
    """

    def __init__(
        self,
        model: str = 'dinov2',
        variant: str ='vits14',
        similarity_metric: Literal['cosine','euclidean'] = 'cosine',
        device: str = 'cpu'
    ):
        """
        Initialize embedding strategy.
        """

        super().__init__(
            model=model,
            self.variant=variant,
            similarity_metric=similarity_metric,
            device=device,
            **kwargs
        )

        self.model = model
        self.variant = variant
        self.similarity_metric = similarity_metric
        self.device = device

        # Create adapter
        self.adapter = FeatureExtractorAdapter(
            model_type=model,
            variant=variant,
            device=device
        )

    
    def compute_signatures(
        self,
        image_paths,
        batch_size=32,
        verbose=True,
        **kwargs
    ):
        """
        Compute embeddings for images.
        """

        n_batches = (len(image_paths) + batch_size - 1) // batch_size

        all_embeddings = []

        with ProgressTracker(
            n_batches,
            desc=f"Extracting {self.model} embeddings",
            disable = not verbose
        ) as progress:

            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i: i + batch_size]

                # Extract features for the batch
                batch_embeddings = self.adapter.extract_features_batch(
                    batch_paths,
                    batch_size=len(batch_paths)
                )

                all_embeddings.append(batch_embeddings)
                progress.update(1, processed=len(all_embeddings) * batch_size)
        
        # Stack all embeddings
        embeddings = np.vstack(all_embeddings)

        # Ensure we get the right number (REVISIT)
        if len(embeddings) > len(image_paths):
            embeddings = embeddings[:len(image_paths)]
        
        return embeddings
    
    def compute_similarity(self, sig1, sig2):
        """
        Compute similarity between two embeddings.
        """

        if self.similarity_metric == 'cosine':
            return cosine_similarity(sig1, sig2)
        elif self.similarity_metric == 'euclidean':
            return euclidean_similarity(sig1, sig2)
        else:
            raise ValueError(f"Unknown metric: {self.similarity_metric}")
    
    def get_method_name(self):
        variant_str = f"_{self.variant}" if self.variant else ""
        return f"_embedding_{self.model}{variant_str}"
