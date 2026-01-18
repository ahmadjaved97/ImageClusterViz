"""
UMAP dimensionality reduction
"""

import numpy as np
from typing import Optional
from .base import DimensionalityReducer

class UMAPReducer(DimensionalityReducer):
    """
    UMAP for non-linear dimensionality reduction.
    """

    def __init__(
        self,
        n_components=50,
        n_neighbors=15,
        min_dist=0.1,
        metric='euclidean',
        random_state=42,
        n_jobs=-1,
        **kwargs
    ):
        super().__init__(n_components=n_components, random_state=random_state, **kwargs)
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.n_jobs = n_jobs
    
    def fit(self, features):
        """
        Fit UMAP to the features.
        """

        try:
            import umap
        except ImportError:
            raise ImportError(
                "umap-learn is not installed. Install it with: pip install umap-learn"
            )
        
        self._validate_features(features)

        self.original_dim = features.shape[1]
        n_samples = features.shape[0]

        # Validate n_neighbors
        if self.n_neighbors >= n_samples:
            self.n_neighbors = max(2, n_samples - 1)
            print(f"Warning: n_neighbors adjusted to {self.n_neighbors} (must be < n_samples)")
        
        # Create and fit UMAP
        self.model = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=False
        )

        self.model.fit(features)
        self.reduced_dim = self.n_components
        self.is_fitted = True

        return self
    
    def transform(self, features):
        """
        Transform features to reduced dimensions.
        """

        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        self._validate_features(features)
        return self.model.transform(features)
    
    def get_metadata(self):
        """
        Get UMAP specific metadata.
        """

        if not self.is_fitted:
            return {}
        
        return {
            'algorithm': 'umap',
            'n_components': int(self.reduced_dim),
            'n_neighbors': int(self.n_neighbors),
            'min_dist': float(self.min_dist),
            'metric': self.metric,
        }
    
    def get_algorithm_name(self):
        """Return the name of the algorithm."""
        return "UMAP"
    
    def get_params(self):
        """Get parameters of the algorithm."""
        
        return {
            'n_components': self.n_components,
            'n_neighbors': self.n_neighbors,
            'min_dist': self.min_dist,
            'metric': self.metric,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs
        }
