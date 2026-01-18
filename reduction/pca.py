"""
Principal Component Analysis
"""

import numpy as np
from typing import Optional
from sklearn.decomposition import PCA, IncrementalPCA
from .base import DimensionalityReducer

class PCAReducer(DimensionalityReducer):
    """
    Principal Component Analysis for dimensionality reduction.
    """

    def __init__(
        self,
        n_components=50,
        whiten=False,
        use_incremental=False,
        batch_size=None,
        random_state=42,
        **kwargs
    ):
        super().__init__(n_components=n_components, random_state=random_state, **kwargs)
        self.whiten = whiten
        self.use_incremental = use_incremental
        self.batch_size = batch_size
    
    def fit(self, features):
        """
        Fit PCA to the features.
        """

        self._validate_features(features)

        self.original_dim = features.shape[1]
        n_samples = features.shape[0]

        # Adjust n_components if needed
        actual_n_components = min(self.n_components, n_samples, self.original_dim)

        if actual_n_components < self.n_components:
            print(f"Warning: Requested {self.n_components} components but limited by "
                f"data dimensions. Using {actual_n_components} components.")
        
        # Choose PCA Variant
        if self.use_incremental or n_samples > 10000:
            # User Incremental PCA for larger datasets.
            batch_size = self.batch_size or min(10000, n_samples // 10)
            self.model = IncrementalPCA(
                n_components=actual_n_components,
                batch_size=batch_size
            )
        
        else:
            # Use standard PCA
            self.model = PCA(
                n_components=actual_n_components,
                whiten=self.whiten,
                random_state=self.random_state
            )
        
        # Fit the model
        self.model.fit(features)
        self.reduced_dim = self.model.n_components_
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
    
    def inverse_transform(self, reduced_features):
        """
        Transform reduced features back to original space.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        return self.model.inverse_transform(reduced_features)
    
    def get_explained_variance(self):
        """
        Get the variance explained by each component.
        """

        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        return self.model.explained_variance_ratio_
    
    def get_cumulative_variance(self):
        """
        Get cumulative explained variance.
        """
        return np.cumsum(self.get_explained_variance())
    
    def get_components(self):
        """
        Get principal components.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        return self.model.components_
    
    def get_metadata(self):
        """
        Get PCA specific metadata
        """
        if not self.is_fitted:
            return {}
        
        variance_explained = self.get_explained_variance()
        total_variance = np.sum(variance_explained)

        return {
            'algorithm': 'pca',
            'variance_explained': variance_explained.tolist(),
            'total_variance_explained': float(total_variance),
            'cumulative_variance': self.get_cumulative_variance().tolist(),
            'n_components': int(self.reduced_dim),
            'used_incremental': self.use_incremental,
        }
    
    def get_algorithm_name(self):
        """Return the name of the algorithm."""
        return "PCA"
    
    def get_params(self):
        """
        Get parameters of the algorithm.
        """
        return {
            'n_components': self.n_components,
            'whiten': self.whiten,
            'use_incremental': self.use_incremental,
            'batch_size': self.batch_size,
            'random_state': self.random_state
        }