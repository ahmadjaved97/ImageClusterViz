"""
t-SNE dimensionality reduction.
"""

import numpy as np
from typing import Optional
from sklearn.manifold import TSNE
from .base import DimensionalityReducer


class TSNEReducer(DimensionalityReducer):
    """
    t-SNE for non-linear dimensionality reduction.
    """

    def __init__(
        self,
        n_components=2,
        perpelxity=30.0,
        learning_rate=200.0,
        n_iter=1000,
        metric='euclidean',
        random_state=42,
        n_jobs=-1,
        **kwargs
    ):
        super().__init__(n_components=n_components, random_state=random_state, **kwargs)
        self.perpelxity=perpelxity
        self.learning_rate=learning_rate
        self.n_iter = n_iter
        self.metric = metric
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._embedded_features = None

    def fit(self, features):
        """
        Fit t-SNE to the features.
        """

        self._validate_features(features)

        self.original_dim = features.shape[1]
        n_samples = features.shape[0]

        # Adjust perplexity if needed
        max_perplexity = (n_samples - 1) / 3
        if self.perpelxity > max_perplexity:
            adjusted_perplexity = max(5, max_perplexity)
            print(f"Warning: perplexity adjusted to {adjusted_perplexity:.1f} "
                f"(must be < n_samples / 3)")
            perplexity = adjusted_perplexity
        else:
            perpelxity = self.perpelxity
        

        # Create t-SNE model
        self.model = TSNE(
            n_components=self.n_components,
            perpelxity=self.perpelxity,
            learning_rate=self.learning_rate,
            n_iter=self.n_iter,
            metric=self.metric,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=0
        )

        self.reduced_dim = self.n_components
        self.is_fitted = True
    
    def transform(self, features):
        """
        Transform is not supported for t-SNE.
        """

        raise NotImplementedError(
            "t-SNE does not support transform(). "
            "Use fit_transform() on the complete dataset."
        )
    
    def fit_transform(self, features):
        """
        Fit the t-SNE and return the embedded features.
        """

        # Adjust perplexity if needed
        max_perplexity = (n_samples - 1) / 3
        if self.perpelxity > max_perplexity:
            adjusted_perplexity = max(5, max_perplexity)
            print(f"Warning: perplexity adjusted to {adjusted_perplexity:.1f} "
                f"(must be < n_samples / 3)")
            perplexity = adjusted_perplexity
        else:
            perpelxity = self.perpelxity
        

        # Create t-SNE model
        self.model = TSNE(
            n_components=self.n_components,
            perpelxity=self.perpelxity,
            learning_rate=self.learning_rate,
            n_iter=self.n_iter,
            metric=self.metric,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=0
        )

        # Fit and transform in one step
        self._embedded_features = self.model.fit_transform(features)
        self.reduced_dim = self.n_components
        self.is_fitted = True

        return self._embedded_features
    
    def get_embedding(self):
        """
        Get the computed embedding.
        """

        return self._embedded_features
    
    def get_metadata(self):
        """
        Get t-SNE specific metadata.
        """

        if not self.is_fitted:
            return {}
        
        metadata = {
            'algorithm': 'tsne',
            'n_components': int(self.reduced_dim),
            'perplexity': float(self.perpelxity),
            'learning_rate': float(self.learning_rate),
            'n_iter': int(self.n_iter),
            'metric': self.metric
        }

        if hasattr(self.model, 'kl_divergence_'):
            metadata['kl_divergence'] = float(self.model.kl_divergence_)
        
        return metadata
    
    def get_algorithm_name(self):
        """
        Return the name of the algorithm.
        """
        return "t-SNE"
    
    def get_params(self):
        """
        Get parameters of the algorithm
        """

        return {
            'n_components': self.n_components,
            'perplexity': self.perpelxity,
            'learning_rate': self.learning_rate,
            'n_iter': self.n_iter,
            'metric': self.metric,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs
        }
    


