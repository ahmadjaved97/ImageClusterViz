from abc import ABC, abstractmethod
import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA


class DimensionalityReducer(ABC):
    """Abstract base class for dimensionality reduction algorithms."""

    def __init__(self, n_components=50, **kwargs):
        self.n_components = n_components
        self.params = kwargs
        self.model = None
        self.is_fitted = False
        self.original_dim = None
        self.reduced_dim = None

    @abstractmethod
    def fit(self, features):
        """
        Fit the reduction model.
        """
        pass
    
    @abstractmethod
    def transform(self, features):
        """
        Transform features to lower dimensions.
        """
        pass
    
    def fit_transform(self, features):
        """
        Fit and transform in one step.
        """
        self.fit(features)
        return self.transform(features)
    
    def get_explained_variance(self):
        """
        Get explained variance ratio if available.
        """

        if hasattr(self.model, 'explained_variance_ratio_'):
            return self.model.explained_variance_ratio_
        return None
    
    
class PCAReducer(DimensionalityReducer):
    """
    Principal Component Analysis.
    """
    def __init__(self, n_components=50, whiten=False, use_incrimental=False, batch_size=None, random_state=42, **kwargs):
        super().__init__(n_components=n_components, **kwargs)
        self.whiten = whiten
        self.use_incrimental = use_incrimental
        self.batch_size = batch_size
        self.random_state = random_state
    
    def fit(self, features):
        self.original_dim = features.shape[1]
        n_samples = features.shape[0]


        actual_n_components = min(self.n_components, n_samples, self.original_dim)

        if self.use_incrimental or n_samples > 10000:
            batch_size = self.batch_size or min(10000, n_samples // 10)
            self.model = IncrementalPCA(
                n_components=actual_n_components,
                batch_size=batch_size
            )
        else:
            self.model = PCA(
                n_components=actual_n_components,
                whiten = self.whiten,
                random_state = self.random_state
            )
        
        self.model.fit(features)
        self.reduced_dim = self.model.n_components_
        self.is_fitted = True

        return self
    
    def transform(self, features):
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.transform(features)


REDUCTION_ALGORITHMS = {
    'pca': PCAReducer,
}

def create_reducer(algorithm, **kwargs):
    """
    Factory function to create dimensionality reduction algorithms.
    """

    algorithm = algorithm.lower()

    if algorithm not in REDUCTION_ALGORITHMS:
        available = ", ".join(sorted(REDUCTION_ALGORITHMS.keys()))
        raise ValueError(
            f"Unknown reduction algorithm: '{algorithm}'",
            f"Avaiable algorithms: {available}"
        )
    
    return REDUCTION_ALGORITHMS[algorithm](**kwargs)