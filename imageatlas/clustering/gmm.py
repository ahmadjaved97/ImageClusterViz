"""
Gaussian Mixture Model (GMM) clustering implementation.
"""

import numpy as np
from typing import Optional, List
from sklearn.mixture import GaussianMixture
from .base import ClusteringAlgorithm, ClusteringResult

class GMMClustering(ClusteringAlgorithm):
    """
    Gaussian Mixture Model clustering algorithm.

    Args:
        n_components: Number of mixture components (clusters)
        covariance_type: Type of covarince parameters ('full', 'diag', 'tied', 'spherical')
        max_iter: Maximum number of EM iterations
        n_init: Number of initializations to perform
        reg_covar: Regularization added to diagonal of covariance (prevents singular matrices)
        random_state: Random seed for reproducibility
    """

    def __init__(
        self,
        n_components = 5,
        covariance_type = 'diag',
        max_iter = 100,
        n_init = 10,
        reg_covar = 1e-6,
        random_state = 42,
        **kwargs,
    ):

        super().__init__(random_state, **kwargs)

        if n_components < 2:
            raise ValueError("n_components must be atleast 2")
        
        valid_covariance_types = ['full', 'tied', 'diag', 'spherical']
        if covariance_type not in valid_covariance_types:
            raise ValueError(f"covariance_type must be one of {valid_covariance_types}")
        
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.n_init = n_init
        self.reg_covar = reg_covar
    
    def fit_predict(
        self,
        features,
        filenames = None,
    ) -> ClusteringResult:

        """
        Fit GMM and predict cluster labels.

        Args:
            features: Feature matrix of shape (n_samples, n_features)
            filenames: Optional list of filenames for cluster mapping
        
        Returns:
            ClusteringResult object with cluster assignments.
        """

        self._validate_features(features)

        n_samples = features.shape[0]

        # Adjust n-components if needed
        actual_n_components = min(self.n_components, n_samples)

        if actual_n_components < self.n_components:
            print(f"Warning: Requested {self.n_components} components only "
                f"{n_samples} samples. Using {actual_n_components} components.")
        
        features_float64 = features.astype(np.float64)

        
        # Create and fit GMM
        self._model = GaussianMixture(
            n_components=actual_n_components,
            covariance_type=self.covariance_type,
            max_iter=self.max_iter,
            n_init=self.n_init,
            reg_covar=self.reg_covar,
            random_state=self.random_state,
            verbose=0
        )

        # Fit and predict
        cluster_labels = self._model.fit_predict(features_float64)

        # Get probability scores
        probabilities = self._model.predict_proba(features_float64)

        # Create cluster dictionary
        cluster_dict =self._create_cluster_dict(cluster_labels, filenames)


        # Prepare metadata
        metadata = {
            'algorithm': 'gmm',
            'converged': bool(self._model.converged_),
            'n_iter': int(self._model.n_iter_),
            'bic': float(self._model.bic(features_float64)),
            'aic': float(self._model.aic(features_float64)),
            'lower_bound': float(self._model.lower_bound_),
            'covariance_type': self.covariance_type,
            'probabilities': probabilities,
            'means': self._model.means_,
            'covariance': self._model.covariances_,
        }

        self.is_fitted = True

        return ClusteringResult(
            cluster_labels = cluster_labels,
            cluster_dict = cluster_dict,
            n_clusters = actual_n_components,
            metadata = metadata
        )
    
    def predict(self, features):
        """
        Predict cluster label for new samples.

        Args:
            features: Feature matrix of shape (n_samples, n_features)
        
        Returns:
            Array of cluster labels
        
        Raises:
            RuntimeError: If model has not been fitted yet.
        """

        if not self.is_fitted or self._model is None:
            raise RuntimeError("Model must be fitted before prediction. Call fit_predict first.")
        
        self._validate_features(features)
        return self._model.predict(features)
    
    def predict_proba(self, features):
        """
        Predict probability of each cluster for new samples.

        Args:
            features: Feature matrix of shape (n_samples, n_features)
        
        Returns:
            Array of cluster labels
        
        Raises:
            RuntimeError: If model has not been fitted yet.
        """

        if not self.is_fitted or self._model is None:
            raise RuntimeError("Model must be fitted before prediction. Call fit_predict first.")
        
        self._validate_features(features)
        return self._model.predict_proba(features)
    
    def get_cluster_means(self):
        """
        Get cluster means (centers) if model is fitted.

        Returns:
            Array of cluster centers or None if not fitted.
        """

        if self.is_fitted and self._model is not None:
            return self._model.means_
        
        return None
    
    def score(self, features):
        """
        Compute the log-likelihood of the data under the model.

        Args:
            features: Feature matrix of shape (n_samples, n_features)
        
        Returns:
            Log-likelihood score
        """

        if not self.is_fitted or self._model is None:
            raise RuntimeError("Model must be fitted before scoring.")
        
        return self._model.score(features)
    
    def get_algorithm_name(self):
        """Return the name of the clustering algorithm."""
        return "GaussianMixtureModel (GMM)"
    
    def get_params(self):
        """Get parameters of the clustering algorithm."""
        return {
            'n_components': self.n_components,
            'covariance_type': self.covariance_type,
            'max_iter': self.max_iter,
            'n_init': self.n_init,
            'random_state': self.random_state,
            'reg_covar': self.reg_covar
        }
