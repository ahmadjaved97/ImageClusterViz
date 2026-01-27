"""HDBSCAN clustering Implementation."""


import numpy as np
from typing import Optional, List
from .base import ClusteringAlgorithm, ClusteringResult

class HDBSCANClustering(ClusteringAlgorithm):
    """
    HDBSCAN (Hierarchical Density-Based Spatial Clustering) Algorithm.

    Args:
        min_cluster_size: Minimum number of samples in a cluster
        min_samples: Number of samples in a neighborhood for core points.
        metric: Distance metric to use
        cluster_selection_method: Method for selecting clusters ('eom' or 'leaf')
        auto_params: Whether to automatically set parameters based on dataset size
        random_state: Random seed (note: HDBSCAN is deterministic, this is for consistency)

    """

    def __init__(
        self,
        min_cluster_size = None,
        min_samples = None,
        metric = 'euclidean',
        cluster_selection_method = 'eom',
        auto_params = True,
        random_state = 42,
        **kwargs,
    ):

        super().__init__(random_state=random_state, **kwargs)

        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        self.cluster_selection_method = cluster_selection_method
        self.auto_params = auto_params

        # Validate parameters
        if cluster_selection_method not in ['eom', 'leaf']:
            raise ValueError("cluster_selection_method must be 'eom' or 'leaf'")
        
    
    def _auto_select_params(self, n_samples):
        """
        Automatically select HDBSCAN parameters based on dataset size.

        Args:
            n_samples: Number of samples in the dataset.
        
        Returns:
            Tuple of (min_cluster_size, min_samples)
        """

        if n_samples < 100:
            min_cluster_size = max(5, n_samples // 20)
            min_samples = 1
        elif n_samples < 2000:
            min_cluster_size = 10
            min_samples = 1
        elif n_samples < 10000:
            min_cluster_size = 20
            min_samples = 3
        else:
            min_cluster_size = 30
            min_samples = 5
        
        return min_cluster_size, min_samples
    
    def fit_predict(
        self,
        features,
        filenames,
    ) -> ClusteringResult:

        """
        Fit HDBSCAN and predict cluster labels.

        Args:
            features: Feature matrix of shape (n_samples, n_features)
            filenames: Optional list of filenames for cluster mapping.
        """

        try:
            import hdbscan
        except ImportError:
            raise ImportError(
                "hdbscan is not installed. Install it with: pip install hdbscan"
                )
        
        self._validate_features(features)

        n_samples = features.shape[0]

        # Auto-select parameters if needed.
        if self.auto_params:
            auto_min_cluster_size, auto_min_samples = self._auto_select_params(n_samples)
            min_cluster_size = self.min_cluster_size or auto_min_cluster_size
            min_samples = self.min_samples or auto_min_samples
        else:
            min_cluster_size = self.min_cluster_size or 10
            min_samples = self.min_samples or 1
        
        # Ensure parameters are valid.
        min_cluster_size = min(min_cluster_size, n_samples)
        min_samples = min(min_samples, min_cluster_size)

        # Create and fit HDBSCAN
        self._model = hdbscan.HDBSCAN(
            min_cluster_size = min_cluster_size,
            min_samples = min_samples,
            metric = self.metric,
            cluster_selection_method = self.cluster_selection_method,
            core_dist_n_jobs=-1 # Use all cores
        )

        # Fit and predict
        cluster_labels = self._model.fit_predict(features)

        # Create cluster dictionary
        cluster_dict = self._create_cluster_dict(cluster_labels, filenames)

        # Count actual clusters
        unique_labels = np.unique(cluster_labels)
        n_clusters = len(unique_labels[unique_labels >= 0])
        n_noise = np.sum(cluster_labels == -1)

        # Prepare metadata
        metadata = {
            'algorithm': 'hdbscan',
            'n_clusters_found': n_clusters,
            'n_noise_points': int(n_noise),
            'min_cluster_size': min_cluster_size,
            'min_samples' : min_samples,
            'metric': self.metric,
            'cluster_selection_method': self.cluster_selection_method,
            'probabilities': self._model.probabilities_,
            'outlier_scores': self._model.outlier_scores_,
        }

        # Add exemplars if available (representative points for each cluster)
        if hasattr(self._model, 'exemplars_'):
            metadata['exemplars'] = self._model.exemplars_
        
        self.is_fitted = True

        return ClusteringResult(
            cluster_labels=cluster_labels,
            cluster_dict=cluster_dict,
            n_clusters=n_clusters,
            metadata=metadata
        )
    
    def get_outlier_score(self):
        """
        Get outlier score for each sample.

        Higher scores indicate more likely outliers.

        Returns:
            Array of outlier scores or None if model is not fitted.
        """

        if self.is_fitted and self._model is not None:
            return self._model.outlier_scores_
        
        return None
    
    def get_condensed_tree(self):
        """
        Get condensed cluster hierarchy tree.

        Returns:
            Array of membership probabilities or None if model not fitted.
        """

        if self.is_fitted and self._model is not None:
            return self._model.condensed_tree_
        
        return None
    
    def get_algorithm_name(self):
        """
        Return the name of the clustering algorithm.
        """
        return "HDBSCAN"
    
    def get_params(self):
        """
        Get parameters of the clustering algorithm.
        """
        return {
            'min_cluster_size': self.min_cluster_size,
            'min_samples': self.min_samples,
            'metric': self.metric,
            'cluster_selection_method': self.cluster_selection_method,
            'auto_params': self.auto_params,
            'random_state': self.random_state
        }

