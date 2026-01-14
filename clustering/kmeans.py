"""
K-Means clustering implementation
"""

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from .base import ClusteringAlgorithm, ClusteringResult

class KMeansClustering(ClusteringAlgorithm):
    """
    K-Means clustering algorithm.
    """

    def __init__(
        self,
        n_clusters: int = 5,
        n_init: int = 15,
        max_iter: int = 300,
        use_minibatch: bool = False,
        batch_size: int = 1024,
        random_state: Optional[int] = 42,
        **kwargs
    ):

        super().__init__(random_state=random_state, **kwargs)

        if n_clusters < 2:
            raise ValueError("n_clusters must be atleast 2.")
        
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.use_minibatch = use_minibatch
        self.batch_size = batch_size
    
    def fit_predict(
        self,
        features,
        filenames = None,
    ) -> ClusteringResult:

        """
        Fit K-Means and predict cluster labels.
        """

        self._validate_features(features)

        n_samples = features[0]

        # Adjust n-clusters if needed.
        actual_n_clusters = min(self.n_clusters, n_samples)

        if actual_n_clusters < self.n_clusters:
            print(f"Warning: Requested {self.n_clusters}clusters but only ",
                    f"{n_samples} samples. Using {actual_n_clusters} clusters.")
        
        # Decide whether to use MiniBatchKMeans

        use_minibatch = self.use_minibatch or n_samples > 10000

        if use_minibatch:
            self._model = MiniBatchKMeans(
                n_clusters=actual_n_clusters,
                init='k-means++',
                max_iter=self.max_iter,
                batch_size=min(self.batch_size, n_samples),
                random_state=self.random_state,
                n_init=self.n_init,
                verbose=0
            )
        
        else:
            self._model = KMeans(
                n_clusters=actual_n_clusters,
                init='k-means++',
                n_init=self.n_init,
                max_iter=self.max_iter,
                random_state=self.random_state,
                verbose=0
            )
        
        # Fit and predict
        cluster_labels = self._model.fit_predict(features)

        # Create cluster dictionary
        cluster_dict = self._create_cluster_dict(cluster_labels, filenames)

        # Prepare metadata
        metadata = {
            'algorithm': 'kmeans',
            'inertia': float(self._model.inertia_),
            'n_iter': int(self._model.n_iter_),
            'cluster_centers': self._model.cluster_centers_,
            'used_minibatch': use_minibatch,
        }

        self.is_fitted = True

        return ClusteringResult(
            cluster_labels=cluster_labels,
            cluster_dict=cluster_dict,
            n_clusters=actual_n_clusters,
            metadata=metadata
        )
    

    def predict(self, features):
        """
        Predict cluster label for new samples.
        """

        if not self.is_fitted or self._model == None:
            raise RuntimeError("Model must be fitted before prediction. Call fit_predict first.")
        
        self._validate_features(features)
        return self._model.predict(features)
    
    def get_cluster_centers(self):
        """
        Get cluster centers if model is fitted.
        """
        if self.is_fitted and self._model is not None:
            return self._model.cluster_centers_
        
        return None
    
    def get_algorithm_name(self):
        """
        Return the name of the clustering algorithm.
        """
        return "KMeans"
    
    def get_params(self):
        """
        Get parameters of the clustering algorithm.
        """

        return {
            'n_clusters': self.n_clusters,
            'n_init': self.n_init,
            'max_iter': self.max_iter,
            'use_minibatch': self.use_minibatch,
            'batch_size': self.batch_size,
            'random_state': self.random_state,
        }


