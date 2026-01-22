from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class ClusteringResult:
    """
    Container for clustering Results.
    """
    cluster_labels: np.ndarray
    cluster_dict: Dict[int, List[int]]
    n_clusters: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_cluster_sizes(self) -> Dict[int, int]:
        """Get the size of each cluster."""
        return {cluster_id: len(indices) for cluster_id, indices in self.cluster_dict.items()}
    
    def get_cluster(self, cluster_id: int) -> List[int]:
        """Get indices belonging to a specific cluster."""
        return self.cluster_dict.get(cluster_id, [])
    
    def get_outliers(self) -> Optional[List[int]]:
        """Get outlier indices (cluster_id = -1) if any exist."""
        return self.cluster_dict.get(-1, None)
    
    def summary(self) -> str:
        """Get a summary string of clustering results."""

        summary_lines = [
            f"Clustering Results Summary:",
            f"   Total Samples: {len(self.clustering_labels)}",
            f"   Number of Clusters: {self.n_clusters}",
            f"   Cluster Sizes: {self.get_cluster_sizes()}",
        ]

        outliers = self.get_outliers()

        if outliers:
            summary_lines.append(f"   Outliers: {len(outliers)}")
        
        if self.metadata:
            summary_lines.append(f"   Metadata: {list(self.metadata.keys())}")
        
        return "\n".join(summary_lines)

class ClusteringAlgorithm(ABC):
    """
    Abstract base class for all clustering algorithms.
    """

    def __init__(self, random_state=42, **kwargs):
        """
        Initialize the clustering algorithm.
        """
        self.random_state = random_state
        self.params = kwargs
        self.is_fitted = False
        self._model = None

    @abstractmethod
    def fit_predict(self, features) -> ClusteringResult:
        """
        Fit the clustering algorithms and predict cluster labels.
        """
        pass
    
    @abstractmethod
    def get_algorithm_name(self):
        """
        Return the name of the clustering algorithm.
        """
        pass
    
    def _validate_features(self, features:np.ndarray) -> None:
        """
        Validate the input feature matrix.
        """
        if not isinstance(features, np.ndarray):
            raise ValueError(f"Feature must be a numpy array, got {type(features)}")
        
        if features.ndim != 2:
            raise ValueError(f"Features must be 2D array, got shape: {features.shape}")
        
        if features.shape[0] == 0:
            raise ValueError("Feature matrix is empty.")
        
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            raise ValueError("Features contain NaN or Inf values")
    
    def _create_cluster_dict(self, cluster_labels, filenames=None):
        """
        Createa dictionary mapping cluster IDs to indices or filenames
        """

        cluster_dict = {}

        for idx, cluster_id in enumerate(cluster_labels):
            cluster_id = int(cluster_id)

            if cluster_id not in cluster_dict:
                cluster_dict[cluster_id] = []
            
            item = filenames[idx] if filenames else idx
            cluster_dict[cluster_id].append(item)
        
        return cluster_dict
    
    def get_params(self):
        """
        Get parameters of the clustering algorithms.
        """
        return {
            'random_state': self.random_state,
            **self.params,
        }
    
    def __repr__(self):
        """
        String representation of the clustering algorithm.
        """
        params_str = ", ".join(f"{key}={value}" for key, value in self.get_params().items())
        return f"{self.get_algorithm_name()}({params_str})"
    



