"""
Factory function for creating clustering algorithms.
"""


from .base import ClusteringAlgorithm
from .kmeans import KMeansClustering
from .hdbscan_clustering import HDBSCANClustering
from .gmm import GMMClustering


# Registry of available clustering algorithms.
CLUSTERING_ALGORITHMS = {
    'kmeans': KMeansClustering,
    'hdbscan': HDBSCANClustering,
    'gmm': GMMClustering,
}

def create_clustering_algorithm(
    method:str,
    **kwargs,
) -> ClusteringAlgorithm:
    """
    Factory function to create clustering algorithms.
    """

    method = method.lower()

    if method not in CLUSTERING_ALGORITHMS:
        available = ", ".join(sorted(CLUSTERING_ALGORITHMS.keys()))
        raise ValueError(
            f"Unknown clustering Method: '{method}'.",
            f"Available methods: {available}"
        )
    
    algorithm_class = CLUSTERING_ALGORITHMS[method]
    return algorithm_class(**kwargs)

def get_available_algorithms():
    """
    Get a list of available clustering algorithms.
    """
    return sorted(CLUSTERING_ALGORITHMS.keys())
