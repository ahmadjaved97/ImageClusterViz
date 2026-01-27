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

    Args:
        method: Name of the clustering algorithm ('kmeans', 'gmm', 'hdbscan')
        **kwargs: Algorithm specific parameters
    
    Returns:
        Instance of the requested clustering algorithm
    
    Raises:
        Value Error: If clustering method is not supported.
    

    Examples:
        >>> # Create KMeans with 5 clusters
        >>> clusterer = create_clustering_algorithm('kmeans', n_clusters=5)

        >>>  # Create GMM with full covariance
        >>> clusterer = create_clustering_algorithm('gmm', n_components=8, covariance_type='full')

        >>> # Create HDBSCAN with auto parameters
        >>> clusterer = create_clustering_algorithm('hdbscan', auto_params=True)
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

    Returns:
        List of algorithm names.
    """
    return sorted(CLUSTERING_ALGORITHMS.keys())
