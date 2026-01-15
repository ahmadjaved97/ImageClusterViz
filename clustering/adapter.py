"""
Adapter module for backward compatibility with existing code.
"""

import numpy as np
from .factory import create_clustering_algorithm


def get_clustered_data(
    feature_dict,
    num_clusters = 5,
    clustering_method = 'kmeans',
    feature_reduction=None
    ):
    """
    Adapter function for backward compatibility with cluster_images.py.
    """
    filenames = list(feature_dict.keys())
    feature_vectors = np.array(list(feature_dict.values()))

    # Apply dimensionality reduction
    if feature_reduction is not None:
        feature_vectors = feature_reduction.fit_transform(feature_vectors)
    
    # Create clustering algorithm
    if clustering_method.lower() == 'kmeans':
        clusterer = create_clustering_algorithm('kmeans', n_clusters=num_clusters)
    elif clustering_method.lower() == 'gmm':
        clusterer = create_clustering_algorithm('gmm', n_components=num_clusters)
    elif clustering_method.lower() == 'hdbscan':
        clusterer = create_clustering_algorithm('hdbscan', auto_params=True)
    else:
        raise ValueError(f"Unknown clustering method: {clustering_method}")
    

    # Perform clustering
    result = clusterer.fit_predict(feature_vectors, filenames=filenames)

    # Return cluster dictionary
    return result.cluster_dict