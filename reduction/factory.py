"""
Factory function for creating dimensionality reduction algorithms.
"""

from typing import List, Type, Dict
from .base import DimensionalityReducer
from .pca import PCAReducer
from .umap import UMAPReducer

# Registry of available reduction algorithms.
REDUCTION_ALGORITHMS = {
    'pca': PCAReducer,
    'umap': UMAPReducer,
}

def create_reducer(algorithm, **kwargs):
    """
    Factory function to create dimensionality reduction algorithms.
    """

    algorithm = algorithm.lower()

    if algorithm not in REDUCTION_ALGORITHMS:
        available = ", ".join(sorted(set(REDUCTION_ALGORITHMS.keys())))
        raise ValueError(
            f"Unknown reduction algorithm: {algorithm}. ",
            f"Available algorithms: {available}"
        )
    
    algorithm_class = REDUCTION_ALGORITHMS[algorithm]
    return algorithm_class(**kwargs)


def get_available_reducers():
    """
    Get a list of available reduction algorithms.
    """
    unique = set(REDUCTION_ALGORITHMS.values())
    names = []
    for names, cls in REDUCTION_ALGORITHMS.items():
        if cls in unique:
            names.append(name)
            unique.remove(cls)
    
    return sorted(names)

