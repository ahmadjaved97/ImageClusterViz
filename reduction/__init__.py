"""
Dimensionality reduction module.
"""


from .base import DimensionalityReducer, ReductionResult
from .pca import PCAReducer
from .umap_reducer import UMAPReducer
from .tsne import TSNEReducer
from .factory import create_reducer, get_available_reducers

_all__ = [
    'DimensionalityReducer',
    'ReductionResult',
    'PCAReducer',
    'UMAPReducer',
    'TSNEReducer',
    'create_reducer',
    'get_available_reducers'
]

__version__ = '1.0.0'