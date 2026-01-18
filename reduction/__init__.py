"""
Dimensionality reduction module.
"""


from .base import DimensionalityReducer, ReductionResult
from .pca import PCAReducer
from .umap_reducer import UMAPReducer

_all__ = [
    'DimensionalityReducer',
    'ReductionResult',
    'PCAReducer',
    'UMAPReducer',
]

__version__ = '1.0.0'