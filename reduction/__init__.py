"""
Dimensionality reduction module.
"""


from .base import DimensionalityReducer, ReductionResult
from .pca import PCAReducer

_all__ = [
    'DimensionalityReducer',
    'ReductionResult',
    'PCAReducer',
]

__version__ = '1.0.0'