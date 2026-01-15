"""
This module provides efficient feature extraction with batch processing, HDF5 caching and progress tracking.
"""

from .metadata import FeatureMetadata
from .loaders import ImageLoader

__all__ = [
    'FeatureMetadata',
    'ImageLoader',
]

__version__ = "1.0.0"