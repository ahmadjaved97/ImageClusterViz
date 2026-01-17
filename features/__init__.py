"""
This module provides efficient feature extraction with batch processing, HDF5 caching and progress tracking.
"""

from .metadata import FeatureMetadata
from .loaders import ImageLoader
from .batch import BatchProcessor
from .cache import FeatureCache, HDF5Cache
from .pipeline import FeaturePipeline

__all__ = [
    'FeatureMetadata',
    'ImageLoader',
    'BatchProcessor',
    'FeatureCache',
    'HDF5Cache',
    'FeaturePipeline',
]

__version__ = "1.0.0"