"""
Core module for ImageClusterViz.

This module provides the main ImageClusterer API that ties together
feature extraction, dimensionality reduction, and clustering.
"""

from .clusterer import ImageClusterer
from .results import ClusteringResults, ExportManager

__all__ = [
    'ImageClusterer',
    'ClusteringResults',
    'ExportManager',
]

__version__ = '1.0.0'