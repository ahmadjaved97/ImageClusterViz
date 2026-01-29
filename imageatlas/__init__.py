"""
ImageAtlas: A toolkit for organizing, cleaning and analysing your image datasets.
"""

__version__ = '0.1.1'


# 1. High level API (The everything tool)
from .core.clusterer import ImageClusterer
from .core.results import ClusteringResults

# 2. Modular APIs (For specific tasks)

# Feature extraction tools
from .features.pipeline import FeaturePipeline
from .features.extractors.factory import create_feature_extractor

# ImageLoader, BatchProcessing later

# Dimensionality Reduction Tools
from .reduction.factory import create_reducer
from .reduction.pca import PCAReducer
from .reduction.umap_reducer import UMAPReducer

# Clustering Tools
from .clustering.factory import create_clustering_algorithm

# Visualization Tools
from .visualization.grids import GridVisualizer, create_cluster_grids

__all__ = [
    'ImageClusterer',
    'ClusteringResults',
    'FeaturePipeline',
    'create_feature_extractor',
    'create_reducer',
    'PCAReducer',
    'UMAPReducer',
    'create_clustering_algorithm',
    'GridVisualizer',
    'create_cluster_grids',
]