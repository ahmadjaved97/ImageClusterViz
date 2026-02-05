"""
Main API for duplicate detection.
"""


import numpy as np
from pathlib import Path

from .base import DuplicateDetectionStrategy, GroupingAlgorithm, BestImageSelector
from .strategies import (
    CryptographicHashStrategy,
    PerceptualHashStrategy,
    EmbeddingStrategy,
    CLIPStrategy
)

from .grouping import (
    GroupBuilder,
    PairwiseGrouping,
    FixedThreshold,
    AdaptivePercentileThreshold
)

from .selection import create_best_selector
from .results import DuplicateResults
from .cache import DuplicateCache
from .utils import (
    validate_image_paths,
    validate_detector_params,
    pairwise_similarity,
    find_pairs_above_threshold,
    compute_similarity_statistics,
    ProgressTracker
)