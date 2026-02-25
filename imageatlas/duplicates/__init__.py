
"""
Duplicate detection module for ImageAtlas.
"""


from .detector import DuplicateDetector
from .results import DuplicateResults
from .visualization import create_duplicate_grids, DuplicateGridVisualizer
from .cache import DuplicateCache

# Expose main classes
__all__ = [
    'DuplicateDetector',
    'DuplicateResults',
    'create_duplicate_grids',
    'DuplicateGridVisualizer',
    'DuplicateCache'
]