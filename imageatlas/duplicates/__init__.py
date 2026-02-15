
"""
Duplicate detection module for ImageAtlas.
"""


from .detector import DuplicateDetector
from .results import DuplicateResults

# Expose main classes
__all__ = [
    'DuplicateDetector',
    'DuplicateResults',
]