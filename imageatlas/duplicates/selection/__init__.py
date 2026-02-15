"""
Best image selection strategies.
"""

from .best_selector import (
    ResolutionSelector,
    AlphabeticSelector,
    FileSizeSelector,
    create_best_selector
)


__all__ = [
    'ResolutionSelector',
    'AlphabeticSelector',
    'FileSizeSelector',
    'create_best_selector'
]