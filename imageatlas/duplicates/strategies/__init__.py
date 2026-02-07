"""
Detection strategies for duplicate detection.
"""

from .hash_strategy import CryptographicHashStrategy, PerceptualHashStrategy
from .embedding_strategy import EmbeddingStrategy
from .clip_strategy import CLIPStrategy


__all__ = [
    'CryptographicHashStrategy',
    'PerceptualHashStrategy',
    'EmbeddingStrategy',
    'CLIPStrategy'
]