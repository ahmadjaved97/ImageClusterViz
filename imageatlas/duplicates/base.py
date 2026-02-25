"""
Core abstractions for Duplicate Detection.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional
import numpy as np


class DuplicateDetectionStrategy(ABC):
    """
    Abstract base class for duplicate detection strategies.

    Each strategy computes signatures (hashes/embeddings) and similarity score between images.
    """

    def __init__(self, **kwargs):
        """Initialize strategy with configuration."""
        self.params = kwargs
    
    @abstractmethod
    def compute_signatures(
        self,
        image_paths,
        **kwargs
    ):
        """
        Compute signatures (hashes/embeddings) for images.
        """

        pass
    
    @abstractmethod
    def compute_similarity(
        self,
        sig1,
        sig2
    ):
        """
        Compute similarity between two signatures.

        Similarity score -> higher = more similar
        """

        pass
    
    @abstractmethod
    def get_method_name(self):
        """Return human readable method name."""
        pass
    
    def get_params(self):
        """
        Get strategy parameters.
        """
        return self.parameters.copy()
    

class GroupingAlgorithm(ABC):
    """
    Abstract base class for grouping duplicate pairs into clusters.
    """

    @abstractmethod
    def group_duplicates(
        self,
        pairs,
        filenames
    ):
        """
        Grouping duplicate pairs into clusters.
        """

        pass
    

class BestImageSelector(ABC):
    """
    Abstract base class for selecting the best image from a group.
    """

    @abstractmethod
    def select_best(
        self,
        image_paths,
        metadata
    ):
        """
        Select the best image from a list.
        """

        pass
    
class ThresholdSelector(ABC):
    """
    Abstract base class for threshold selection.
    """

    @abstractmethod
    def select_threshold(
        self,
        similarities,
        user_threshold
    ):
        """
        Determine the similarity threshold.
        """
        pass
