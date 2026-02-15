"""
Threshold selection strategies.
"""

import numpy as np
from typing import Optional

from ..base import ThresholdSelector

class FixedThreshold(ThresholdSelector):
    """
    Uses a fixed user-provided threshold.
    """

    def select_threshold(
        self,
        similarities,
        user_threshold
    ):
        """
        Return the user provided threshold.
        """

        if user_threshold is None:
            raise ValueError("Fixed threshold requires user_threshold to be provided.")
        
        return float(user_threshold)

class AdaptivePercentileThreshold(ThresholdSelector):
    """
    Automatically selects threshold based on similarity distribution.

    Uses precentile-based approach: keep top X% most similar pairs.
    """

    def __init__(self, percentile = 95.0):
        """
        Initialize adaptive threshold selector.
        """

        if not 0 < percentile < 100:
            raise ValueError("Percentile must be between 0 and 100")
        
        self.percentile = percentile
    
    def select_threshold(
        self,
        similarities,
        user_threshold
    ):
        """
        Compute threshold from similarity distribution.
        """

        if len(similarities) == 0:
            raise ValueError("Cannot compute adaptive threshold with no similarities")
        
        # Compute percentile
        threshold = np.percentile(similarities, self.percentile)

        return float(threshold)
    

# AdaptiveGapThreshold later (REVISIT)