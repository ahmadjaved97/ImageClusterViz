"""
Abstract base class for dimensionality reduction algorithms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class ReductionResult:
    """
    Container for dimensionality reduction result.
    """

    reduced_features: np.ndarray
    original_dim: int
    reduced_dim: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self):
        """
        Get a summary string of reduction results.
        """
        lines = [
            "Dimensionality reduction summary:",
            f"   Original dimension: {self.original_dim}",
            f"   Reduced dimension: {self.reduced_dim}",
            f"   Reduction ratio: {self.original_dim / self.reduced_dim:.2f}",
            f"   Output shape: {self.reduced_features.shape}",
        ]

        if 'variance_explained' in self.metadata:
            var_exp = self.metadata['variance_explained']
            if isinstance(var_exp, (list, np.ndarray)):
                total_var = sum(var_exp) if len(var_exp) > 0 else 0
                lines.append(f"   Variance explained: {total_var:.2%}")
            else:
                lines.append(f"   Variance explained: {var_exp:.2%}")
        
        if 'reconstruction_error' in self.metadata:
            lines.append(f"   Reconstruction error: {self.metadata['reconstruction_error']:.4f}")
        
        return "\n".join(lines)

class DimensionalityReducer(ABC):
    """
    Abstract base class for all dimensionality reduction algorithms.
    """

    def __init__(self, n_components=50, random_state=42, **kwargs):
        """
        Initialize the dimensionality reduction algorithm.
        """

        self.n_components = n_components
        self.random_state = random_state
        self.params = kwargs
        self.is_fitted = False
        self.model = None
        self.original_dim = None
        self.reduced_dim = None

    
    @abstractmethod
    def fit(self, features):
        """
        Fit the reduction model to the data.
        """
        pass
    
    @abstractmethod
    def transform(self, features):
        """
        Transform the features to lower dimension.
        """
        pass
    
    def fit_transform(self, features):
        """
        Fit and transform in one step.
        """
        self.fit(features)
        return self.transform(features)
    
    @abstractmethod
    def get_algorithm_name(self):
        """
        Return the name of the reduction algorithm.
        """
        pass
    
    def _validate_features(self, features):
        """
        Validate the input feature matrix.
        """
        if not isinstance(features, np.ndarray):
            raise ValueError(f"Feature must be a numpy array, got {type(features)}")
        
        if features.ndim != 2:
            raise ValueError(f"Features must be 2D array, got shape: {features.shape}")
        
        if features.shape[0] == 0:
            raise ValueError("Feature matrix is empty.")
        
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            raise ValueError("Features contain NaN or Inf values")
    
    def get_params(self):
        """
        Get parameters of the reduction algorithm.
        """
        return {
            'n_components': self.n_components,
            'random_state': self.random_state,
            **self.params
        }
    
    def get_metadata(self):
        """
        Get algorithm specific metadata after fitting.
        """
        return {}
    
    def __repr__(self):
        """
        String representation of the reducer.
        """
        params_str = ", ".join(f"{k}={v}" for k, v in self.get_params().items())
        return f"{self.get_algorithm_name()}({params_str})"
