"""
Metadata management for feature extraction.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
from datetime import datetime
import json


@dataclass
class FeatureMetadata:
    """
    Metadata for extracted features.

    Tracks information about the feature extractionn process including
    model details, extraction parameters, and statistics.
    """

    # Model information
    model_type: str
    model_variant: Optional[str] = None
    feature_dim: int = 0

    # Extraction information
    n_samples: int = 0
    extraction_date: str = field(default_factory=lambda: datetime.now().isoformat())
    device: str = 'cpu'
    batch_size: int = 1

    # Statistics
    total_time: float = 0.0
    images_per_second: float = 0.0
    n_corrupted: int = 0
    n_skipped: int = 0

    # Additional information
    description: str = ""
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        """Convert metadata to dictionary."""
        return asdict(self)
    
    def to_json(self, path):
        """Save metadata to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, data):
        """Create metadata from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_json(cls, path):
        "Load metadata from JSON file."
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def summary(self):
        """Get a human readable summary."""
        lines = [
            "Feature Extraction Summary:",
            f"Model: {self.model_type}" + (f"({self.model_variant})" if self.model_variant else ""),
            f"Feature dimension: {self.feature_dim}",
            f"Samples processed: {self.n_samples}",
            f"Extraction date: {self.extraction_date}",
            f"Device: {self.device}",
            f"Batch size: {self.batch_size}",
            f"Total time: {self.total_time}",
            f"Speed: {self.images_per_second:.2f} images/sec",
        ]

        if self.n_corrupted > 0:
            lines.append(f"   Corrupted images: {self.n_corrupted}")
        if self.n_skipped > 0:
            lines.append(f"   Skipped images: {self.n_skipped}")
        
        if self.description:
            lines.append(f"   Description: {self.description}")
        
        return "\n".join(lines)
