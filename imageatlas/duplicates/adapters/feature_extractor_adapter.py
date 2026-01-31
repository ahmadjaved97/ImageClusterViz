"""
Adapter to integrate existing feature extractors with duplicate detection.
"""

import numpy as np
from typing import List
from PIL import Image

from ..features.extractors import create_feature_extractor

class FeatureExtractorAdapter:
    """
    Adapting existing feature extractors for duplicate detection.
    """

    def __init__(
        self,
        model_type,
        variant=None,
        device='cpu'
    ):
        """
        Initialize adapter.
        """

        self.model_type = model_type
        self.variant = variant
        self.device = device

        self.extractor = create_feature_extractor(
            model_type=model_type,
            variant=variant,
            device=device
        )
    
    def extract_features_batch(
        self,
        image_paths,
        batch_size=32
    ):
        """
        Extract features from a batch of images.
        """

        all_features = []

        # Process in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]

            # Load images
            images = []
            for path in batch_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    images.append(img)
                except Exception as e:
                    print(f"Image not found {path}: {e}")
            
            # Extract features
            if len(images) > 1:
                batch_features = self.extractor.extract_batch(images)
            else:
                batch_features = self.extractor.extract_features(images[0])
                batch_features = batch_features.reshape(1, -1)
            
            all_features.append(batch_features)
        
        return np.vstack(all_features)
    
    def get_feature_dim(self):
        """
        Get feature dimensionality
        """

        dummy_img = Image.new('RGB', (224, 224))
        features = self.extractor.extract_features(dummy_img)
        return features.shape[0]
    