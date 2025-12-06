"""MobileNet V3 Feature extractor."""

import torch
import torchvision.models as models

from .base import FeatureExtractor

class MobileNetV3FeatureExtractor(FeatureExtractor):
    """MobileNet V3 feature extractor."""

    VARIANTS = {
        'small': (models.mobilenet_v3_small, mobilenet_V3_Small_Weights),
        'large': (models.mobilenet_v3_large, mobilenet_V3_Large_Weights)
    }

    def __init__(self, variant='large', weights='DEFAULT',
                custom_weights_path=None, device='cpu'):
        
        self.variant = variant
        self.weights_name = weights
        self.custom_weights_path = custom_weights_path
        super().__init__(device)
        self.load_model()
    
    def load_model(self):
        if self.variant not in self.VARIANTS:
            raise ValueError(f"Unknown MobileNetV3 variant: {self.variant}"
                            f"Available: {list(self.VARIANTS.keys())}")
        
        model_fn , weights_enum = self.VARIANTS[self.variant]

        if self.weights_name is None:
            weights = None
        else:
            weights = getattr(weights_enum, self.weights_name, weights_enum.DEFAULT)
        
        