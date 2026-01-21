"""MobileNet V3 Feature extractor."""

import torch
import torchvision.models as models

from .base import FeatureExtractor

class MobileNetV3FeatureExtractor(FeatureExtractor):
    """MobileNet V3 feature extractor."""

    VARIANTS = {
        'small': (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights),
        'large': (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights)
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
        
        mobilenet = model_fn(weights=weights).to(self.device)
        mobilenet.eval()
        self.model = mobilenet.features

        if weights:
            self.preprocess = weights.transforms()
        else:
            self.preprocess = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
        
        if self.custom_weights_path:
            self.load_custom_weights(self.custom_weights_path)
    

    def _forward(self, image_tensor):
        features = self.model(image_tensor)
        return features.flatten(start_dim=1)
