"""EfficientNet feature extractor."""

import torch
import torchvision.models as models

from .base import FeatureExtractor

class EfficientNetExtractor(FeatureExtractor):
    """EfficientNet V2 feature extractor with variants."""

    VARIANTS = {
        's': (models.efficientnet_v2_s, models.EfficientNet_V2_S_Weights),
        'm': (models.efficientnet_v2_m, models.EfficientNet_V2_M_Weights),
        'l': (models.efficientnet_v2_l, models.EfficientNet_V2_L_Weights),
    }

    def __init__(self, variant = 's', weights = 'DEFAULT', custom_weights_path = None, device = 'cpu'):
        self.variant = variant.lower()
        self.weights_name = weights
        self.custom_weights_path = custom_weights_path
        super().__init__(device)
        self.load_model()
    
    def load_model(self):
        if self.variant not in self.VARIANTS:
            raise ValueError(f"Unknown EfficientNet variant: {self.variant}."
                            f"Available: {list(self.VARIANTS.keys())}")
        
        model_fn , weights_enum = self.VARIANTS[self.variant]

        if self.weights_name is None:
            weights = None
        else:
            weights = getattr(weights_enum, self.weights_name, weights_enum.DEFAULT)
        
        self.model = model_fn(weights=weights).to(self.device)
        self.model.eval()
        self.model.classifier = torch.nn.Identity()

        if weights:
            self.preprocess = weights.transforms()
        else:
            self.preprocess = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                    std = [0.229, 0.224, 0.225])
            ])
        
        if self.custom_weights_path:
            self.load_custom_weights(self.custom_weights_path)
    
    def _forward(self, image_tensor):
        return self.model(image_tensor)