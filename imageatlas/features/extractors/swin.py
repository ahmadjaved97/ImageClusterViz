"""Swin feature extractor."""

import torch
import torchvision.models as models
from torchvision import transforms

from .base import FeatureExtractor

class SwinExtractor(FeatureExtractor):
    """Swin Transformer feature extractor with variants."""

    VARIANTS = {
        't':    (models.swin_t, models.Swin_T_Weights),
        's':    (models.swin_s, models.Swin_S_Weights),
        'b':    (models.swin_b, models.Swin_B_Weights),
        'v2_t': (models.swin_v2_t, models.Swin_V2_T_Weights),
        'v2_s': (models.swin_v2_s, models.Swin_V2_S_Weights),
        'v2_b': (models.swin_v2_b, models.Swin_V2_B_Weights),
    }

    def __init__(self, variant='v2_s', weights='DEFAULT', custom_weights_path=None, device='cpu'):
        self.variant = variant.lower()
        self.weights_name = weights
        self.custom_weights_path = custom_weights_path
        super().__init__(device)
        self.load_model()
    
    def load_model(self):
        if self.variant not in self.VARIANTS:
            raise ValueError(f"Unknown Swin Variant: {self.variant}"
                            f"Available : {list(self.VARIANTS.keys())}")
        
        model_fn, weights_enum = self.VARIANTS[self.variant]

        if self.weights_name is None:
            weights = None
        else:
            weights = getattr(weights_enum, self.weights_name, weights_enum.DEFAULT)
        
        self.model = model_fn(weights=weights).to(self.device)
        self.model.eval()
        self.model.head = torch.nn.Identity()

        if weights:
            self.preprocess = weights.transforms()
        else:
            self.preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
        
        if self.custom_weights_path:
            self.load_custom_weights(self.custom_weights_path)
    
    def _forward(self, image_tensor):
        return self.model(image_tensor)
        