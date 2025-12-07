"""ResNet feature extractor."""

import torch
import torchvision.models as models
from torchvision import transforms

from .base import FeatureExtractor

class ResNetExtractor(FeatureExtractor):
    """ResNet feature extractor with multiple variants."""

    VARIANTS = {
        '18': (models.resnet18, models.ResNet18_Weights),
        '34': (models.resnet34, models.ResNet34_Weights),
        '50': (models.resnet50, models.ResNet50_Weights),
        '101': (models.resnet101, models.ResNet101_Weights),
        '152': (models.resnet152, models.ResNet152_Weights),
    }

    def __init__(self, variant='50', weights='DEFAULT',
                custom_weights_path=None, device='cpu'):
        
        self.variant = variant
        self.weights_name = weights
        self.custom_weights_path = custom_weights_path
        super().__init__(device)
        self.load_model()
    

    def load_model(self):
        if self.variant not in self.VARIANTS:
            raise ValueError(f"Unknown ResNet Variant: {self.variant}"
                            f"Available: {list(self.VARIANTS.keys())}")
        
        model_fn, weights_enum = self.VARIANTS[self.variant]

        if self.weights_name is None:
            weights = None
        elif self.weights_name == 'DEFAULT':
            weights = weights_enum.DEFAULT
        else:
            weights = self.getattr(weights_enum, self.weights_name, weights_enum.DEFAULT)
        
        resnet = model_fn(weights=weights).to(self.device)
        resnet.eval()
        self.model = torch.nn.Sequential(*list(resnet.children())[:-1])

        if weights:
            self.preprocess = weights.transforms()
        else:
            self.preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
        
        if self.custom_weights_path:
            self.load_custom_weights(self.custom_weights_path)
    
    def _forward(self, image_tensor):
        features = self.model(image_tensor)
        return features.flatten(start_dim=1)