"""VGG 16 Feature extractor."""

import torch
import torchvision.models as models
from torchvision import transforms

from .base import FeatureExtractor


class VGG16Extractor(FeatureExtractor):
    """VGG16 Feature Extractor."""

    def __init__(self, weights='DEFAULT', custom_weights_path=None, device='cpu'):
        self.weights_name = weights
        self.custom_weights_path = custom_weights_path
        super().__init__(device)
        self.load_model()
    

    def load_model(self):
        if self.weights_name is None:
            weights = None
        else:
            weights = getattr(models.VGG16_Weights, self.weights_name, models.VGG16_Weights.DEFAULT)
        
        vgg16 = models.vgg16(weights=weights).to(self.device)
        vgg16.eval()
        self.model = torch.nn.Sequential(*list(vgg16.children())[:-1])

        if weights:
            self.preprocess = weights.transforms()
        else:
            self.preprocess = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                    std = [0.229, 0.224, 0.225])
            ])
        
        if self.custom_weights_path:
            self.load_custom_weights(self.custom_weights_path)
    
    def _forward(self, image_tensor):
        features = self.model(image_tensor)
        return features.flatten(start_dim=1)