"""Dino V2 feature extractor."""

import torch
from torchvision import transforms

from .base import FeatureExtractor


class DinoV2FeatureExtractor(FeatureExtractor):
    """DINOv2 feature extractor with variants."""

    VARIANTS = ['vits14', 'vitb14', 'vitl14', 'vitg14']

    def __init__(self, variant='vits14', custom_weights_path=None, device='cpu'):
        self.variant = variant.lower()
        self.custom_weights_path = custom_weights_path
        super().__init__(device)
        self.load_model()
    
    def load_model(self):
        if self.variant not in self.VARIANTS:
            raise ValueError(f"Unknown DINOv2 variant: {self.variant}",
                            f"Available: {self.VARIANTS}")
        
        model_name = f'dinov2_{self.variant}'
        self.model = torch.hub.load('facebookresearch/dinov2', model_name).to(self.device)

        self.model.eval()

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                std= [0.229, 0.224, 0.225])
        ])

        if self.custom_weights_path:
            self.load_custom_weights(self.custom_weights_path)
    
    def _forward(self, image_tensor):
        return self.model(image_tensor)