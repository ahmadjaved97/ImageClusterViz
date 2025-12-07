"""Vision Transformer (ViT) feature extractor."""

import torch
import torchvision.models as models
from torchvision import transforms

# from .base
from .base import FeatureExtractor


class ViTExtractor(FeatureExtractor):
    """Vision Transformer feature extractor."""

    VARIANTS = {
        'b_16': (models.vit_b_16, models.ViT_B_16_Weights),
        'b_32': (models.vit_b_32, models.ViT_B_32_Weights),
        'l_16': (models.vit_l_16, models.ViT_L_16_Weights),
        'l_32': (models.vit_l_32, models.ViT_L_32_Weights),
        'h_14': (models.vit_h_14, models.ViT_H_14_Weights),
    }

    def __init__(self, variant='b_16', weights='DEFAULT',
                custom_weights_path=None, device='cpu'):
        
        self.variant = variant
        self.weights_name = weights
        self.custom_weights_path = custom_weights_path
        super().__init__(device)
        self.load_model()  # Model loaded during initialization.
    
    def load_model(self):
        if self.variant not in self.VARIANTS:
            raise ValueError(f"Unknown ViT variant: {self.variant}"
                            f"Available: {list(self.VARIANTS.keys())}")
        
        model_fn, weights_enum = self.VARIANTS[self.variant]

        if self.weights_name is None:
            weights = None
        elif self.weights_name == 'DEFAULT':
            weights = weights_enum.DEFAULT
        else:
            weights = getattr(weights_enum, self.weights_name, weights_enum.DEFAULT)
        
        self.model = model_fn(weights=weights).to(self.device)
        self.model.eval()

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
        feats = self.model._process_input(image_tensor)
        batch_class_token = self.model.class_token.expand(image_tensor.shape[0], -1, -1)
        feats = torch.cat([batch_class_token, feats], dim=1)
        feats = self.model.encoder(feats)

        return feats[:, 0]