"""CLIP Feature Extractor."""
import clip
from .base import FeatureExtractor

class CLIPExtractor(FeatureExtractor):
    """CLIP Feature Extractor with variants."""
    VARIANTS = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64',
                'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
    
    def __init__(self, variant='ViT-B/16', device='cpu'):
        self.variant = variant
        super().__init__(device)
        self.load_model()
    
    def load_model(self):
        if self.variant not in self.VARIANTS:
            raise ValueError(f"Unknown CLIP Variant: {self.variant}"
                            f"Available: {self.VARIANTS}")
        
        self.model, self.preprocess = clip.load(self.variant, device=self.device)
        self.model.eval()
    
    def _forward(self, image_tensor):
        features = self.model.encode_image(image_tensor)
        features = features / features.norm(dim=-1, keepdim=True)
        return features
