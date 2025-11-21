"""Factory function for creating feature extractors."""

from .base import FeatureExtractor

def create_feature_extractor(model_type, variant=None, device='cpu', **kwargs):
    """
    Factory function to create feature extractors.
    """

    # Lazy imports
    extractors = {}

    if model_type == 'vit':
        from .vit import ViTExtractor
        extractors['vit'] = ViTExtractor
    elif model_type == 'resnet':
        from .resnet import ResNetExtractor
        extractors['resnet'] = ResNetExtractor
    else:
        raise ValueError(f"Unknown model type: {model_type}"
                        f"Supported types: vit, resnet, .....")
    
    if variant is not None:
        kwargs['variant'] = variant
    
    return extractors[model_type](device=device, **kwargs)
