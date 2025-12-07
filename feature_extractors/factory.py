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
    elif model_type == 'efficientnet':
        from .efficientnet import EfficientNetExtractor
        extractors['efficientnet'] = EfficientNetExtractor
    elif model_type == 'dinov2':
        from .dinov2 import DinoV2FeatureExtractor
        extractors['dinov2'] = DinoV2FeatureExtractor
    elif model_type == 'mobilenet':
        from .mobilenet import MobileNetV3FeatureExtractor
        extractors['mobilenet'] = MobileNetV3FeatureExtractor
    elif model_type == 'clip':
        from .clip import CLIPExtractor
        extractors['clip'] = CLIPExtractor
    elif model_type == 'swin':
        from .swin import SwinExtractor
        extractors['swin'] = SwinExtractor
    elif model_type == 'vgg':
        from .vgg import VGG16Extractor
        extractors['vgg'] = VGG16Extractor
    else:
        raise ValueError(f"Unknown model type: {model_type}"
                        f" Supported types: vit, resnet, efficientnet, dinov2, .....")
    
    if variant is not None:
        kwargs['variant'] = variant
    
    return extractors[model_type](device=device, **kwargs)
