from PIL import Image
from feature_extractors import create_feature_extractor


image = Image.open("/home/s63ajave/test_qwen.png")

#Example 1: simple usage
vit_extractor = create_feature_extractor(model_type='vit', variant='b_32', device='cuda')
features = vit_extractor.extract_features(image.convert('RGB'))
print(features[0])
print(features[0].shape)

# Example 2: Direct class instantiation
from feature_extractors import ViTExtractor
extractor = ViTExtractor(variant='b_16', device='cpu')
features = extractor.extract_features(image.convert("RGB"))
print(features[0])
print(features[0].shape)
