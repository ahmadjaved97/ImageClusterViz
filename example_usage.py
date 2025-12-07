from PIL import Image
from feature_extractors import create_feature_extractor


image = Image.open("/home/s63ajave/test_qwen.png")

#Example 1: simple usage
# vit_extractor = create_feature_extractor(model_type='vit', variant='b_32', device='cuda')
# features = vit_extractor.extract_features(image.convert('RGB'))
# print(features[0])
# print(features[0].shape)

# # Example 2: Direct class instantiation
# from feature_extractors import ViTExtractor
# extractor = ViTExtractor(variant='b_16', device='cpu')
# features = extractor.extract_features(image.convert("RGB"))
# print(features[0])
# print(features[0].shape)

# # Dino V2
# dinov2_extractor = create_feature_extractor(model_type='dinov2', device='cuda')
# features = dinov2_extractor.extract_features(image.convert('RGB'))
# print(features[0])
# print(features[0].shape)

# # Efficient Net
# efficientnet_extractor = create_feature_extractor(model_type='efficientnet', variant='s', device='cpu')
# features = efficientnet_extractor.extract_features(image.convert('RGB'))
# print(features[0])
# print(features[0].shape)

#Mobilenet V3
# mobilenet_extractor = create_feature_extractor(model_type='mobilenet', variant='large', device='cuda')
# features = mobilenet_extractor.extract_features(image.convert('RGB'))
# print(features[0])
# print(features.shape)

# CLIP
# clip_extractor = create_feature_extractor(model_type='clip', variant='RN50', device='cuda')
# features = clip_extractor.extract_features(image.convert('RGB'))
# print(features[0])
# print(features.shape)


# Swin 
# swin_extractor = create_feature_extractor(model_type='swin', variant='v2_b', device='cuda')
# features = swin_extractor.extract_features(image.convert('RGB'))
# print(features[0])
# print(features[0].shape)

#VGG
# vgg_extractor = create_feature_extractor(model_type='vgg', device='cuda')
# features = vgg_extractor.extract_features(image.convert('RGB'))
# print(features[0].min(), features[0].max())
# print(features[0].shape)

#ConvNext
convnext_extractor = create_feature_extractor(model_type='convnext', device='cuda')
features = convnext_extractor.extract_features(image.convert('RGB'))
print(features[0])
print(features[0].shape)