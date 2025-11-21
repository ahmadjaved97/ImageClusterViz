from vit import ViTExtractor
from PIL import Image


image = Image.open("/home/s63ajave/test_qwen.png")
vit_extractor = ViTExtractor()
print(vit_extractor)

features = vit_extractor.extract_features(image.convert('RGB'))
print(features[0])