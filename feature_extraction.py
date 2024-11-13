import torch
import torchvision.models as models
from torchvision import transforms

def load_vit_model(weights=models.ViT_B_16_Weights.DEFAULT):
    """
    Loads a pre-trained Vision Transformer (ViT) model using the specified weights, and returns both the model and 
    its associated preprocessing transforms.

    Parameters:
    - weights (torchvision.models.ViT_B_16_Weights, optional): The pre-trained weights to load for the ViT model. 
      The default is `ViT_B_16_Weights.DEFAULT`, which uses the standard pre-trained weights for the ViT-B/16 model. 
      Other weights can be specified based on available options in `torchvision.models`.

    Returns:
    - model (torchvision.models.ViT): The Vision Transformer model initialized with the specified pre-trained weights.
    - preprocess (torchvision.transforms): A set of preprocessing transforms that are associated with the model's 
      pre-trained weights. These transforms ensure the input data is properly normalized and resized for the ViT model.

    Notes:
    - The function relies on the `torchvision.models` module to load the Vision Transformer model architecture 
      and weights.
    - The associated preprocessing transforms include resizing, normalization, and other image processing steps needed 
      to prepare inputs for the model.
    """

    vit = models.vit_b_16(weights=weights)
    vit.eval()
    preprocess = weights.transforms()
    return vit, preprocess

def load_resnet50_model():
    """
    Loads a pre-trained ResNet-50 model with the final fully connected layer removed, and returns the model 
    along with its preprocessing transforms. This setup is useful for feature extraction tasks where the model's 
    output will be used as high-level image features.

    Parameters:
    - None

    Returns:
    - model (torchvision.models.ResNet): The ResNet-50 model pre-trained on ImageNet, with the final fully connected 
      layer excluded.
    - preprocess (torchvision.transforms): A set of preprocessing transforms necessary to prepare input images 
      for the ResNet-50 model, including resizing, normalization, and tensor conversion.

    Notes:
    - The ResNet-50 model is loaded from the `torchvision.models` module, and is pre-trained on ImageNet.
    - The final fully connected layer (used for classification) is excluded to make the model suitable for feature extraction.
    """

    resnet = models.resnet50(pretrained=True)
    resnet.eval()
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return resnet, preprocess

def load_vgg16_model():
    vgg16 = models.vgg16(pretrained=True)
    vgg16.eval()
    vgg16 = torch.nn.Sequential(*list(vgg16.children())[:-1])
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return vgg16, preprocess
    

def load_mobilenetv3_model(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2):
    mobilenet = models.mobilenet_v3_large(weights=weights)
    mobilenet.eval()
    mobilenet = mobilenet.features
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return mobilenet, preprocess

def extract_features(image, model, preprocess, model_type='vit'):
    """
    Extracts features from a given image using a specified pre-trained model (Vision Transformer or ResNet).
    The function first applies the necessary preprocessing steps to the image, forwards the processed image 
    through the model, and returns the extracted feature vector.

    Parameters:
    - image (PIL.Image or numpy.ndarray): The input image from which features will be extracted. 
      It should be in a format compatible with the preprocessing pipeline.
    - model (torch.nn.Module): The pre-trained model used for feature extraction, such as a ViT or ResNet model.
    - preprocess (torchvision.transforms): The preprocessing pipeline associated with the pre-trained model, which 
      ensures the image is appropriately resized, normalized, and converted to a tensor.
    - model_type (str, optional): A string indicating the type of model being used for feature extraction. 
      Default is 'vit' for Vision Transformer, but 'resnet' can be specified for a ResNet model. This parameter 
      may affect how the model is used for inference.

    Returns:
    - features (torch.Tensor): A tensor representing the extracted feature vector from the image. 
      This feature vector can be used for tasks such as image clustering, similarity analysis, or as input to another model.

    Notes:
    - The input image is first preprocessed using the provided `preprocess` function to ensure it is in the correct format 
      for the model (e.g., size and normalization).
    - The feature extraction process depends on the `model_type`. The Vision Transformer (ViT) and ResNet have different 
      architectures and return feature vectors of varying dimensions.
    - Ensure that the pre-trained model and preprocessing transforms are compatible (i.e., both ViT or both ResNet).
    - The returned feature vector can be used for various downstream tasks, including transfer learning, image retrieval, 
      or clustering.
    """
     
    image = preprocess(image)
    if model_type == 'resnet':
        image = image.unsqueeze(0)
        features = model(image)
        features = torch.flatten(features, start_dim=1)
        features = features[0].cpu().detach().numpy()

    elif model_type == 'vgg16':
        image = image.unsqueeze(0)
        with torch.no_grad():
            features = model(image)
        features = torch.flatten(features, start_dim=1)
        features = features[0].cpu().detach().numpy()
    
    elif model_type == 'mobilenetv3':
        image = image.unsqueeze(0)
        with torch.no_grad():
            features = model(image)
        features = torch.flatten(features, start_dim=1)
        features = features[0].cpu().detach().numpy()
        
    elif model_type == 'vit':  # ViT
        image = image.unsqueeze(0)
        feats = model._process_input(image)
        batch_class_token = model.class_token.expand(image.shape[0], -1, -1)
        feats = torch.cat([batch_class_token, feats], dim=1)
        feats = model.encoder(feats)
        feats = feats[:, 0]
        features = feats.cpu().detach().numpy()[0]
    return features