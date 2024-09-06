import os
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
from rich import print
import cv2
from rich.progress import track
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import math
import shutil
import pickle
import argparse

# add support for different models apart from resnet50 and ViT
# explore the use of CLIP

def create_image_grid(image_list, num_rows=None, num_cols=None, image_size=(100, 100), space_color=(255, 255, 255), space_width=10):
    """
    Arranges a list of images into a grid format.Automatically calculates the number of rows and columns if not provided,
    resizes images to a specified size, and places them into the grid with optional spacing and numbering.
    """
    if num_rows is None and num_cols is None:
        num_images = len(image_list)
        num_cols = int(math.sqrt(num_images))
        num_rows = math.ceil(num_images / num_cols)
    elif num_rows is None:
        num_rows = math.ceil(len(image_list) / num_cols)
    elif num_cols is None:
        num_cols = math.ceil(len(image_list) / num_rows)

    grid_width = num_cols * (image_size[0] + space_width) - space_width
    grid_height = num_rows * (image_size[1] + space_width) - space_width
    grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    grid_image[:, :] = space_color

    for i, image in enumerate(image_list):
        image = cv2.resize(image, image_size)
        row = i // num_cols
        col = i % num_cols
        x = col * (image_size[0] + space_width)
        y = row * (image_size[1] + space_width)
        grid_image[y: y + image_size[1], x: x + image_size[0]] = image
    
    for row in range(num_rows):
        y = row * (image_size[1] + space_width) + 5
        cv2.putText(grid_image, str(row + 1), (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    for col in range(num_cols):
        x = col * (image_size[0] + space_width) + 5
        cv2.putText(grid_image, str(col + 1), (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return grid_image

def hconcat_resize(img_list, interpolation=cv2.INTER_CUBIC):
    """
    Horizontally concatenates a list of images after resizing them to the height of the smallest image in the list, using the specified interpolation method.
    """
    h_min = min(img.shape[0] for img in img_list)
    im_list_resize = [cv2.resize(img, (int(img.shape[1] * h_min / img.shape[0]), h_min), interpolation=interpolation) for img in img_list]
    return cv2.hconcat(im_list_resize)

def load_vit_model(weights=models.ViT_B_16_Weights.DEFAULT):
    """
    Loads a pre-trained Vision Transformer (ViT) model along with its associated preprocessing transforms.
    """
    vit = models.vit_b_16(weights=weights)
    vit.eval()
    preprocess = weights.transforms()
    return vit, preprocess

def load_resnet50_model():
    """
    Loads a pre-trained ResNet-50 model, excluding the final fully connected layer, and sets up the necessary preprocessing steps.
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

def extract_features(image, model, preprocess, model_type='vit'):
    """
    Extracts features from an image using the specified model (ViT or ResNet). 
    The function preprocesses the image, forwards it through the model, and returns the extracted feature vector.
    """
    image = preprocess(image)
    if model_type == 'resnet':
        image = image.unsqueeze(0)
        features = model(image)
        features = torch.flatten(features, start_dim=1)
        features = features[0].cpu().detach().numpy()
    else:  # ViT
        image = image.unsqueeze(0)
        feats = model._process_input(image)
        batch_class_token = model.class_token.expand(image.shape[0], -1, -1)
        feats = torch.cat([batch_class_token, feats], dim=1)
        feats = model.encoder(feats)
        feats = feats[:, 0]
        features = feats.cpu().detach().numpy()[0]
    return features

def get_clustered_data(feature_dict, num_clusters=5, clustering_method='kmeans'):
    """
    Clusters the feature vectors from images into a specified number of clusters using either KMeans or Gaussian Mixture Models (GMM). 
    It returns a dictionary mapping each cluster ID to a list of corresponding image filenames.
    """
    filenames = list(feature_dict.keys())
    feature_vectors = list(feature_dict.values())
    feature_vectors = np.array(feature_vectors)

    if clustering_method == 'gmm':
        gmm = GaussianMixture(n_components=num_clusters, random_state=42)
        cluster_labels = gmm.fit_predict(feature_vectors)
    else:  # Default to KMeans
        kmeans = KMeans(n_clusters=num_clusters, n_init=15, random_state=42)
        cluster_labels = kmeans.fit_predict(feature_vectors)

    cluster_assignments = dict(zip(filenames, cluster_labels))
    cluster_dict = {}
    for file, cluster_id in cluster_assignments.items():
        if cluster_id not in cluster_dict:
            cluster_dict[cluster_id] = []
        cluster_dict[cluster_id].append(file)
    return cluster_dict

def create_cluster_folders(cluster_data, source_folder_path, output_folder):
    """
    Creates folders for each cluster and copies the images belonging to each cluster into their respective folders.
    """
    for key, value in track(cluster_data.items(), description="📂 Transferring images to cluster folders", complete_style="red"):
        folder_path = os.path.join(output_folder, 'cluster_' + str(key))
        os.makedirs(folder_path, exist_ok=True)
        for file in value:
            source_path = os.path.join(source_folder_path, file)
            destination_path = os.path.join(folder_path, file)
            shutil.copy(source_path, destination_path)

def create_cluster_grids(cluster_data, source_path, output_folder):
    """
    Creates a grid image for each cluster by arranging the images belonging to that cluster and saves the grid images.
    """
    for key, value in track(cluster_data.items(), description="📂 Writing cluster grid images", complete_style="white"):
        cluster_name = 'cluster_' + str(key) + '.jpg'
        hf = [cv2.imread(os.path.join(source_path, file)) for file in value]
        grid = create_image_grid(hf, image_size=(300, 300))
        output_path = os.path.join(output_folder, cluster_name)
        cv2.imwrite(output_path, grid)

def save_dict(feature_dict, path):
    """
    Saves a dictionary of image features to a file using pickle.
    """
    output_path = os.path.join(path, "feature_dictionary.pkl")
    with open(output_path, "wb") as file:
        pickle.dump(feature_dict, file)
    print(f"[yellow]Feature dictionary saved at: [bold red]{output_path}")

def read_dict(path):
    """
    Loads a dictionary of image features from a file using pickle.
    """
    input_path = os.path.join(path, "feature_dictionary.pkl")
    with open(input_path, "rb") as file:
        feature_dict = pickle.load(file)
    return feature_dict

def create_feature_dict(dataset_path, model, preprocess, model_type):
    """
    Creates a dictionary mapping image filenames to their corresponding feature vectors, extracted using the specified model.
    """
    feature_dict = {}
    for file in track(os.listdir(dataset_path), total=len(os.listdir(dataset_path)), description="Getting image features", complete_style="yellow"):
        if file.endswith(".jpg"):
            image_path = os.path.join(dataset_path, file)
            try:
                image = Image.open(image_path)
            except RuntimeError as e:
                image = Image.open(image_path).convert('RGB')

            image_feature = extract_features(image, model, preprocess, model_type)
            feature_dict[file] = image_feature
    return feature_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cluster images using ViT or ResNet and create grids.')
    parser.add_argument('--image_dataset_path', type=str, required=True, help='Path to the image dataset.')
    parser.add_argument('--grid_folder', type=str, default='./', help='Path to save the output grids (default: current directory).')
    parser.add_argument('--cluster_folder', type=str, default='./', help='Path to save the clustered images (default: current directory).')
    parser.add_argument('--feature_dict_path', type=str, default='./', help='Path to save/load the feature dictionary (default: current directory).')
    parser.add_argument('--num_clusters', type=int, default=5, help='Number of clusters.')
    parser.add_argument('--use_feature_dict', action='store_true', help='Use existing feature dictionary instead of recalculating.')
    parser.add_argument('--model', type=str, choices=['vit', 'resnet'], default='vit', help='Model to use for feature extraction (default: ViT).')
    parser.add_argument('--clustering_method', type=str, choices=['kmeans', 'gmm'], default='kmeans', help='Clustering method to use (default: KMeans).')

    args = parser.parse_args()

    if args.use_feature_dict and args.feature_dict_path is None:
        parser.error('--feature_dict_path is required when --use_feature_dict is set')

    os.makedirs(args.cluster_folder, exist_ok=True)
    os.makedirs(args.grid_folder, exist_ok=True)

    if args.model == 'vit':
        model, preprocess = load_vit_model()
        model_type = 'vit'
    else:
        model, preprocess = load_resnet50_model()
        model_type = 'resnet'

    if not args.use_feature_dict:
        image_feature_dict = create_feature_dict(args.image_dataset_path, model, preprocess, model_type)
        save_dict(image_feature_dict, args.feature_dict_path)
    else:
        image_feature_dict = read_dict(args.feature_dict_path)

    cluster_data = get_clustered_data(image_feature_dict, args.num_clusters, args.clustering_method)
    create_cluster_folders(cluster_data, args.image_dataset_path, args.cluster_folder)
    create_cluster_grids(cluster_data, args.image_dataset_path, args.grid_folder)
