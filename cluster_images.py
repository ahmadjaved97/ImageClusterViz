import os
from PIL import Image
import numpy as np
from rich import print
import cv2
from rich.progress import track
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import shutil
import argparse
from utils import create_image_grid,read_dict, save_dict
from feature_extractors.factory import create_feature_extractor
from dimensionality_reduction import create_reducer



def get_clustered_data(feature_dict, num_clusters=5, clustering_method='kmeans', feature_reduction=None):
    """
    Clusters the feature vectors from images into a specified number of clusters using either KMeans or Gaussian Mixture Models (GMM). 
    It returns a dictionary mapping each cluster ID to a list of corresponding image filenames.
    """
    filenames = list(feature_dict.keys())
    feature_vectors = list(feature_dict.values())
    feature_vectors = np.array(feature_vectors)
    num_feature_vectors = len(feature_vectors)

    if feature_reduction:
        reduced_features = feature_reduction.fit_transform(feature_vectors)
        # print(reduced_features.explained_variance_ratio_.cumsum()[:20])
        feature_vectors = reduced_features


    if clustering_method == 'gmm':
        gmm = GaussianMixture(n_components=num_clusters, random_state=42)
        cluster_labels = gmm.fit_predict(feature_vectors)
    
    elif clustering_method == 'hdbscan':
        import hdbscan
        if num_feature_vectors < 2000:
            min_cluster_size = 10
            min_samples = 1
        elif num_feature_vectors >= 2000 and num_feature_vectors < 10000:
            min_cluster_size = 20
            min_samples = 3
        else:
            min_cluster_size = 30
            min_samples = 5

        hdbscan_clustering = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                            min_samples=min_samples,
                                            metric='euclidean')
        cluster_labels = hdbscan_clustering.fit_predict(feature_vectors)
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


def create_feature_dict(dataset_path, feature_extractor,  n=10, save_path="."):
    """
    Creates a dictionary mapping image filenames to their corresponding feature vectors, extracted using the specified model.
    """
    # Check if the feature dictionary file already exists and load it
    feature_dict = {}
    if os.path.exists(os.path.join(save_path, "feature_dictionary.pkl")):
        feature_dict = read_dict(save_path)
        print(f"[yellow]Loaded existing feature dictionary with {len(feature_dict)} items.")
    else:
        print(f"[yellow]No existing feature dictionary found. Creating a new one at: {save_path}")
    
    file_list = os.listdir(dataset_path)
    extensions = (".jpg", ".jpeg", ".png")

    for idx, file in enumerate(track(file_list, total=len(file_list), description="Getting image features", complete_style="yellow")):
        if file.endswith(extensions):
            # Skip the file if it's already in the existing feature_dict
            if file in feature_dict:
                print(f"Skipping [green]{file}[/green], already in feature_dict.")
                continue

            image_path = os.path.join(dataset_path, file)
            try:
                image = Image.open(image_path).convert('RGB')
                image_feature = feature_extractor.extract_features(image)[0]
                feature_dict[file] = image_feature
            except Exception as e:
                print(f"Error processing [red]{file}[/red]: {str(e)}")

            # Save the feature dict every `n` iterations
            if (idx + 1) % n == 0:
                save_dict(feature_dict, save_path)
                # feature_dict.clear()  # Clear the in-memory dict to free up memory
    
    # Final save after the loop completes
    if feature_dict:
        save_dict(feature_dict, save_path)

    return feature_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cluster images using ViT or ResNet and create grids.')
    parser.add_argument('--image_dataset_path', type=str, required=True, help='Path to the image dataset.')
    parser.add_argument('--grid_folder', type=str, default='./', help='Path to save the output grids (default: current directory).')
    parser.add_argument('--cluster_folder', type=str, default='./', help='Path to save the clustered images (default: current directory).')
    parser.add_argument('--feature_dict_path', type=str, default='./', help='Path to save/load the feature dictionary (default: current directory).')
    parser.add_argument('--num_clusters', type=int, default=5, help='Number of clusters.')
    parser.add_argument('--use_feature_dict', action='store_true', help='Use existing feature dictionary instead of recalculating.')
    parser.add_argument('--model', type=str, choices=['vit', 'resnet', 'vgg', 'mobilenet', 'clip', 'dinov2', 'swin', 'efficientnet', 'convnext'], default='vit', help='Model to use for feature extraction (default: ViT).')
    parser.add_argument('--clustering_method', type=str, choices=['kmeans', 'gmm', 'hdbscan'], default='kmeans', help='Clustering method to use (default: KMeans).')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cpu', help='Device used for inference')
    parser.add_argument('--reducer', type=str, choices=['pca', 'umap'], default=None, help='Dimensionality reduction algorithm to be used.')
    parser.add_argument('--reduced_components', type=int, default=50, help='Number of features after dimensionality reduction.')
    # add argument  and modify function to limit the number of images for clustering. also provide a check to see if the number defined is <= the number
    # of images in the folder.

    # Make a resume parameter(by default: False) if true, then the feature_dict won't be deleted
    
    # make a default feature dictionary name and pass it in the functions
    args = parser.parse_args()

    if args.use_feature_dict and args.feature_dict_path is None:
        parser.error('--feature_dict_path is required when --use_feature_dict is set')

    os.makedirs(args.cluster_folder, exist_ok=True)
    os.makedirs(args.grid_folder, exist_ok=True)

    feature_extractor = create_feature_extractor(model_type=args.model, device=args.device)

    dimensionality_reducer = None
    if args.reducer:
        print("Reduction used: ", args.reducer)
        dimensionality_reducer = create_reducer(algorithm=args.reducer, n_components=args.reduced_components)
    


    if not args.use_feature_dict:
        # Delete existing feature dict if present
        if args.feature_dict_path and os.path.exists(os.path.join(args.feature_dict_path, "feature_dictionary.pkl")):
            print(f"Removing exisiting feature dictionary at: {os.path.join(args.feature_dict_path, 'feature_dictionary.pkl')}")
            os.remove(os.path.join(args.feature_dict_path, "feature_dictionary.pkl"))
        # write a function/feature to add more info to feature dict such as dataset folder, model used, if they
        # match with the argument then load the old feature dict otherwise create a new one.
        image_feature_dict = create_feature_dict(args.image_dataset_path, feature_extractor, n=10, save_path=args.feature_dict_path)
    else:
        # should be able to take a file path(such as .csv, .txt) as well/ currently only takes a folder path.
        image_feature_dict = read_dict(args.feature_dict_path)

    cluster_data = get_clustered_data(image_feature_dict, args.num_clusters, args.clustering_method, dimensionality_reducer)
    create_cluster_folders(cluster_data, args.image_dataset_path, args.cluster_folder)
    create_cluster_grids(cluster_data, args.image_dataset_path, args.grid_folder)
