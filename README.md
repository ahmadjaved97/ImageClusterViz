# Image Clustering and Grid Creation

## Overview

This project focuses on organizing and visualizing a collection of images using state-of-the-art deep learning models. It uses Vision Transformer (ViT) and ResNet-50 models to extract features from images, which are then used to cluster the images into groups based on visual similarity. The resulting clusters are represented both through the creation of folders containing the clustered images and visually through composite grid images.

This is particularly useful for applications involving large datasets of images where manual sorting and organization would be impractical. By automatically grouping similar images, it helps in data management, curation, and analysis tasks.

### Key Features:
- **Feature Extraction:** Utilizes pre-trained deep learning models to extract features from images.
- **Clustering:** Implements clustering algorithms such as KMeans and Gaussian Mixture Models (GMM) to categorize images based on their features.
- **Visualization:** Provides an intuitive visualization of the clustering results through grid images.
- **Data Management:** Automates the organization of images into directory structures for easy access and analysis.

The tool is designed to be flexible, allowing users to choose between different models and clustering methods, and to either calculate features on the fly or use precomputed ones.

## Usage

### Prerequisites
Ensure all dependencies are installed using the following command:

```pip install -r requirements.txt```

### Running the Script
```
python cluster_images.py --image_dataset_path <path_to_image_dataset> \
                      --grid_folder <path_to_save_grid_images> \
                      --cluster_folder <path_to_save_clustered_images> \
                      --feature_dict_path <path_to_save_or_load_feature_dict> \
                      --num_clusters <number_of_clusters> \
                      --model <model_type> \
                      --clustering_method <clustering_method> \
                      [--use_feature_dict]
```

### Example Command
```
python cluster_images.py --image_dataset_path ./images \
                      --grid_folder ./grids \
                      --cluster_folder ./clusters \
                      --feature_dict_path ./features \
                      --num_clusters 5 \
                      --model vit \
                      --clustering_method kmeans \
                      --use_feature_dict
```

### Argument Description:
- **--image_dataset_path**: Path to the folder containing images.
- --**grid_folder**: Path to save the generated grid images (default: current directory).
- --**cluster_folder**: Path to save images sorted into clusters (default: current directory).
- --**feature_dict_path:** Path to save/load the feature dictionary (default: current directory).
- --**num_clusters**: The number of clusters to create.
- --**model**: The model type to use for feature extraction, either vit or resnet.
- --**clustering_method**: The method to use for clustering, either kmeans or gmm.
- --**use_feature_dict**: Use this flag if you want to use an existing feature dictionary instead of recalculating it.
