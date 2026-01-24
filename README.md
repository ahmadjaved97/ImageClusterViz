# ImageAtlas

## Overview

ImageAtlas is a comprehensive toolkit designed to organize, clean, and analyze image datasets.

⚠️ Note: ImageAtlas is currently in active development. The current version focuses on clustering and visualization functionality, with additional features coming soon.

Perfect for dataset curation, duplicate detection, quality control, and exploratory data analysis.

## 📦 Installation

**Basic Installation**

```
pip install imageatlas
```

**Full Installation**

```
pip install imageatlas[full]
```

**Note on CLIP**: If you wish to use the CLIP model, you must install it manually from GitHub using:

```
pip install git+https://github.com/openai/CLIP.git
```

**From Source**
```
git clone https://github.com/ahmadjaved97/ImageAtlas.git
cd ImageAtlas
pip install -e .
```

## 🚀 Quick Start

### Minimal Working Example

```python
import os
from imageatlas import ImageClusterer

# Initialize clusterer
clusterer = ImageClusterer(
    model='dinov2',           # State-of-the-art features
    clustering_method='kmeans',
    n_clusters=10,
    device='cuda'             # or 'cpu'
)

# Run clustering on your images
results = clusterer.fit("./path/to/images")

# Save results to JSON
results.to_json("./output/clustering_results.json")

# Create visual grids for each cluster
results.create_grids(
    image_dir="./path/to/images",
    output_dir="./output/grids"
)

# Organize images into cluster folders
results.create_cluster_folders(
    image_dir="./path/to/images",
    output_dir="./output/clusters"
)
```

That's it! Your images are now clustered, visualized, and organized.

## Available Models & Algorithms

### Feature Extraction Models

| Model            | Variants                                            |
| ---------------- | --------------------------------------------------- |
| **DINOv2**       | `vits14`, `vitb14`, `vitl14`, `vitg14`              |
| **ViT**          | `b_16`, `b_32`, `l_16`, `l_32`, `h_14`              |
| **ResNet**       | `18`, `34`, `50`, `101`, `152`                      |
| **EfficientNet** | `s`, `m`, `l`                                       |
| **CLIP**         | `RN50`, `RN101`, `ViT-B/32`, `ViT-B/16`, `ViT-L/14` |
| **ConvNeXt**     | `tiny`, `small`, `base`, `large`                    |
| **Swin**         | `t`, `s`, `b`, `v2_t`, `v2_s`, `v2_b`               |
| **MobileNetV3**  | `small`, `large`                                    |
| **VGG16**        | \-                                                  |

### Clustering Algorithms

| Algorithm   | Parameters                        |
| ----------- | --------------------------------- |
| **K-Means** | `n_clusters`                      |
| **HDBSCAN** | `min_cluster_size`, `min_samples` |
| **GMM**     | `n_components`, `covariance_type` |

### Dimensionality Reduction

| Method                    | Parameters                                |
| --------------------------| ----------------------------------------- |
| **PCA**                   | `n_components`, `whiten`                  |
| **UMAP**                  | `n_components`, `n_neighbors`, `min_dist` |
| **t-SNE(in development)** | `n_components`, `perplexity`              |


## 📝 Citation

If you use ImageAtlas in your research, please cite:

```bibtex
@software{imageatlas2024,
  author = {Javed, Ahmad},
  title = {ImageAtlas: A Toolkit for Organizing and Analyzing Image Datasets},
  year = {2024},
  url = {https://github.com/ahmadjaved97/ImageAtlas}
}
```
##  Acknowledgments

- [DINOv2](https://github.com/facebookresearch/dinov2): Facebook Research
- [CLIP](https://github.com/openai/CLIP): OpenAI
- [Vision Transformers](https://github.com/google-research/vision_transformer): Google Research
- Built with [PyTorch](https://github.com/pytorch/pytorch), [scikit-learn](https://github.com/scikit-learn/scikit-learn), and [OpenCV](https://github.com/opencv/opencv)


### Sample Output
- Dataset Used: [Fruit and Vegetable Classification](https://www.kaggle.com/code/abdelrahman16/fruit-and-vegetable-classification/input)
- Number of Clusters: 8
- Model Used: ViT
- Clustering Method: Kmeans
- Output:
    <p align="center">
        <img src="./output_grids/cluster_0.jpg" alt="Image 1" width="250" height="250">
        <img src="./output_grids/cluster_1.jpg" alt="Image 2" width="250" height="250">
        <img src="./output_grids/cluster_2.jpg" alt="Image 3" width="250" height= "250">
        <img src="./output_grids/cluster_3.jpg" alt="Image 3" width="250" height= "250">
        <img src="./output_grids/cluster_4.jpg" alt="Image 3" width="250" height= "250">
        <img src="./output_grids/cluster_5.jpg" alt="Image 3" width="250" height= "250">
        <img src="./output_grids/cluster_6.jpg" alt="Image 3" width="250" height= "250">
        <img src="./output_grids/cluster_7.jpg" alt="Image 3" width="250" height= "250">
    </p>


