# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-01-22
### Added
- Core `ImageClusterer` API for high-level clustering workflows.
- Feature extraction support for DINOv2, ResNet, ViT, CLIP, Swin, and more.
- Clustering algorithms: K-Means, GMM, and HDBSCAN.
- Dimensionality reduction wrappers for PCA, UMAP, and t-SNE.
- Visualization tools (`GridVisualizer`) for creating image grids from clusters.
- HDF5 caching system for efficient feature storage.
- Export functionality to CSV, JSON, and Excel.