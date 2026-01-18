import os
import numpy as np
from pathlib import Path

from features import FeaturePipeline
from feature_extractors import create_feature_extractor
from clustering import create_clustering_algorithm
from reduction import create_reducer

class ImageClusterer:
    """
    Main API for Image Clustering.
    """

    def __init__(
        self,
        model='resnet',
        model_variant=None,
        n_clusters=5,
        clustering_method='kmeans',
        reducer=None,
        n_components=50,
        batch_size=32,
        device='auto',
        random_state=42,
        verbose=True
    ):
        """
        Initialize ImageClusterer.
        """

        self.model = model
        self.model_variant = model_variant
        self.n_clusters = n_clusters
        self.clustering_method = clustering_method
        self.reducer = reducer
        self.n_components = n_components
        self.batch_size = batch_size
        self.device = device
        self.random_state = random_state
        self.verbose = verbose


        # Initializer components
        self._extractor = None
        self._feature_pipeline = None
        self._reducer_instance = None
        self._clusterer_instance = None

        # Storage for fitted data
        self.features_ = None
        self.reduced_features_ = None
        self.filenames_ = None
        self.results_ = None
        self.is_fitted_ = False

    
    def _validate_device(self, device):
        """Validate and normalize device string."""
        if device == 'auto':
            import torch
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        
        return device
    
    def _get_extractor(self):
        """
        Get or create feature extractor.
        """
        if self._extractor is None:
            if self.verbose:
                print(f"Creating feature extractor: {self.model}")
            
            self._extractor = create_feature_extractor(
                model_type=self.model,
                variant=self.model_variant,
                device=self.device
            )
        
        return self._extractor
    

    def _get_reducer(self):
        """
        Get or create dimensionality reducer.
        """
        if self.reducer is None:
            return None
        
        if self._reducer_instance is None:
            if self.verbose:
                print(f"Creating dimensionality reducer: {self.reducer}")
            
            self._reducer_instance = create_reducer(
                algorithm=self.reducer,
                n_components=self.n_components,
                random_state=self.random_state
            )
        
        return self._reducer_instance
    
    def _get_clusterer(self):
        """
        Get or create clustering algorithm.
        """

        if self._clusterer_instance is None:
            if self.verbose:
                print(f"Creating clusterer: {self.clustering_method}")
        
        # Create clusterer based on method
        if self.clustering_method in ['kmeans', 'gmm']:
            self._clusterer_instance = create_clustering_algorithm(
                self.clustering_method,
                n_clusters=self.n_clusters if self.clustering_method == 'kmeans' else None,
                n_components=self.n_clusters if self.clustering_method == 'gmm' else None,
                random_state=self.random_state
            )
        elif self.clustering_method == 'hdbscan':
            self._clusterer_instance = create_clustering_algorithm(
                'hdbscan',
                auto_params=True,
                random_state=self.random_state
            )
        
        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")
        
        return self._clusterer_instance
    

    

    

    
