import os
import numpy as np
from pathlib import Path

from ..features import FeaturePipeline
from ..features.extractors import create_feature_extractor
from ..clustering.factory import create_clustering_algorithm
from ..reduction.factory import create_reducer
from .results import ClusteringResults


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
        self.device = self._validate_device(device)
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
    
    def _get_feature_pipeline(self):
        """
        Get or create feature extraction pipeline.
        """
        if self._feature_pipeline is None:
            extractor = self._get_extractor()
            self._feature_pipeline = FeaturePipeline(
                extractor=extractor,
                batch_size=self.batch_size,
                device=self.device,
                verbose=self.verbose
            )
        
        return self._feature_pipeline
    

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
    
    def fit(
        self,
        image_dir,
        pattern="*",
        recursive=False,
        cache_path=None,
        use_cache=False
    ):
        """
        Fit the clusterer to the images in a directory.
        This method:
        1. Extract features from images.
        2. Optionally reduces dimensionality.
        3. Clusters the features.
        4. Returns results object.
        """

        if self.verbose:
            print("="*60)
            print("IMAGE CLUSTERING PIPELINE")
            print("="*60)

        # Step 1: Feature Extraction
        if self.verbose:
            print("\nStep 1: Feature Extraction.")
            
        pipeline = self._get_feature_pipeline()
        
        # Check for cached features
        if use_cache and cache_path and os.path.exists(cache_path):
            if self.verbose:
                print(f"Loading cached features from: {cache_path}")
            pipeline.load(cache_path)
        else:
            # Extract features
            pipeline.extract_from_directory(
                image_dir,
                pattern=pattern,
                recursive=recursive
            )

            # Save cache if path provided
            if cache_path:
                pipeline.save(cache_path)
        
        # Get features and filenames
        self.features_ = pipeline.get_features()
        self.filenames_= pipeline.get_filenames()

        if self.verbose:
            print(f"  Extracted features: {self.features_.shape}")
        
        # Step 2: Dimensionality Reduction (optional)
        features_for_clustering = self.features_
        
        if self.reducer is not None:
            if self.verbose:
                print(f"\nStep 2: Dimensionality Reduction ({self.reducer.upper()})")
            
            reducer = self._get_reducer()
            self.reduced_features_ = reducer.fit_transform(self.features_)
            features_for_clustering = self.reduced_features_

            if self.verbose:
                print(f"   Reduced to: {self.reduced_features_.shape}")


                # Print variance info for  PCA
                metadata = reducer.get_metadata()
                if 'total_variance_explained' in metadata:
                    print(f"  Variance explained: {metadata['total_variance_explained']:.2%}")
        
        else:
            if self.verbose:
                print(f"\n2: Dimensionality Reduction (skipped)")
            self.reduced_features_ = None
        

        # Step 3: Clustering
        if self.verbose:
            print(f"\n3. Step 3: Clustering ({self.clustering_method.upper()})")
        
        clusterer = self._get_clusterer()
        clustering_result = clusterer.fit_predict(
            features_for_clustering,
            filenames=self.filenames_
        )

        if self.verbose:
            print(f"   Found {clustering_result.n_clusters} clusters")
            print(f"   Cluster sizes: {clustering_result.get_cluster_sizes()}")
        
        # Step 4: Create results object
        metadata = {
            'model_type': self.model,
            'model_variant': self.model_variant,
            'clustering_method': self.clustering_method,
            'n_clusters': clustering_result.n_clusters,
            'reducer': self.reducer,
            'n_components': self.n_components if self.reducer else None,
            'device': self.device,
            'batch_size': self.batch_size
        }


        # Add clustering metadata
        if clustering_result.metadata:
            metadata['clustering_metadata'] = clustering_result.metadata
        
        self.results_ = ClusteringResults(
            cluster_labels=clustering_result.cluster_labels,
            cluster_dict=clustering_result.cluster_dict,
            filenames=self.filenames_,
            features=self.features_,
            reduced_features=self.reduced_features_,
            n_clusters=clustering_result.n_clusters,
            metadata=metadata
        )

        self.is_fitted_ = True

        if self.verbose:
            print("\n" + "="*60)
            print("CLUSTERING COMPLETE")
            print("="*60)
            print(f"\n{self.results_.summary()}")
        
        return self.results_
    

    def fit_features(
        self,
        features,
        filenames=None
    ):
        """
        Fit the clusterer to pre-computed featuers.
        """

        if filenames is None:
            filenames = [f"sample_{i}" for i in range((len(features)))]
        
        self.features_ = features
        self.filenames_ = filenames

        if self.verbose:
            print("="*60)
            print("CLUSTERING PRE-COMPUTED FEATURES")
            print("="*60)
            print(f"\nInput features: {features.shape}")
        

        # Apply dimensionality reduction if specified
        features_for_clustering = features
        
        if self.reducer is not None:
            if self.verbose:
                print(f"\nApplying {self.reducer.upper()} reduction...")
            
            reducer = self._get_reducer()
            self.reduced_features_ = reducer.fit_transform(features)
            features_for_clustering = self.reduced_features_
            
            if self.verbose:
                print(f"   Reduced to: {self.reduced_features_.shape}")
        else:
            self.reduced_features_ = None
        
        # Cluster
        if self.verbose:
            print(f"\nClustering with {self.clustering_method.upper()}...")
        
        clusterer = self._get_clusterer()
        clustering_result = clusterer.fit_predict(
            features_for_clustering,
            filenames=filenames
        )
        
        if self.verbose:
            print(f"   Found {clustering_result.n_clusters} clusters")
        
        # Create results
        metadata = {
            'clustering_method': self.clustering_method,
            'n_clusters': clustering_result.n_clusters,
            'reducer': self.reducer,
            'n_components': self.n_components if self.reducer else None,
            'clustering_metadata': clustering_result.metadata,
        }
        
        self.results_ = ClusteringResults(
            cluster_labels=clustering_result.cluster_labels,
            cluster_dict=clustering_result.cluster_dict,
            filenames=filenames,
            features=features,
            reduced_features=self.reduced_features_,
            n_clusters=clustering_result.n_clusters,
            metadata=metadata
        )
        
        self.is_fitted_ = True
        
        if self.verbose:
            print(f"\n{self.results_.summary()}")
        
        return self.results_

    
    def get_results(self):
        """
        Get clustering results
        """
        if not self.is_fitted_:
            raise RuntimeError("Clusterer has not been fitted yet. Call fit() first.")
        return self.results_
    
    def __repr__(self):
        """
        String representation.
        """
        status = "fitted" if self.is_fitted_  else "not fitted"
        return (f"ImageClusterer(model='{self.model}', "
                f"n_clusters={self.n_clusters}, "
                f"method='{self.clustering_method}', "
                f"status={status})")



    

    
