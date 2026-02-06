"""
Main feature extraction pipeline.
"""

import os
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings

from .cache import HDF5Cache, FeatureCache
from .batch import BatchProcessor
from .loaders import ImageLoader
from .metadata import FeatureMetadata

class FeaturePipeline:
    """
    Main pipeline for feature extraction.

    Handles batch processing, caching, progress tracking, and error recovery.

    Example:
        >>> from features import FeaturePipeline
        >>> from feature_extractors import create_feature_extractor
        >>> 
        >>> extractor = create_feature_extractor('dinov2', device='cuda')
        >>> pipeline = FeaturePipeline(extractor, batch_size=32)
        >>> 
        >>> result = pipeline.extract_from_directory('./images')
        >>> pipeline.save('./features/features.h5')
    """

    def __init__(
        self,
        extractor,
        batch_size=8,
        device='cpu',
        cache_backend='hdf5',
        max_image_size=None,
        verbose=True
    ):
        """
        Initialize feature extraction pipeline.

        Args:
            extractor: Feature extractor (from feature_extractors module)
            batch_size: Number of images to process at once
            device: Device for processing ('cpu', 'cuda')
            cache_backend: Cache backend to use ('hdf5')
            max_image_size: Optional max size for images (width, height)
            verbose: Whether to show progress bars
        """

        self.extractor = extractor
        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose

        # Initialize components
        self.batch_processor = BatchProcessor(
            batch_size=batch_size,
            device=device,
            clear_cache=True
        )

        self.image_loader = ImageLoader(
            max_size=max_image_size,
            convert_mode='RGB',
            handle_exif=True
        )

        # Initialize cache backend
        if cache_backend == 'hdf5':
            self.cache = HDF5Cache(compression='gzip', compression_level=4)
        else:
            raise ValueError(f"Unknown cache backend: {cache_backend}")
        
        # Storage for extracted features
        self.features = None
        self.filenames = []
        self.metadata = None

        # Statistics
        self.n_processed = 0
        self.n_corrupted = 0
        self.n_skipped = 0
        self.extraction_time = 0.0

    
    def extract_from_directory(
        self,
        directory,
        pattern = "*",
        recursive = True,
        save_every = None,
        save_path = None
    ):

        """
        Extract features from all images in a directory.

        Args:
            directory: Directory containing images
            pattern: Glob pattern for filenames
            recursive: Whether to search recursively
            save_every: Save checkpoint every N images (optional)
            save_path: Path for checkpoint saves (required if save_every is set)
        
        Returns:
            Self for method chaining
        """

        # Find all images
        image_paths = self.image_loader.find_images(
            directory,
            pattern=pattern,
            recursive=recursive
        )

        if not image_paths:
            raise ValueError(f"No images found in {directory}")
        
        if self.verbose:
            print(f"Found {len(image_paths)} images in {directory}")
        

        # Extract features
        return self.extract_from_files(
            image_paths,
            save_every=save_every,
            save_path=save_path
        )
    

    def extract_from_files(
        self,
        file_paths,
        save_every=None,
        save_path=None
    ):

        """
        Extract features from a list of filepaths.

        Args:
            file_paths: List of image file paths
            save_every: Save checkpoint every N images (optional)
            save_path: Path for checkpoint saves (required if save_every is set)

        Returns:
            Self for method chaining
        """

        if save_every is not None and save_path is None:
            raise ValueError("save_path must be provided when save_every is set")
        
        # Reset state
        self.features = None
        self.filenames = []
        self.n_processed = 0
        self.n_corrupted = 0
        self.n_skipped = 0

        # Get model information for metadata
        model_info = self._get_model_info()

        # Create batches
        batches = list(self.image_loader.create_batches(file_paths, self.batch_size))

        # Process batches
        start_time = time.time()

        feature_list = []
        successful_filenames = []

        progress_bar = tqdm(
            batches,
            desc="Extracting features",
            disable=not self.verbose,
            unit="batch"
        )

        for batch_idx, batch_paths in enumerate(progress_bar):
            # Load images
            images, laoded_paths, failed_paths = self.image_loader.load_batch(batch_paths)

            self.n_corrupted += len(failed_paths)

            if not images:
                # All images in batch failed.
                self.n_skipped += len(batch_paths)
                continue
            
            try:
                # Extract features for batch
                batch_features = self.batch_processor.process_batch(
                    images,
                    self.extractor,
                    return_numpy=True
                )

                feature_list.append(batch_features)
                successful_filenames.extend(laoded_paths)

                self.n_processed += len(images)

                # Update progress bar

                if self.verbose:
                    progress_bar.set_postfix({
                        'processed': self.n_processed,
                        'corrupted': self.n_corrupted
                    })
                
                # Save checkpoint if needed
                if save_every and (batch_idx + 1) % save_every == 0:
                    self.save_checkpoint(
                        feature_list,
                        successful_filenames,
                        save_path,
                        batch_idx
                    )
            
            except Exception as e:
                warnings.warn(f"Error processing batch {batch_idx}: {str(e)}")
                self.n_skipped += len(images)
                continue
        
        self.extraction_time = time.time() - start_time

        # Combine all features
        if feature_list:
            self.features = np.vstack(feature_list)
            self.filenames = successful_filenames
        else:
            raise RuntimeError("No features were extracted successfully")
        

        # Create metadata
        self.metadata = FeatureMetadata(
            model_type=model_info['model_type'],
            model_variant=model_info.get('variant'),
            feature_dim=self.features.shape[1],
            n_samples=self.n_processed,
            device=self.device,
            batch_size=self.batch_size,
            total_time=self.extraction_time,
            images_per_second=self.n_processed / self.extraction_time if self.extraction_time > 0 else 0,
            n_corrupted=self.n_corrupted,
            n_skipped=self.n_skipped
        )

        if self.verbose:
            print(f"\nExtraction completed!")
            print(self.metadata.summary())
        
        return self
    

    def save(self, path, format='hdf5'):
        """
        Save extracted features to disk.

        Args:
            path: Path to save features
            format: Format to use ('hdf5')
        """

        if self.features is None or self.metadata is None:
            raise RuntimeError("No features to save. Run extract_from_* first")
        
        if format != "hdf5":
            raise ValueError(f"Unsupported format: {format}. Only 'hdf5' is supported.")
        
        self.cache.save(
            features=self.features,
            filenames=self.filenames,
            metadata=self.metadata,
            path=path
        )

        if self.verbose:
            print(f"Features saved to: {path}")
    
    def load(self, path):
        """
        Load features from disk.

        Args:
            path: Path to feature cache
        
        Returns: Self for method chaining
        """

        self.features, self.filenames, self.metadata = self.cache.load(path)

        if self.verbose:
            print(f"Loaded features from: {path}")
            print(self.metadata.summary())
        
        return self
    
    def get_features(self):
        """
        Get extracted features as numpy array.
        """
        if self.features is None:
            raise RuntimeError("No features available. Run extract_features_from_* or load() first.")
        return self.features
    
    def get_filenames(self):
        """
        Get list of filenames corresponding to features.
        """
        return self.filenames
    
    def get_feature_dict(self):
        """
        Get features as dictionary

        Returns:
            Dictionary mapping filenames to feature vectors
        """

        if self.features is None:
            raise RuntimeError("No features available. run extract_from_* or load() first.")
        
        return {fn: feat for fn, feat in zip(self.filenames, self.features)}
    
    def get_metadata(self):
        """
        Get feature metadata.
        """
        if self.metadata is None:
            raise RuntimeError("No metadata available. Run extract_from_* or load() first.")
        
        return self.metadata
    
    def _get_model_info(self):
        """
        Get information about the feature extractor.
        """

        info = {
            'model_type': 'unknown',
            'variant': None
        }

        # Try to get model type from extractor
        if hasattr(self.extractor, 'get_algorithm_name'):
            info['model_type'] = self.extractor.get_algorithm_name()
        elif hasattr(self.extractor, '__class__'):
            info['model_type'] = self.extractor.__class__.__name__.replace('Extractor', '').lower()
        

        # Try to get variant
        if hasattr(self.extractor, 'variant'):
            info['variant'] = self.extractor.variant
        
        return info
    
    def _save_checkpoint(
        self,
        feature_list,
        filenames,
        save_path,
        batch_idx
    ):
        """
        Save intermediate checkpoint.
        """

        checkpoint_path = save_path.replace(".h5", f'_checkpoint_{batch_idx}.h5')

        # Combine features so far
        features_so_far = np.vstack(feature_list)

        # Create temporary metadata
        temp_metadata = FeatureMetadata(
            model_type=self._get_model_info()['model_type'],
            feature_dim=features_so_far.shape[1],
            n_samples=len(filenames),
            device=self.device,
            batch_size=self.batch_size,
            description=f"Checkpoint at batch {batch_idx}"
        )

        self.cache.save(
            features=features_so_far,
            filenames=filenames,
            metadata=temp_metadata,
            path=checkpoint_path
        )

        if self.verbose:
            print(f"\nCheckpoint saved: {checkpoint_path}")
