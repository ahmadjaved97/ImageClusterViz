"""
Batch processing for efficient feature extraction.
"""

import torch
import numpy as np
from PIL import Image
import warnings

class BatchProcessor:
    """
    Handles batch processing of images through feature extractors.

    Manages batching, device placement and memory cleanup.
    """

    def __init__(
        self,
        batch_size=8,
        device='cpu',
        clear_cache=True
    ):
        """
        Initialize batch processor.

        Args:
            batch_size: Number of images to process at once.
            device: Device to use ('cpu', 'cuda', 'cuda:0', etc.)
            clear_cache: Whether to clear GPU cache after each batch
        """

        self.batch_size = batch_size
        self.device = device
        self.clear_cache = clear_cache

    
    def validate_device(self, device):
        """
        Validate and normalize device string.
        """
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if device.startswith('cuda'):
            if not torch.cuda.is_available():
                warnings.warn(f"CUDA not available, falling back to CPU")
                return 'cpu'
        
        return device
    

    def process_batch(
        self,
        images,
        extractor,
        return_numpy
    ):
        """
        Process a batch of extractors through the feature extractor.
        Args:
            images: List of PIL Images.
            extractor: Feature extractor with extract_features method
            return_numpy: Whether to return numpy array (vs torch tensor)
        
        Returns:
            Array of feature vectors, shape (batch_size, feature_dim)
        """
        # TODO: use the correct batching method in the feature_extractors module.

        if not images:
            return np.array([])

        try:
            # Check if extractor support batch processing.
            if hasattr(extractor, 'extract_batch'):
                features = extractor.extract_batch(images)
            else:
                # Fall back to loop.
                warnings.warn(
                    f"Extractor {type(extractor).__name__} doesn't support extract_batch()."
                    "Using slower one-by-one processing. Consider updating the extractor.",
                    UserWarning
                    )
                
                features_list = []
                for image in images:
                    feature = extractor.extract_features(image)
                    features_list.append(feature)
                features = np.vstack(features_list)
        
        except Exception as e:
            warnings.warn(f"Failed to extract features : {str(e)}")
            # Try fallback method
            features_list = []
            for image in images:
                try:
                    feature = extractor.extract_features(image)
                    features_list.append(feature)
                except Exception as e:
                    # Use zero vector as a placeholder for failed extraction
                    if features_list:
                        feature_dim = features_list[0].shape[0]
                        features_list.append(np.zeros(feature_dim))
                    else:
                        # Can't determine feature dim
                        raise
                
            # Stack into batch
            features = np.vstack(features_list)

        # Clear GPU cache if needed
        if self.clear_cache and 'cuda' in self.device:
            torch.cuda.empty_cache()
        
        return features
    
    def estimate_memory_usage(
        self,
        n_images,
        feature_dim,
        dtype = np.float32,
    ):
        """
        Estimate memory usage for a batch.

        Args:
            n_images: Number of images in a batch
            feature_dim: Dimensions of feature vector
            dtype: Data type of features

        Returns:
            Estimated memory in GB
        """

        bytes_per_element = np.dtype(dtype).itemsize
        total_bytes = n_images * feature_dim * bytes_per_element

        return total_bytes / (1024 **3) # Convert to GB
    
    def get_device_info(self):
        """
        Get information about the current device.
        """

        info = {
            'device': self.device,
            'device_type': 'cuda' if 'cuda' in self.device else 'cpu'
        }

        if 'cuda' in self.device:
            if torch.cuda.is_available():
                info['cuda_available'] = True
                info['cuda_device_name'] = torch.cuda.get_device_name(0)
                info['cuda_memory_allocated'] = torch.cuda.memory_allocated(0) / (1024 ** 3)   # GB
                info['cuda_memory_reserved'] = torch.cuda.cuda_memory_reserved(0) / (1024 **3) # GB
            else:
                info['cuda_available'] = False
        
        return info
    

    
    
    
