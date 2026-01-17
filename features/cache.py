"""
Feature cache backend for efficient storage and retrieval.
"""

import h5py
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
import json
import warnings
from .metadata import FeatureMetadata


class FeatureCache(ABC):
    """
    Abstract base class for feature cache backends.
    """

    @abstractmethod
    def save(
        self,
        features,
        filenames,
        metadata,
        path
    ):
        """
        Save features to cache.
        """
        pass

    @abstractmethod
    def load(self, path):
        """
        Load features from cache.
        """
        pass
    
    @abstractmethod
    def exists(self, path):
        """
        Check if cache exists.
        """
        pass
    

class HDF5Cache(FeatureCache):
    """
    HDF5-based feature cache.
    """

    def __init__(self, compression='gzip', compression_level=4):
        """
        Initialize HDF5 cache.

        Args:
        compression: Compression algorithm ('gzip', 'lzf', None)
        compression_level: Compression level (0-9 for gzip)
        """

        self.compression = compression
        self.compression_level = compression_level
    
    def save(
        self,
        features,
        filenames,
        metadata,
        path
    ):
        """
        Save features to HDF5 file.
        """

        # Make sure path has .h5 extension
        if not path.endswith(".h5"):
            path = path + ".h5"
        
        # Create directory if needed
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Save to HDF5
        with h5py.File(path, 'w') as f:
            # Save features with compression
            f.create_dataset(
                'features',
                data=features,
                compression=self.compression,
                compression_opts=self.compression_level if self.compression == 'gzip' else None,
                dtype=np.float32
            )

        # Save filenames as variable-length strings
        dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset('filenames', data=filenames, dtype=dt)

        # Save metadata as attributes
        for key, value in metadata.to_dict().items():
            if isinstance(value, (str, int, float, bool)):
                f.attrs[key] = value
            elif isinstance(value, dict):
                f.attrs[key] = json.dumps(value)
        
        # Also save metadata as seperate JSON for easy access
        metadata_path = path.replace('.h5', '_metadata.json')
        metadata.to_json(metadata_path)
    

    def load(
        self,
        path,
        lazy=False
    ):
        """
        Load features from HDF5 file.
        """

        if not path.endswith(".h5"):
            path = path + ".h5"
        
        if not self.exists(path):
            raise FileNotFoundError("fCache file not found: {path}")
        
        with h5py.File(path, 'r') as f:
            # Load filenames
            filenames = [s.decode('utf-8') if isinstance(s, bytes) else s
                            for s in f['filenames'][:]]
            
            # Load features
            if lazy:
                # Return memory mapped access
                warnings.warn(
                    "Lazy loading not fully implemented. Loading to memory"
                )
                features = f['features'][:]
            else:
                features = f['features'][:]
            
            # Load metadata from attributes
            metadata_dict = {}
            for key, value in f.attr.items():
                if isinstance(value, str) and value.startswith('{'):
                    # Try to parse as JSON
                    try:
                        metadata_dict[key] = json.loads(value)
                    except json.JSONDecodeError:
                        metadata_dict[key] = value
                else:
                    metadata_dict[key] = value
            
            metadata = FeatureMetadata.from_dict(metadata_dict)
        
        return features, filenames, metadata
    
    def load_subset(
        self,
        path,
        indices=None,
        filenames=None
    ):
        """
        Load a subset of features.
        """

        if not path.endswith(".h5"):
            path = path + ".h5"
        
        with h5py.File(path, 'r') as f:
            all_filenames = [s.decode('utf-8') if isinstance(s, bytes) else s
                            for s in f['filenames'][:]]
            
            if filenames is not None:
                # Find indices for requested filenames
                filename_to_idx = {fn: i for i, fn in enumerate(all_filenames)}
                indices = [filename_to_idx[fn] for fn in filenames if fn in filename_to_idx]
            
            if indices is not None:
                features = f['features'][indices]
                loaded_filenames = [all_filenames[i] for i in indices]
            else:
                # ToDO: don't load all just say not available
                features = f['features'][:]
                loaded_filenames = all_filenames
        
        return features, loaded_filenames
    
    def append(
        self,
        path,
        new_features,
        new_filenames
    ):
        """
        Append new features to existing cache.
        """

        if not path.endswith(".h5"):
            path = path + ".h5"
        
        # Load existing features
        exisiting_features, existing_filenames, metadata = self.load(path)

        # Concatenate
        all_features = np.vstack([exisiting_features, new_features])
        all_filenames = existing_filenames + new_filenames

        # Update metadata
        metadata.n_samples = len(all_filenames)

        # Save combined data
        self.save(all_features, all_filenames, metadata, path)
    
    def exists(self, path):
        """
        Check if cache file exists.
        """
        if not path.endswith(".h5"):
            path = path + ".h5"
        
        return Path(path).exists()
    
    def get_feature_dict(self, path):
        """
        Load features as dictionary (for backward compatibility)
        """
        features, filenames, _ = self.load(path)
        return {fn: feat for fn, feat in zip(filenames, features)}
    
    def get_info(self, path):
        """
        Get information about the cache without loading data.
        """

        if not path.endswith(".h5"):
            path = path + ".h5"
        
        if not self.exists(path):
            return {'exists': False}
        
        with h5py.File(path, 'r') as f:
            info = {
                'exists': True,
                'n_samples': f['features'].shape[0],
                'feature_dim': f['features'].shape[1],
                'dtype': str(f['features'].dtype),
                'compression': f['features'].compression,
                'file_size_mb': Path(path).stat().st_size / (1024 * 1024)
            }

            # Add metadata
            for key, value in f.attrs.items():
                info[key] = value
            
        return info


