"""
Caching system for duplicate detection.
"""

import h5py
import json
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
import hashlib

class DuplicateCache:
    """
    HDF5 based caching for duplicate detection signatures and results.
    """

    def __init__(self, cache_path):
        """
        Initialize cache.
        """
        self.cache_path = Path(cache_path)

        # Make sure .h5 extension
        if self.cache_path.suffix not in ['.h5', '.hdf5']:
            self.cache_path = self.cache_path.with_suffix('.h5')
        
        # Create directory if needed
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _compute_cache_key(
        self,
        method,
        filenames
    ):
        """
        Compute cache key from method and filenames.
        """

        # Sort filenames for consistency
        sorted_filenames = sorted(filenames)

        # Create hash of filenames
        filenames_str = "|".join(sorted_filenames)
        filenames_hash = hashlib.md5(filenames_str.encode()).hexdigest()[:16]

        # Combine method and hash
        cache_key = f"{method}_{filenames_hash}"

        return cache_key
    
    def save_signatures(
        self,
        signatures,
        filenames,
        method,
        metadata=None
    ):
        """
        Save signatures to cache.
        """
        cache_key = self._compute_cache_key(method, filenames)
        
        with h5py.File(self.cache_path, 'a') as f:
            # Create group for this cache entry
            if cache_key in f:
                del f[cache_key]  # Overwrite existing
            
            group = f.create_group(cache_key)
            
            # Save signatures
            group.create_dataset(
                'signatures',
                data=signatures,
                compression='gzip',
                compression_opts=4
            )
            
            # Save filenames
            dt = h5py.special_dtype(vlen=bytes)
            filenames_bytes = [fn.encode('utf-8') for fn in filenames]
            group.create_dataset('filenames', data=filenames_bytes, dtype=dt)
            
            # Save metadata
            group.attrs['method'] = method
            if metadata:
                group.attrs['metadata'] = json.dumps(metadata)
    
    def load_signatures(
        self,
        method: str,
        filenames: List[str]
    ) -> Optional[np.ndarray]:
        """
        Load signatures from cache.
        
        Args:
            method: Detection method name
            filenames: Image filenames
        
        Returns:
            Cached signatures if available, None otherwise
        """
        if not self.cache_path.exists():
            return None
        
        cache_key = self._compute_cache_key(method, filenames)
        
        try:
            with h5py.File(self.cache_path, 'r') as f:
                if cache_key not in f:
                    return None
                
                group = f[cache_key]
                
                # Verify filenames match
                cached_filenames = [
                    s.decode('utf-8') if isinstance(s, bytes) else s
                    for s in group['filenames'][:]
                ]
                
                if sorted(cached_filenames) != sorted(filenames):
                    return None  # Mismatch
                
                # Load signatures
                signatures = group['signatures'][:]
                
                return signatures
        
        except Exception as e:
            # If any error, return None
            return None
    
    def exists(self, method: str, filenames: List[str]) -> bool:
        """
        Check if signatures are cached.
        
        Args:
            method: Detection method name
            filenames: Image filenames
        
        Returns:
            True if cached, False otherwise
        """
        if not self.cache_path.exists():
            return False
        
        cache_key = self._compute_cache_key(method, filenames)
        
        try:
            with h5py.File(self.cache_path, 'r') as f:
                return cache_key in f
        except Exception:
            return False
    
    def clear(self) -> None:
        """
        Clear all cache entries.
        """
        if self.cache_path.exists():
            self.cache_path.unlink()
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about cache.
        
        Returns:
            Dictionary with cache information
        """
        if not self.cache_path.exists():
            return {
                'exists': False,
                'path': str(self.cache_path)
            }
        
        try:
            with h5py.File(self.cache_path, 'r') as f:
                cache_entries = list(f.keys())
                
                total_size = 0
                for entry in cache_entries:
                    if 'signatures' in f[entry]:
                        total_size += f[entry]['signatures'].size
                
                file_size_mb = self.cache_path.stat().st_size / (1024 * 1024)
                
                return {
                    'exists': True,
                    'path': str(self.cache_path),
                    'num_entries': len(cache_entries),
                    'entries': cache_entries,
                    'file_size_mb': file_size_mb,
                    'total_signatures': total_size
                }
        
        except Exception as e:
            return {
                'exists': True,
                'path': str(self.cache_path),
                'error': str(e)
            }
    
    def remove_entry(self, method: str, filenames: List[str]) -> bool:
        """
        Remove a specific cache entry.
        
        Args:
            method: Detection method name
            filenames: Image filenames
        
        Returns:
            True if removed, False if not found
        """
        if not self.cache_path.exists():
            return False
        
        cache_key = self._compute_cache_key(method, filenames)
        
        try:
            with h5py.File(self.cache_path, 'a') as f:
                if cache_key in f:
                    del f[cache_key]
                    return True
                return False
        except Exception:
            return False