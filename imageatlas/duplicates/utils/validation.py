"""
Input validation utilities.
"""

import os
from pathlib import Path

def validate_image_paths(
    paths,
    check_exists=True
):
    """
    Validate and normalize image paths.
    """

    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

    # Handle single string
    if isinstance(paths, str):
        path = Path(paths)

        # If directory, find all images.
        if path.is_dir():
            image_paths = []
            for ext in VALID_EXTENSIONS:
                image_paths.extend(path.glob(f'*{ext}'))
                image_paths.extend(path.glob(f'*{ext.upper()}'))
            
            if not image_paths:
                raise ValueError(f"No images found in directory: {paths}")
            
            return [str(p) for p in sorted(image_paths)]
        
        # Single file
        elif path.is_file():
            if path.suffix.lower() not in VALID_EXTENSIONS:
                raise ValueError(f"Invalid image format: {path.suffix}")
            return [str(path)]
        
        else:
            raise ValueError(f"Path does not exist: {paths}")
    
    # Handle list of paths
    if isinstance(paths, list):
        if not paths:
            raise ValueError(f"Empty path list provided")
        
        validated_paths = []
        for p in paths:
            p = Path(p)

            if check_exists and not p.exists():
                raise ValueError(f"File does not exist: {p}")
            
            if p.suffix.lower() not in VALID_EXTENSIONS:
                raise ValueError(f"Invalid image format: {p.suffix}")
            
            validated_paths.append(str(p))
        
        return validated_paths
    
    else:
        raise TypeError(f"Invalid type for paths: {type(paths)}")


def validate_threshold(
    threshold,
    method,
    allow_none=True
):
    """
    Validate threshold value for a given method.
    """

    if threshold is None:
        if not allow_none:
            raise ValueError("Threshold cannot be None")
        return
    
    if not isinstance(threshold, (int, float)):
        raise TypeError(f"Threshold must be numeric, got {type(threshold)}")
    
    # Method-specific validation.
    if method in ['phash', 'dhash', 'ahash', 'whash']:
        # Hamming distance: 0-64 typically
        if not 0 <= threshold <= 64:
            raise ValueError(f"Hash threshold should be 0-64, got {threshold}")
    
    elif method in ['embedding', 'clip']:
        # Cosine/Euclidean similarity: typically 0-1
        if not 0 <= threshold <= 1:
            raise ValueError(f"Embedding threshold should be 0-1, got {threshold}")
    
    elif method == 'crypto_hash':
        # Binary match only
        if threshold not in [0, 1]:
            raise ValueError("Cryptographic hash only supports threshold 0 or 1")


def validate_detector_params(params):
    """
    Validate duplicate detector parameters.
    """

    method = params.get('method')
    valid_methods = ['crypto_hash', 'phash', 'dhash', 'ahash', 'whash', 'embedding', 'clip']

    if method not in valid_methods:
        raise ValueError(
            f"Invalid method: {method}. ",
            f"Valid methods: {', '.join(valid_methods)}"
        )
    
    # Validate threshold
    threshold = params.get('threshold')
    adaptive_percentile = params.get('adaptive_percentile')

    if threshold is None and adaptive_percentile is None:
        raise ValueError("Must provide either threshold or adaptive percentile")
    
    if threshold is not None and adaptive_percentile is not None:
        raise ValueError("Cannot provide both threshold and adaptive percentile")
    
    if adaptive_percentile is not None:
        if not isinstance(adaptive_percentile, (int, float)):
            raise TypeError("adaptive_percentile must be numeric")
        if not 0 < adaptive_percentile < 100:
            raise ValueError("adaptive_percentile must be between 0 and 100")
    

    # Validate grouping
    grouping = params.get('grouping')
    if not isinstance(grouping, bool):
        raise TypeError("grouping must be boolean")
    
    # Validate best selection
    best_selection = params.get('best_selection')
    valid_selections = ['resolution', 'alphabetic', 'both']
    if best_selection not in valid_selections:
        raise ValueError(
            f"Invalid Best selection: {best_selection}. "
            f"Valid options are: {', '.join(valid_selections)}"
        )
    
    # Validate batch size
    batch_size = params.get('batch_size')
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")


def validate_similarity_metric(
    metric,
    valid_metrics=None
):
    """
    Validate similarity metric.
    """

    if valid_metrics is None:
        valid_metrics = ['cosine', 'euclidean']
    
    if metric not in valid_metrics:
        raise ValueError(
            f"Invalid metric: {metric}",
            f"Valid metric: {', '.join(valid_metrics)}"
        )
