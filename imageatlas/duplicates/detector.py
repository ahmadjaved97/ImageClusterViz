"""
Main API for duplicate detection.
"""


import numpy as np
from typing import List, Union, Optional, Literal
from pathlib import Path

from .base import DuplicateDetectionStrategy, GroupingAlgorithm, BestImageSelector
from .strategies import (
    CryptographicHashStrategy,
    PerceptualHashStrategy,
    EmbeddingStrategy,
    CLIPStrategy
)

from .grouping import (
    GroupBuilder,
    PairwiseGrouping,
    FixedThreshold,
    AdaptivePercentileThreshold
)

from .selection import create_best_selector
from .results import DuplicateResults
# from .cache import DuplicateCache
from .utils import (
    validate_image_paths,
    validate_detector_params,
    pairwise_similarity,
    find_pairs_above_threshold,
    compute_similarity_statistics,
    ProgressTracker
)

class DuplicateDetector:
    """
    Main API for duplicate image detection.

    Supports multiple detection methods: hash-based, embedding based, CLIP.
    Provides flexible configuration for threshold, grouping and selection.
    """

    def __init__(
        self,
        method: Literal['crypto_hash', 'phash', 'dhash', 'ahash', 'whash', 'embedding', 'clip'] = 'phash',
        model = None,
        variant = None,
        hash_algorithm = 'md5',
        similarity_metric = 'cosine',
        threshold = None,
        adaptive_percentile = None,
        grouping = True,
        best_selection = 'resolution',
        device = 'auto',
        batch_size = 32,
        use_cache = False,
        cache_path = None,
        verbose = True
    ):
        """
        Initialize duplicate detector.
        """

        # Validate parameters
        params = {
            'method': method,
            'threshold': threshold,
            'adaptive_percentile': adaptive_percentile,
            'grouping': grouping,
            'best_selection': best_selection,
            'device': device,
            'batch_size': batch_size
        }

        validate_detector_params(params)

        # Store configuration
        self.method = method
        self.model = model
        self.variant = variant
        self.hash_algorithm = hash_algorithm
        self.similarity_metric = similarity_metric
        self.threshold = threshold
        self.adaptive_percentile = adaptive_percentile
        self.grouping = grouping
        self.best_selection = best_selection
        self.device = self._validate_device(device)    # REVISIT (create a common function to validate device across features)
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.cache_path = cache_path
        self.verbose = verbose

        # Initialize components
        self._strategy = None
        self._grouping_algo = None
        self._best_selector = None
        self._cache = None

        # Results
        self.results_ = None
        self.is_fitted_ = False

    
    def _validate_device(self, device):
        """
        Validate and normalize device string.
        """
        if device == 'auto':
            import torch
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def _get_strategy(self):
        """
        Get or create detection strategy.
        """
        print(self.method)
        if self._strategy is None:
            if self.method == 'crypto_hash':
                self._strategy = CryptographicHashStrategy(
                    algorithm=self.hash_algorithm
                )
            elif self.method in ['phash', 'dhash', 'ahash', 'whash']:
                self._strategy = PerceptualHashStrategy(
                    algorithm=self.method,
                    hash_size=8
                )
            elif self.method in 'embedding':
                if self.model is None:
                    raise ValueError("Must specify 'model' for embedding method.")
                
                self._strategy = EmbeddingStrategy(
                    model = self.model,
                    variant=self.variant,
                    similarity_metric=self.similarity_metric,
                    device=self.device
                )
            
            elif self.method == 'clip':
                variant = self.variant or 'ViT/B-16'
                self._strategy = CLIPStrategy(
                    variant=variant,
                    device=self.device
                )
            else:
                raise ValueError(f"Unknown method: {self.method}")
        
        return self._strategy
        
    
    def _get_grouping_algo(self):
        """
        Get or create grouping algorithm.
        """
        if not self.grouping:
            return PairwiseGrouping()
        
        if self._grouping_algo is None:
            self._grouping_algo = GroupBuilder()
        
        return self._grouping_algo
    
    def _get_best_selector(self):
        """
        Get or create best image selector.
        """
        if self._best_selector is None:
            self._best_selector = create_best_selector(self.best_selection)
        
        return self._best_selector
    
    def _get_cache(self):
        """
        Get or create cache.
        """
        if self.use_cache and self.cache_path:
            if self._cache is None:
                self._cache = DuplicateCache(self.cache_path)
            return self._cache
        
        return None
    
    def detect(
        self,
        image_paths,
    ):
        """
        Detect duplicates in images.
        """

        if self.verbose:
            print("="*60)
            print("DUPLICATE DETECTION")
            print("="*60)

        
        # Step 1: Validate and collect image paths
        if self.verbose:
            print("\nStep1: Validating image paths...")
        
        image_paths = validate_image_paths(image_paths)

        if self.verbose:
            print(f"   Found {len(image_paths)} images")
        
        # Step 2: Compute signatures
        if self.verbose:
            print(f"\nStep 2: Computing signatures ({self.method})...")
        
        cache = self._get_cache()
        strategy = self._get_strategy()
        print(strategy)


        # Try to load from cache
        signatures = None
        if cache is not None:
            signatures = cache.load_signatures(
                method=strategy.get_method_name(),
                filenames=image_paths
            )
        
        # Compute if not cached
        if signatures is None:
            signatures = strategy.compute_signatures(
                image_paths,
                batch_size=self.batch_size,
                verbose=self.verbose
            )

            # Save to cache
            if cache is not None:
                cache.save_signatures(
                    signatures=signatures,
                    filenames=image_paths,
                    method=strategy.get_method_name()
                )
        
        if self.verbose:
            print(f"   Computed signatures: {signatures.shape}")
        
        # Step 3: compute pairsise similarities.
        if self.verbose:
            print("\nStep 3: Computing pairwise similarities...")
        
        # For hash methods, use special handling
        if self.method in ['crypto_hash', 'phash', 'dhash', 'ahash', 'whash']:
            similarity_matrix = self._compute_hash_similarities(
                signatures,
                strategy,
                image_paths
            )
        else:
            # For embeddings usse matrix computation
            similarity_matrix = pairwise_similarity(
                signatures,
                metric=self.similarity_metric,
                batch_size=1000
            )
        if self.verbose:
            stats = compute_similarity_statistics(similarity_matrix)
            print(f"   Similarity range: [{stats['min']:.3f}, {stats['max']:.3f}]")
            print(f" Mean similarity: {stats['mean']:.3f}")
        
        # Step 4: Determine threshold
        if self.verbose:
            print("\nStep 4: Determining threshold...")
        
        if self.threshold is not None:
            # Fixed threshold
            threshold_selector = FixedThreshold()
            actual_threshold = threshold_selector.select_threshold(
                similarity_matrix.flatten(),
                user_threshold=self.threshold
            )
        else:
            # Adaptive threshold
            threshold_selector = AdaptivePercentileThreshold(
                percentile=self.adaptive_percentile
            )
            actual_threshold = threshold_selector.select_threshold(
                similarity_matrix.flatten()
            )
        
        if self.verbose:
            print(f"   Threshold: {actual_threshold:.3f}")
        
        # Step 5: find pairs above threshold
        if self.verbose:
            print("\nStep 5: Finding duplicate pairs")
        
        pairs = find_pairs_above_threshold(
            similarity_matrix,
            actual_threshold,
            image_paths
        )

        if self.verbose:
            print(f"   Found {len(pairs)} images with duplicates")
        
        # Step 6: Group duplicates
        if self.verbose:
            print(f"\nStep 6: Grouping duplicates...")
        
        grouping_algo = self._get_grouping_algo()
        groups = grouping_algo.group_duplicates(pairs, image_paths)

        if self.verbose:
            if self.grouping:
                print(f"   Created {len(groups)} duplicate groups")
            else:
                print(f"   Pairwise mode (no grouping)")
        
        # Step 7: Select best images
        if self.verbose:
            print(f"\nStep 7: Selecting best images")
        
        best_selector = self._get_best_selector()
        best_images = []

        for representative, duplicates in groups.items():
            # All images in group
            group_images = [representative] + duplicates

            # Select best
            best = best_selector.select_best(group_images)
            best_images.append(best)
        
        if self.verbose:
            print(f"   Selected {len(best_images)} best images")
        
        # Step 8: Create results
        metadata = {
            'method': strategy.get_method_name(),
            'threshold': actual_threshold,
            'adaptive': self.adaptive_percentile is not None,
            'grouping': self.grouping,
            'best_selection': self.best_selection,
            'device': self.device,
            'batch_size': self.batch_size
        }

        if self.model:
            metadata['model'] = self.model
        if self.variant:
            metadata['variant'] = self.variant
        
        self.results_ = DuplicateResults(
            pairs=pairs,
            groups=groups,
            filenames=image_paths,
            signatures=signatures,
            best_images=best_images,
            metadata=metadata
        )

        self.is_fitted_ = True

        if self.verbose:
            print("\n" + "=" * 60)
            print("DETECTION COMPLETE")
            print("=" * 60)
            print(f"\n{self.results_.summary()}")
        
        return self.results_
    
    def _compute_hash_similarities(
        self,
        signatures,
        strategy,
        filenames
    ):
        """
        Compute similarity matrix for hash-based methods.
        """

        n = len(signatures)
        similarity_matrix = np.zeros((n, n), dtype=np.float32)

        # Set diagonal to 1 (self-similarity)
        np.fill_diagonal(similarity_matrix, 1.0)

        with ProgressTracker(
            n * (n - 1) // 2,
            desc="Computing hash similarities",
            disable=not self.verbose
        ) as progress:

            # Compute upper triangle
            for i in range(n):
                for j in range(i + 1, n):
                    sim = strategy.compute_similarity(signatures[i], signatures[j])
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim  # Similarity
                    progress.update(1)
        
        return similarity_matrix
    
    def detect_from_embeddings(
        self,
        embeddings,
        filenames
    ):
        """
        Detect duplicates from pre-computed embeddings.
        """

        if self.method in ['crypto_hash', 'phash', 'dhash', 'ahash', 'whash']:
            raise ValueError("detect_from_embeddings only works with embedding-based methods")
        
        if len(embeddings) != len(filenames):
            raise ValueError("Number of embeddings must match number of filenames")
        
        if self.verbose:
            print("=" * 60)
            print("DUPLICATE DETECTION (FROM EMBEDDINGS)")
            print("=" * 60)
            print(f"\nInput: {len(filenames)} images, {embeddings.shape[1]}D embeddings")
        
        # Use embeddings as signatures
        signatures = embeddings

        # Compute similarities
        if self.verbose:
            print("\nComputing pairwise similarities...")
        
        similarity_matrix = pairwise_similarity(
            signatures,
            metric=self.similarity_metric,
            batch_size=1000
        )
        
        # Determine threshold
        if self.threshold is not None:
            threshold_selector = FixedThreshold()
            actual_threshold = threshold_selector.select_threshold(
                similarity_matrix.flatten(),
                user_threshold=self.threshold
            )
        else:
            threshold_selector = AdaptivePercentileThreshold(
                percentile=self.adaptive_percentile
            )
            actual_threshold = threshold_selector.select_threshold(
                similarity_matrix.flatten()
            )
        
        # Find pairs
        pairs = find_pairs_above_threshold(
            similarity_matrix,
            actual_threshold,
            filenames
        )
        
        # Group
        grouping_algo = self._get_grouping_algo()
        groups = grouping_algo.group_duplicates(pairs, filenames)
        
        # Select best
        best_selector = self._get_best_selector()
        best_images = []
        
        for representative, duplicates in groups.items():
            group_images = [representative] + duplicates
            best = best_selector.select_best(group_images)
            best_images.append(best)
        
        # Create results
        metadata = {
            'method': 'pre_computed_embeddings',
            'threshold': actual_threshold,
            'adaptive': self.adaptive_percentile is not None,
            'grouping': self.grouping,
            'best_selection': self.best_selection
        }
        
        self.results_ = DuplicateResults(
            pairs=pairs,
            groups=groups,
            filenames=filenames,
            signatures=signatures,
            best_images=best_images,
            metadata=metadata
        )
        
        self.is_fitted_ = True
        
        if self.verbose:
            print(f"\n{self.results_.summary()}")
        
        return self.results_
    

    def get_results(self):
        """
        Get direction results.
        """

        if not self.is_fitted_:
            raise RuntimeError("No results available. Run detect() first.")
        
        return self.results_
    
    def __repr__(self):
        """
        String representation
        """
        status = "fitted" if self.is_fitted_ else "not fitted"
        return (f"DuplicateDetector(method='{self.method}', "
                f"threshold={self.threshold}, "
                f"status={status})")


