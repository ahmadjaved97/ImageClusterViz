"""
Result container and export functionality for duplicate detection.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings

class DuplicateResults:
    """
    Container with duplicate detection results with export capabilities.
    """

    def __init__(
        self,
        pairs,
        groups,
        filenames,
        signatures=None,
        best_images=None,
        metadata=None
    ):
        """
        Initialize duplicate results.
        """

        self.pairs = pairs
        self.groups = groups
        self.filenames = filenames
        self.signatures = signatures
        self.best_images = best_images or list(groups.keys())
        self.metadata = metadata or {}

        # Store timestamp
        if 'timestamp' not in self.metadata:
            self.metadata['timestamp'] = datetime.now().isoformat()
    
    def get_statistics(self):
        """
        Compute statistics about duplicate detection results.
        """

        total_images = len(self.filenames)
        images_with_duplicates = len(self.pairs)
        num_groups = len(self.groups)

        # Count total duplicates
        total_duplicates = sum(len(dups) for dups in self.groups.values())

        # Reduction percentage
        if total_images > 0:
            reduction_pct = (total_duplicates / total_images) * 100
        
        else:
            reduction_pct = 0.0
        
        # Group size statistics
        if self.groups:
            group_sizes = [len(dups) + 1 for dups in self.groups.values()]  # +1 for representative
            avg_group_size = np.mean(group_sizes)
            max_group_size = np.max(group_sizes)
            min_group_size = np.min(group_sizes)
        else:   # (REVISIT)
            avg_group_size = 0
            max_group_size = 0
            min_group_size = 0
        
        # Similarity statistics (from pairs)

        if self.pairs:
            all_scores = []
            for duplicates in self.paris.values():
                all_scores.extend([score for _, score in duplicates])
            
            if all_scores:
                avg_similarity = np.mean(all_scores)
                max_similarity = np.max(all_scores)
                min_similarity = np.min(all_scores)
            else:
                avg_similarity = 0
                max_similarity = 0
                min_similarity = 0
        else:
            avg_similarity = 0
            max_similarity = 0
            min_similarity = 0
        

        return {
            'total_images': total_images,
            'images_with_duplicates': images_with_duplicates,
            'num_duplicate_groups': num_groups,
            'total_duplicates': total_duplicates,
            'reduction_percentage': reduction_pct,
            'avg_group_size': avg_group_size,
            'max_group_size': max_group_size,
            'min_group_size': min_group_size,
            'avg_similarity': avg_similarity,
            'max_similarity': max_similarity,
            'min_similarity': min_similarity,
        }
    
    def summary(self):
        """
        Get human readable summary of results.
        """

        stats = self.get_statistics()
        
        lines = [
            "Duplicate Detection Results",
            "=" * 50,
            f"Total images analyzed: {stats['total_images']}",
            f"Images with duplicates: {stats['images_with_duplicates']}",
            f"Duplicate groups found: {stats['num_duplicate_groups']}",
            f"Total duplicates (removable): {stats['total_duplicates']}",
            f"Dataset reduction: {stats['reduction_percentage']:.1f}%",
            "",
            "Group Statistics:",
            f"  Average group size: {stats['avg_group_size']:.1f}",
            f"  Largest group: {stats['max_group_size']} images",
            f"  Smallest group: {stats['min_group_size']} images",
            "",
            "Similarity Statistics:",
            f"  Average similarity: {stats['avg_similarity']:.3f}",
            f"  Max similarity: {stats['max_similarity']:.3f}",
            f"  Min similarity: {stats['min_similarity']:.3f}",
        ]

        # Add method info if available
        if 'method' in self.metadata:
            lines.insert(2, f"Detection method: {self.metadata['method']}")
        
        if 'threshold' in self.metadata:
            lines.insert(3, f"Threshold used: {self.metadata['threshold']:.3f}")
        
        return "\n".join(lines)
    
    def get_best_images(self):
        """
        Get list of best images (representatives to keep)
        """

        return self.best_images.copy()
    
    def get_images_to_remove(self):
        """
        Get list of duplicate images that can be removed.
        """
        to_remove = []
        for duplicates in self.groups.values():
            to_remove.extend(duplicates)
        
        return to_remove
    
    def get_group(self, image_path):
        """
        Get the duplicate group containing a specific image.
        """

        # Check if it's a representative
        if image_path in self.groups:
            return [image_path] + self.groups[image_path]
        
        # Check if it's in any group
        for rep, dups in self.groups.items():
            if image_path in dups:
                return [rep] + dups
        
        return None

    def to_dict(self, include_signatures=False):
        """
        Convert results to dictionary format.
        """

        data = {
            'pairs': self.paris,
            'groups': self.groups,
            'filenames': self.filenames,
            'best_images': self.best_images,
            'metadata': self.metadata,
            'statistics': self.get_statistics()
        }

        if include_signatures and self.signatures is not None:
            data['signatures'] = self.signatures.tolist()
        
        return data
    
    