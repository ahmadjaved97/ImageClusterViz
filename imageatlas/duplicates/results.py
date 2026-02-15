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
    
    
    def to_json(self, path, include_signatures=False):
        """
        Export results to JSON File.
        """

        data = self.to_dict(include_signatures=include_signatures)

        # Create directory if needed
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Results exported to JSON: {path}")
    
    def to_csv(self, path):
        """
        Export results to CSV file.
        """

        try:
            import pandas as pd
        except Exception as e:
            raise ImportError("pandas is required for CSV export. Install with: pip install pandas")
        
        # Build rows

        rows = []
        group_id = 0
        group_mapping = {}

        for rep, dups in self.groups.items():
            group_mapping[rep] = group_id

            # Add pairs for this group
            all_in_group = [rep] + dups

            # Generate all pairs within group
            for i, img1 in enumerate(all_in_group):
                for img2 in all_in_group[i+1:]:
                    # Find similarity score
                    score = None
                    if img1 in self.pairs and any(d[0] == img2 for d in self.pairs[img1]):
                        score = next(d[1] for d in self.pairs[img1] if d[0] == img2)
                    elif img2 in self.pairs and any(d[0] == img1 for d in self.pairs[img2]):
                        score = next(d[1] for d in self.pairs[img2] if d[0] == img1)
                    
                    rows.append({
                        'image_1': img1,
                        'image_2': img2,
                        'similarity_score': score if score is not None else '',
                        'method': self.metadata.get('method', ''),
                        'threshold_used': self.metadata.get('threshold', ''),
                        'is_best_image_1': img1 == rep,
                        'is_best_image_2': img2 == rep,
                        'group_id': group_id
                    })
            
            group_id += 1

        # Create DataFrame
        df = pd.DataFrame(rows)

        # Create directory if needed
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Save to CSV
        df.to_csv(path, index=False)

        print(f"Results exported to CSV: {path}")
    
    def to_excel(self, path):
        """
        Export results to Excel file with multiple sheets.

        Creates sheets: Pairs, Groups, Summary
        """
        
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas and openpyxl are required for Excel export")
        
        # Create directory if needed
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            # Sheet 1: Pairs
            pairs_rows = []
            for img1, duplicates in self.pairs.items():
                for img2, score in duplicates:
                    pairs_rows.append({
                        'Image 1': img1,
                        'Image 2': img2,
                        'Similarity': score
                    })
            
            if pairs_rows:
                df_pairs = pd.DataFrame(pairs_rows)
                df_pairs.to_excel(writer, sheet_name='Pairs', index=False)
            
            # Sheet 2: Groups
            groups_rows = []
            for rep, dups in self.groups.items():
                groups_rows.append({
                    'Representative (Keep)': rep,
                    'Duplicates (Remove)': ', '.join(dups),
                    'Group Size': len(dups) + 1
                })
            
            if groups_rows:
                df_groups = pd.DataFrame(groups_rows)
                df_groups.to_excel(writer, sheet_name='Groups', index=False)
            
            # Sheet 3: Summary
            stats = self.get_statistics()
            summary_rows = [
                {'Metric': k, 'Value': v}
                for k, v in stats.items()
            ]
            
            # Add metadata
            for k, v in self.metadata.items():
                summary_rows.append({'Metric': k, 'Value': str(v)})
            
            df_summary = pd.DataFrame(summary_rows)
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            
            # Sheet 4: Best Images (to keep)
            best_rows = [{'Best Images': img} for img in self.best_images]
            df_best = pd.DataFrame(best_rows)
            df_best.to_excel(writer, sheet_name='Best Images', index=False)
        
        print(f"Results exported to Excel: {path}")
    
    def to_dataframe(self):
        """
        Convert results to pandas dataframe.
        """

        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required. Install with: pip install pandas")
        
        rows = []
        for img1, duplicates in self.pairs.items():
            for img2, score in duplicates:
                rows.append({
                    'image_1': img1,
                    'image_2': img2,
                    'similarity': score,
                    'is_best_1': img1 in self.best_images,
                    'is_best_2': img2 in self.best_images
                })
        
        return pd.DataFrame(rows)
    
    @classmethod
    def load(cls, path):
        """
        Load results from saved file.
        """

        path = Path(path)

        if path.suffix == '.json':
            return cls._load_json(path)
        elif path.suffix in ['.h5', '.hdf5']:
            return cls._load_hdf5(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        # REVISIT: add options for loading dict, csv, excel
    
    @classmethod
    def _load_json(cls, path: Path) -> 'DuplicateResults':
        """Load from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Convert pairs back to proper format (tuples)
        pairs = {}
        for img, dups in data['pairs'].items():
            pairs[img] = [(d[0], d[1]) for d in dups]
        
        # Convert signatures if present
        signatures = None
        if 'signatures' in data:
            signatures = np.array(data['signatures'])
        
        return cls(
            pairs=pairs,
            groups=data['groups'],
            filenames=data['filenames'],
            signatures=signatures,
            best_images=data.get('best_images'),
            metadata=data.get('metadata', {})
        )
    
    @classmethod
    def _load_hdf5(cls, path: Path) -> 'DuplicateResults':
        """Load from HDF5 file."""
        import h5py
        
        with h5py.File(path, 'r') as f:
            # Load pairs (stored as JSON string in attribute)
            pairs_json = f.attrs.get('pairs', '{}')
            pairs_dict = json.loads(pairs_json)
            pairs = {img: [(d[0], d[1]) for d in dups] 
                    for img, dups in pairs_dict.items()}
            
            # Load groups
            groups_json = f.attrs.get('groups', '{}')
            groups = json.loads(groups_json)
            
            # Load filenames
            filenames = [s.decode('utf-8') if isinstance(s, bytes) else s 
                        for s in f['filenames'][:]]
            
            # Load signatures if present
            signatures = None
            if 'signatures' in f:
                signatures = f['signatures'][:]
            
            # Load metadata
            metadata_json = f.attrs.get('metadata', '{}')
            metadata = json.loads(metadata_json)
            
            # Load best images
            best_images_json = f.attrs.get('best_images', '[]')
            best_images = json.loads(best_images_json)
        
        return cls(
            pairs=pairs,
            groups=groups,
            filenames=filenames,
            signatures=signatures,
            best_images=best_images,
            metadata=metadata
        )
    
    def save(self, path: str) -> None:
        """
        Save results to file.
        
        Args:
            path: Output path (.json or .h5)
        """
        path = Path(path)
        
        if path.suffix == '.json':
            self.to_json(str(path), include_signatures=True)
        elif path.suffix in ['.h5', '.hdf5']:
            self._save_hdf5(path)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}. Use .json or .h5")
    
    def _save_hdf5(self, path: Path) -> None:
        """Save to HDF5 file."""
        import h5py
        
        # Create directory if needed
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(path, 'w') as f:
            # Save pairs as JSON in attributes
            f.attrs['pairs'] = json.dumps(self.pairs)
            
            # Save groups as JSON
            f.attrs['groups'] = json.dumps(self.groups)
            
            # Save filenames
            dt = h5py.special_dtype(vlen=bytes)
            filenames_bytes = [fn.encode('utf-8') for fn in self.filenames]
            f.create_dataset('filenames', data=filenames_bytes, dtype=dt)
            
            # Save signatures if present
            if self.signatures is not None:
                f.create_dataset(
                    'signatures',
                    data=self.signatures,
                    compression='gzip',
                    compression_opts=4
                )
            
            # Save metadata
            f.attrs['metadata'] = json.dumps(self.metadata)
            
            # Save best images
            f.attrs['best_images'] = json.dumps(self.best_images)
        
        print(f"Results saved to HDF5: {path}")
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"DuplicateResults(images={len(self.filenames)}, "
                f"groups={len(self.groups)}, "
                f"duplicates={sum(len(d) for d in self.groups.values())})")

