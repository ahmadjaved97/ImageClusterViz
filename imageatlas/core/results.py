"""
Results management and export facility.
"""

import os
import json
import numpy as np
from pathlib import Path
import warnings
from typing import Optional, Dict

class ClusteringResults:
    """
    Container for clustering results with analysis and export methods.
    """

    def __init__(
        self,
        cluster_labels,
        cluster_dict,
        filenames,
        features,
        reduced_features,
        n_clusters,
        metadata
    ):
        """
        Initialize clustering results.
        """

        self.cluster_labels = cluster_labels
        self.cluster_dict = cluster_dict
        self.filenames = filenames
        self.features = features
        self.reduced_features = reduced_features
        self.n_clusters = n_clusters or len(cluster_dict)
        self.metadata = metadata or {}


        # Initialize Export manager.
        self._exporter = ExportManager(self)

    
    def get_cluster_sizes(self):
        """
        Get the size of each cluster.
        """
        return {cluster_id: len(files) for cluster_id, files in self.cluster_dict.items()}
    
    def get_cluster(self, cluster_id):
        """
        Get filenames belonging to a specific cluster.
        """
        return self.cluster_dict.get(cluster_id, [])
    
    def get_outliers(self):
        """
        Get outlier filenames.
        """
        return self.cluster_dict.get(-1, None)
    
    def summary(self):
        """
        Get a human readable summary of results.
        """
        lines = [
            "Clustering results summary: "
            f"   Total Images: {len(self.filenames)}",
            f"   Number of clusters: {self.n_clusters}",
            f"   Cluster sizes: {self.get_cluster_sizes()}"
        ]

        outliers = self.get_outliers()
        if outliers:
            lines.append(f"   Outliers: {len(outliers)}")
        
        if self.features is not None:
            lines.append(f"   Feature dimension: {self.features.shape[1]}")
        
        if self.reduced_features is not None:
            lines.append(f"   Reduced dimension: {self.reduced_features.shape[1]}")
        
        if 'clustering_method' in self.metadata:
            lines.append(f"   Clustering method: {self.metadata['clustering_method']}")
        
        if 'model_type' in self.metadata:
            lines.append(f"   Model: {self.metadata['model_type']}")
        
        return "\n".join(lines)
    
    def to_dict(self):
        """
        Convert results to dictionary format.
        """
        return self._exporter.to_dict()
    
    def to_dataframe(self):
        """
        Convert results to pandas DataFrame.
        """
        return self._exporter.to_dataframe()
    
    def to_csv(self, path, include_features=False):
        """
        Export results to CSV.
        """
        self._exporter.to_csv(path, include_features=include_features)
    
    def to_json(self, path, include_features=False):
        """
        Export results to JSON.
        """
        self._exporter.to_json(path, include_features=include_features)
    
    def to_excel(self, path, include_features=False):
        """
        Export features to Excel file.
        """
        self._exporter.to_excel(path, include_features=include_features)
    
    def create_grids(
        self,
        image_dir: str,
        output_dir: str,
        image_size: tuple = (300, 300),
        verbose: bool = True
    ) -> Dict[int, str]:
        """
        Create grid visualizations for all clusters.
        
        Args:
            image_dir: Directory containing the original images
            output_dir: Directory to save grid images
            image_size: Size to resize each image to (width, height)
            verbose: Whether to show progress
        
        Returns:
            Dictionary mapping cluster_id to grid image path
        
        Example:
            >>> results = clusterer.fit('./images')
            >>> grid_paths = results.create_grids('./images', './grids')
            >>> print(grid_paths)
            {0: './grids/cluster_0.jpg', 1: './grids/cluster_1.jpg', ...}
        """
        from ..visualization import create_cluster_grids
        
        return create_cluster_grids(
            cluster_dict=self.cluster_dict,
            image_dir=image_dir,
            output_dir=output_dir,
            image_size=image_size,
            verbose=verbose
        )
    
    def create_cluster_folders(
        self,
        image_dir: str,
        output_dir: str,
        copy_images: bool = True,
        verbose: bool = True
    ) -> Dict[int, str]:
        """
        Organize images into cluster folders.
        
        Creates a folder for each cluster and copies/moves images into them.
        
        Args:
            image_dir: Directory containing the original images
            output_dir: Directory to create cluster folders in
            copy_images: If True, copy images; if False, move images
            verbose: Whether to show progress
        
        Returns:
            Dictionary mapping cluster_id to folder path
        
        Example:
            >>> results = clusterer.fit('./images')
            >>> folders = results.create_cluster_folders('./images', './clusters')
            >>> print(folders)
            {0: './clusters/cluster_0', 1: './clusters/cluster_1', ...}
        """
        import shutil
        from tqdm import tqdm
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        cluster_folders = {}
        
        # Progress bar
        iterator = self.cluster_dict.items()
        if verbose:
            iterator = tqdm(iterator, desc="Creating cluster folders", unit="cluster")
        
        for cluster_id, filenames in iterator:
            # Create cluster folder
            cluster_folder = os.path.join(output_dir, f'cluster_{cluster_id}')
            Path(cluster_folder).mkdir(parents=True, exist_ok=True)
            cluster_folders[cluster_id] = cluster_folder
            
            # Copy/move images
            for filename in filenames:
                #Extract basename
                basename = os.path.basename(filename)
                source_path = os.path.join(image_dir, basename)
                dest_path = os.path.join(cluster_folder, basename)
                
                try:
                    if copy_images:
                        shutil.copy2(source_path, dest_path)
                    else:
                        shutil.move(source_path, dest_path)
                except Exception as e:
                    if verbose:
                        print(f"Warning: Failed to {'copy' if copy_images else 'move'} {filename}: {e}")
        
        if verbose:
            action = "copied" if copy_images else "moved"
            print(f"\n✓ Images {action} to {len(cluster_folders)} cluster folders in {output_dir}")
        
        return cluster_folders
    

    def __repr__(self):
        """
        String representation.
        """
        return f"ClusteringResults(n_samples={len(self.filenames)}, n_clusters={self.n_clusters})"
    


    



class ExportManager:
    """
    Manages exporting clustering results to various formats.
    
    Handles CSV, JSON, Excel, DataFrame exports with optional features.
    """

    def __init__(self, results):
        """
        Initialize export manager.
        """
        self.results = results
    
    def to_dict(self, include_features=False):
        """
        Convert result to dictionary format.
        """
        data = {
            'n_samples': len(self.results.filenames),
            'n_clusters': self.results.n_clusters,
            'cluster_sizes': self.results.get_cluster_sizes(),
            'clusters': {},
            'metadata': self.results.metadata
        }

        # Add cluster assignments.
        for cluster_id, files in self.results.cluster_dict.items():
            data['clusters'][str(cluster_id)] = files
        
        # Add features if requested.
        if include_features:
            if self.results.features is not None:
                data['features'] = {
                    fn: self.results.features[i].tolist()
                    for i, fn in enumerate(self,results.filenames)
                }
            
            if self.results.reduced_features is not None:
                data['reduced_features'] = {
                    fn: self.results.reduced_features[i].tolist()
                    for i, fn in enumerate(self.results.filenames)
                }
        
        return data
    
    def to_dataframe(self, include_features=False):
        """
        Convert results to pandas dataframe.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for DataFrame support. Install with: pip install pandas")
        
        # Create base dataframe
        data = {
            'filename': self.results.filenames,
            'cluster_id': self.results.cluster_labels
        }

        df = pd.DataFrame(data)

        # Add features if requested.
        if include_features and self.results.features is not None:
            # Add features as seperate column
            n_features = self.results.features.shape[1]
            for i in range(n_features):
                df[f'feature_{i}'] = self.results.features[:, i]
        
        if include_features and self.results.reduced_features is not None:
            # Add reduced features
            n_features = self.results.reduced_features.shape[1]
            for i in range(n_features):
                df[f'reduced_features_{i}'] = self.results.reduced_features[:, i]
        
        return df
    
    def to_csv(self, path, include_features=False):
        """
        Export results to CSV format.
        """

        df = self.to_dataframe(include_features=include_features)

        # Create directory if needed
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(path, index=False)
        print(f"Results exported as CSV: {path}")
    
    def to_json(self, path, include_features=False):
        """
        Export reuslts to JSON file.
        """
        data = self.to_dict(include_features=include_features)

        # Create directory if needed
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Results exported to JSON: {path}")
    
    def to_excel(self, path, include_features=False):
        """
        Export results to excel file.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for Excel export. Install with: pip install pandas openpyxl")
        
        df = self.to_dataframe(include_features=include_features)

        # Create directory if needed
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Export to excel
        try:
            df.to_excel(path, index=False, engine='openpyxl')
            print(f"Results exported to Excel: {path}")
        except ImportError:
            raise ImportError("openpyxl is required for Excel support. Install with: pip install openpyxl")


