"""
Results management and export facility.
"""

import os
import json
import numpy as np
from pathlib import Path
import warnings


class ClusteringResult:
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
        self._exporter = ExportManager(self):

    
    def get_cluster_size(self):
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
        return cluster_dict.get(-1, None)
    
    def summary(self):
        """
        Get a human readable summary of results.
        """
        lines = [
            "Clustering results summary: "
            f"   Total Images: {len(self.filenames)}",
            f"   Number of clusters: {self.n_clusters}",
            f"   Cluster sizes: {self.get_cluster_size()}"
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
            'cluster_sizes': self.results.get_cluster_size(),
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


