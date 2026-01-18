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
    pass

