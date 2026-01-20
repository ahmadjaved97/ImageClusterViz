"""
Grid visualization for clustering results.
"""


import os
import math
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

class GridVisualizer:
    """
    Create grid visualizations of clustered images.
    """

    def __init__(
        self,
        image_size=(300, 300),
        space_width=10,
        space_color=(255, 255, 255),
        add_labels=True
    ):
        """
        Initialize grid visualizer.
        """

        self.image_size = image_size
        self.space_width = space_width
        self.space_color = space_color
        self.add_labels = add_labels
    
    def create_grid(
        self,
        image_list,
        num_rows=None,
        num_columns=None
    ):
        """
        Create a grid from a list of images.
        """

        if not image_list:
            # Return empty grid
            return np.full((100, 100, 3), self.space_color, dtype=np.uint8)
        
        # Calculate grid dimensions
        if num_rows is None and num_cols is None:
            num_images = len(image_list)
            num_cols = int(math.sqrt(num_images))
            num_rows = math.ceil(num_images / num_cols)
        elif num_rows is None:
            num_rows = math.ceil(len(image_list) / num_cols)
        elif num_cols is None:
            num_cols = math.ceil(len(image_list) / num_rows)
        
        # Calculate grid size
        grid_width = num_cols * (self.image_size[0] + self.space_width) - self.space_width
        grid_height = num_rows * (self.image_size[1] + self.space_width) - self.space_width
        
        # Create empty grid
        grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        grid_image[:, :] = self.space_color
        
        # Place images in grid
        for i, image in enumerate(image_list):
            # Resize image
            resized = cv2.resize(image, self.image_size)
            
            # Calculate position
            row = i // num_cols
            col = i % num_cols
            x = col * (self.image_size[0] + self.space_width)
            y = row * (self.image_size[1] + self.space_width)
            
            # Place image
            grid_image[y:y + self.image_size[1], x:x + self.image_size[0]] = resized
        
        # Add labels if requested
        if self.add_labels:
            grid_image = self._add_labels(grid_image, num_rows, num_cols)
        
        return grid_image
    
    def _add_labels(
        self,
        grid_image,
        num_rows,
        num_cols
    ):
        """
        Add row and column labels to grid.
        """

        # Add row labels
        for row in range(num_rows):
            y = row * (self.image_size[1] + self.space_width) + 20
            cv2.putText(
                grid_image,
                str(row + 1),
                (5, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )
        
        # Add column labels
        for col in range(num_cols):
            x = col * (self.image_size[0] + self.space_width) + 5
            cv2.putText(
                grid_image,
                str(col + 1),
                (x, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )
        
        return grid_image
    
    def create_cluster_grids(
        self,
        cluster_dict,
        image_dir,
        output_dir,
        verbose=True
    ):
        """
        Create grid images for all clusters.
        """

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        grid_paths = {}

        # Progress bar
        iterator = cluster_dict.items()
        if verbose:
            iterator = tqdm(iterator, desc="Creating cluster grids", unit="cluster")
        
        for cluster_id, filenames in iterator:
            # Load images for this cluster
            images = []
            for filename in filenames:
                image_path = os.path.join(image_dir, filename)

                try:
                    image = cv2.imread(image_path)
                    if image is not None:
                        images.append(image)
                
                except Exception as e:
                    if verbose:
                        print(f"Warning: Failed to load {filename}: {e}")
                    continue
            
            if not images:
                if verbose:
                    print(f"Warning: No valid images in cluster {cluster_id}")
                continue
            
            # Create grid
            grid = self.create_grid(images)
            
            # Save grid
            output_path = os.path.join(output_dir, f'cluster_{cluster_id}.jpg')
            cv2.imwrite(output_path, grid)
            grid_paths[cluster_id] = output_path
        
        if verbose:
            print(f"\n Created {len(grid_paths)} cluster grids in {output_dir}")
        
        return grid_paths


def create_cluster_grids(
    cluster_dict,
    image_dir,
    output_dir,
    image_size,
    verbose
):
    """
    Convenience function to create cluster grid.
    """

    visualizer = GridVisualizer(image_size=image_size)
    return visualizer.create_cluster_grids(
        cluster_dict=cluster_dict,
        image_dir=image_dir,
        output_dir=output_dir,
        verbose=verbose
    )
