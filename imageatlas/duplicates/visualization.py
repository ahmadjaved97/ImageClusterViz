"""
Visualization for duplicate detection results.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm


class DuplicateGridVisualizer:
    """
    Creates grid visualizations for duplicate groups.
    
    Similar to grid visualizer in clustering
    In future combine both (REVISIT)
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (300, 300),
        space_width: int = 10,
        space_color: Tuple[int, int, int] = (255, 255, 255),
        font_scale: float = 1.0,
        font_thickness: int = 2
    ):
        """
        Initialize visualizer.
        """
        self.image_size = image_size
        self.space_width = space_width
        self.space_color = space_color
        self.font_scale = font_scale
        self.font_thickness = font_thickness
    
    def add_text_overlay(
        self,
        image: np.ndarray,
        text: str,
        position: str = 'top'
    ):
        """
        Add text overlay to image.
        """
        img_copy = image.copy()
        
        # Calculate text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            self.font_thickness
        )
        
        # Position
        if position == 'top':
            x = 10
            y = 30
        else:  # bottom
            x = 10
            y = image.shape[0] - 10
        
        # Add semi-transparent background
        overlay = img_copy.copy()
        cv2.rectangle(
            overlay,
            (x - 5, y - text_height - 5),
            (x + text_width + 5, y + baseline + 5),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.6, img_copy, 0.4, 0, img_copy)
        
        # Add text
        cv2.putText(
            img_copy,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            (255, 255, 255),
            self.font_thickness
        )
        
        return img_copy
    
    def add_best_indicator(
        self,
        image: np.ndarray,
        border_color: Tuple[int, int, int] = (0, 255, 0),
        border_thickness: int = 10
    ):
        """
        Add border to indicate best image.
        """
        img_copy = image.copy()
        
        # Draw border
        cv2.rectangle(
            img_copy,
            (0, 0),
            (image.shape[1] - 1, image.shape[0] - 1),
            border_color,
            border_thickness
        )
        
        # Add "BEST" text
        img_copy = self.add_text_overlay(img_copy, "★ BEST", position='top')
        
        return img_copy
    
    def create_duplicate_grid(
        self,
        image_dir: str,
        representative: str,
        duplicates: List[str],
        scores: Optional[Dict[str, float]] = None,
        top_n: Optional[int] = None
    ) -> np.ndarray:
        """
        Create grid for one duplicate group.
        """
        # Limit to top_n if specified
        if top_n is not None and len(duplicates) > top_n:
            # Sort by score if available
            if scores:
                sorted_dups = sorted(
                    duplicates,
                    key=lambda x: scores.get(x, 0),
                    reverse=True
                )
                duplicates = sorted_dups[:top_n]
            else:
                duplicates = duplicates[:top_n]
        
        # All images to show
        all_images = [representative] + duplicates
        n_images = len(all_images)
        
        # Calculate grid dimensions
        n_cols = min(5, n_images)  # Max 5 columns
        n_rows = (n_images + n_cols - 1) // n_cols
        
        # Calculate grid size
        grid_width = n_cols * (self.image_size[0] + self.space_width) - self.space_width
        grid_height = n_rows * (self.image_size[1] + self.space_width) - self.space_width
        
        # Create empty grid
        grid = np.full((grid_height, grid_width, 3), self.space_color, dtype=np.uint8)
        
        # Place images
        for idx, img_name in enumerate(all_images):
            row = idx // n_cols
            col = idx % n_cols
            
            # Load and resize image
            img_path = os.path.join(image_dir, os.path.basename(img_name))
            
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                img = cv2.resize(img, self.image_size)
                
                # Add similarity score (if not representative)
                if idx > 0 and scores and img_name in scores:
                    score_text = f"Sim: {scores[img_name]:.3f}"
                    img = self.add_text_overlay(img, score_text, position='bottom')
                
                # Add best indicator for representative
                if idx == 0:
                    img = self.add_best_indicator(img)
                
                # Place in grid
                x = col * (self.image_size[0] + self.space_width)
                y = row * (self.image_size[1] + self.space_width)
                
                grid[y:y + self.image_size[1], x:x + self.image_size[0]] = img
            
            except Exception as e:
                # Skip images that fail to load
                continue
        
        return grid
    
    def create_all_duplicate_grids(
        self,
        duplicate_results: 'DuplicateResults',
        image_dir: str,
        output_dir: str,
        top_n: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, str]:
        """
        Create grid images for all duplicate groups.
        """
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        grid_paths = {}
        
        # Iterate through groups
        iterator = enumerate(duplicate_results.groups.items())
        if verbose:
            iterator = tqdm(
                list(iterator),
                desc="Creating duplicate grids",
                unit="group"
            )
        
        for group_id, (representative, duplicates) in iterator:
            # Get scores for this group
            scores = {}
            if representative in duplicate_results.pairs:
                for dup_name, score in duplicate_results.pairs[representative]:
                    scores[dup_name] = score
            
            # Create grid
            grid = self.create_duplicate_grid(
                image_dir=image_dir,
                representative=representative,
                duplicates=duplicates,
                scores=scores,
                top_n=top_n
            )
            
            # Save grid
            output_path = os.path.join(output_dir, f'duplicate_group_{group_id}.jpg')
            cv2.imwrite(output_path, grid)
            grid_paths[group_id] = output_path
        
        if verbose:
            print(f"\n✓ Created {len(grid_paths)} duplicate grids in {output_dir}")
        
        return grid_paths


def create_duplicate_grids(
    duplicate_results: 'DuplicateResults',
    image_dir: str,
    output_dir: str,
    image_size: Tuple[int, int] = (300, 300),
    top_n: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, str]:
    """
    Convenience function to create duplicate grids.
    
    Example:
        >>> from imageatlas.duplicates import DuplicateDetector, create_duplicate_grids
        >>> detector = DuplicateDetector(method='phash')
        >>> results = detector.detect('./images')
        >>> grids = create_duplicate_grids(results, './images', './grids')
    """
    visualizer = DuplicateGridVisualizer(image_size=image_size)
    
    return visualizer.create_all_duplicate_grids(
        duplicate_results=duplicate_results,
        image_dir=image_dir,
        output_dir=output_dir,
        top_n=top_n,
        verbose=verbose
    )