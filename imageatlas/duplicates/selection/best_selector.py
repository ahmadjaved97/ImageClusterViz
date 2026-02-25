"""
Best image selection strategies for duplicate groups.
"""

import os
from pathlib import Path
from PIL import Image

from ..base import BestImageSelector


class ResolutionSelector(BestImageSelector):
    """
    Select image with highest resolution (width x height)
    """

    def select_best(
        self,
        image_paths,
        metadata=None,
    ):
        """
        Select image with highest resolution.
        """

        if not image_paths:
            raise ValueError("No images provided")
        
        if len(image_paths) == 1:
            return image_paths[0]
        
        best_image = None
        best_resolution = -1

        for path in image_paths:
            try:
                # Get image size without loading full image
                with Image.open(path) as img:
                    width, height = img.size
                    resolution = width * height
                
                if resolution > best_resolution:
                    best_resolution = resolution
                    best_image = path
            
            except Exception:
                # If we can't open image, skip it (REVISIT)
                continue
        
        # Fallback to first image if all failed
        if best_image is None:
            best_image = image_paths[0]
        
        return best_image

class AlphabeticSelector(BestImageSelector):
    """
    Selects image that comes first alphabetically.
    """

    def select_best(
        self,
        image_paths,
        metadata=None
    ):
        """
        Select image that comes first alphabetically.
        """

        if not image_paths:
            raise ValueError("No images provided")
        
        # Sort and return first
        return sorted(image_paths)[0]
    
class FileSizeSelector(BestImageSelector):
    """
    Select image with largest file size.
    """

    def select_best(
        self,
        image_paths,
        metadata=None
    ):
        """
        Select image with largest file size.
        """

        if not image_paths:
            raise ValueError("No images provided")
        
        if len(image_paths) == 1:
            return image_paths[0]
        
        best_image = None
        best_size = -1

        for path in image_paths:
            try:
                file_size = os.path.getsize(path)

                if file_size > best_size:
                    best_size = file_size
                    best_image = path
            
            except Exception:
                # If we can't get filesize, skip
                continue
        
        # Fallback to first image
        if best_image is None:
            best_image = image_paths[0]
        

        return best_image

# Implement a composite selector (REVISIT)


def create_best_selector(selection_type):
    """
    Factory function to create best image selector.

    Args:
        selection_type: Type of selector
            - resolution: Highest resolution
            - alphabetic: First alphabetically
            - filesize: Largest file size
    
    Returns:
        BestImageSelector instance
    
    """

    if selection_type == 'resolution':
        return ResolutionSelector()
    elif selection_type == 'alphabetic':
        return AlphabeticSelector()
    elif selection_type == 'filesize':
        return FileSizeSelector()
    else:
        raise ValueError(f"Unknown selection type: {selection_type}")