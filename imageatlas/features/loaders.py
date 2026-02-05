"""
Image loading utilities with validation and error handling.
"""

import os
from pathlib import Path
from PIL import Image
import numpy as np
import warnings


class ImageLoader:
    """
    Image loader with validation and error handling.

    Handles corrupted images, format conversions, and EXIF orientations.
    """
    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

    def __init__(
        self,
        max_size=None,
        convert_mode='RGB',
        handle_exif=True,
    ):
        """
        Initialize image loader.

        Args:
            max_size: Optional maximum size (width, height) for images
            convert_mode: PIL odel to covert images to ('RGB', 'L', etc.)
            handle_exif: Whether to handle EXIF orientation
        """

        self.max_size = max_size
        self.convert_mode = convert_mode
        self.handle_exif = handle_exif

    
    def validate_path(self, path):
        """
        Check if path is valid

        Args:
            path: Path to image file
        
        Returns:
            True if valid, False otherwise
        """
        if not os.path.exists(path):
            return False
        
        ext = Path(path).suffix.lower()
        return ext in self.VALID_EXTENSIONS
    
    def load_image(self, path):
        """
        Load a single image.

        Args:
            path: Path to image file
        
        Returns:
            PIL Image or None if loadig failed
        """

        try:
            if not self.validate_path(path):
                warnings.warn(f"Invalid image path or format: {path}")
                return None
            
            # Load image
            image = Image.open(path)


            # Handle EXIF orientation
            if self.handle_exif:
                image = self._handle_orientation(image)
            
            # Convert mode
            if image.mode != self.convert_mode:
                image = image.convert(self.convert_mode)
            
            # Resize if needed
            if self.max_size:
                image = self._resize_if_needed(image)
            
            return image
        
        except (IOError, OSError, Image.DecompressionBombError) as e:
            warnings.warn(f"Failed to load image {path}: {str(e)}")
            return None
        except Exception as e:
            warnings.warn(f"Unexpected error loading {path}: {str(e)}")
            return None
    
    def load_batch(self, paths):
        """
        Load a batch of images.

        Args:
            paths: List of image paths
        
        Returns:
            Tuple of (loaded_images, successful_paths, failed_paths)
        """

        images = []
        successful = []
        failed = []

        for path in paths:
            image = self.load_image(path)
            if image is not None:
                images.append(image)
                successful.append(path)
            else:
                failed.append(path)
        
        return images, successful, failed
    
    def _handle_orientation(self, image):
        """
        Handle EXIF orientation tag.

        Args:
            image: PIL Image
        
        Returns:
            Oriented image
        """

        try:
            # Get EXIF data
            exif = image._getexif()
            if exif is None:
                return image
            

            # Get orientation tag (274 is the EXIF orientation tag)
            orientation = exif.get(274)

            if orientation is None:
                return image
            
            # Apply rotation based on orientation
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
        
        except (AttributeError, KeyError, TypeError):
            # No EXIF data or orientation tag.
            pass
        
        return image
    
    def _resize_if_needed(self, image):
        """
        Resize image if it exceeds max size.

        Args:
            image: PIL Image
        
        Returns:
            Resized image
        """
        if self.max_size is None:
            return image
        
        # Check if resize is needed
        if image.size[0] <= self.max_size[0] and image.size[1] <= self.max_size[1]:
            return image
        
        # Resize maintaining aspect ratio
        image.thumbnail(self.max_size, Image.Resampling.LANCZOS)
        return image
    
    @staticmethod
    def find_images(
        directory,
        pattern = "*",
        recursive = True
    ):
        """
        Find all images in a directory.

        Args:
            directory: Directory to search
            pattern: Glob pattern for filenames
            recursive: Whether to search recursively
        
        Returns:
            List of image paths
        """
        path = Path(directory)

        if not path.exists():
            raise ValueError(f"Directory does not exist: {directory}")
        
        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")
        
        # Find all files.
        if recursive:
            files = path.rglob(pattern)
        else:
            files = path.glob(pattern)
        
        # Filter for valid image extensions
        images = [
            str(f) for f in files
            if f.is_file() and f.suffix.lower() in ImageLoader.VALID_EXTENSIONS
        ]

        return sorted(images)
    
    @staticmethod
    def create_batches(
        items,
        batch_size
    ):

        """
        Create batches from a list of items.

        Args:
            items: List of items to batch
            batch_size: Size of each batch
        
        Yeilds:
            Batches of items
        """

        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]

