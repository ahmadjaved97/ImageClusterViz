"""Abstract base class for all feature extractors."""

import numpy as np
import torch
from abc import ABC, abstractmethod
from pathlib import Path


class FeatureExtractor(ABC):
    """Abstract base class for feature extractors."""

    def __init__(self, device='cpu'):
        self.device = device
        self.model = None
        self.preprocess = None
    
    
    @abstractmethod
    def load_model(self):
        """Load the model and the preprocessing pipeline."""
        pass
    
    def extract_features(self, image):
        """Extract features from an image."""
        if self.model is None or self.preprocess is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        # Extract features.
        with torch.no_grad():
            features = self._forward(image_tensor)
        
        return features.cpu().numpy()
    
    @abstractmethod
    def _forward(self, image_tensor):
        """ Model specific forward pass."""
        pass
    
    def load_custom_weights(self, weights_path):
        """ Load custom weights into the model."""

        if self.model is None:
            raise RuntimeError("Model must be initialized before loading custom weights.")
        
        state_dict = torch.load(weights_path, map_location=self.device)

        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        self.model.load_state_dict(state_dict, strict=False)
        print(f"Loaded custom weights from {weights_path}")