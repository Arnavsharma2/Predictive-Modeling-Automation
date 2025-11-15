"""
Base model class for ML models.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd


class BaseMLModel(ABC):
    """Base class for all ML models."""
    
    def __init__(self, model_type: str, **kwargs):
        """
        Initialize base model.
        
        Args:
            model_type: Type of model (regression, classification, anomaly_detection)
            **kwargs: Additional model parameters
        """
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        self.feature_names = None
        self.metadata = {}
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X: Feature matrix
            y: Target vector
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training metrics
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions array
        """
        pass
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance if available.
        
        Returns:
            Dictionary mapping feature names to importance scores, or None
        """
        if not self.is_trained or self.model is None:
            return None
        
        # Try to get feature_importances_ attribute (tree-based models)
        if hasattr(self.model, 'feature_importances_'):
            if self.feature_names is not None:
                return dict(zip(self.feature_names, self.model.feature_importances_))
            return dict(enumerate(self.model.feature_importances_))
        
        # Try to get coef_ attribute (linear models)
        if hasattr(self.model, 'coef_'):
            if self.feature_names is not None:
                return dict(zip(self.feature_names, self.model.coef_))
            return dict(enumerate(self.model.coef_))
        
        return None
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get model metadata."""
        return {
            "model_type": self.model_type,
            "is_trained": self.is_trained,
            "feature_names": self.feature_names,
            **self.metadata
        }

