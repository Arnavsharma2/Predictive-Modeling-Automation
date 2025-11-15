"""
Base trainer class for ML models.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


class BaseTrainer(ABC):
    """Base class for model trainers."""
    
    def __init__(self, model_type: str):
        """
        Initialize trainer.
        
        Args:
            model_type: Type of model to train
        """
        self.model_type = model_type
    
    @abstractmethod
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        hyperparameters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train a model.
        
        Args:
            X: Feature matrix
            y: Target vector
            hyperparameters: Model hyperparameters
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with trained model and metrics
        """
        pass

