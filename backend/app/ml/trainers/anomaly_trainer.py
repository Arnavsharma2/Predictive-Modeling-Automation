"""
Anomaly detection model trainer.
"""
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from app.ml.trainers.trainer import BaseTrainer
from app.ml.models.anomaly_detector import AnomalyDetector
from app.core.logging import get_logger

logger = get_logger(__name__)


class AnomalyTrainer(BaseTrainer):
    """Trainer for anomaly detection models."""
    
    def __init__(self, algorithm: str = "isolation_forest"):
        """
        Initialize anomaly trainer.
        
        Args:
            algorithm: Algorithm to use (currently only isolation_forest)
        """
        super().__init__(model_type="anomaly_detection")
        self.algorithm = algorithm.lower()
        self.model = None
        self.feature_names = None
    
    def train(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        contamination: float = 0.1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train an anomaly detection model.
        
        Args:
            X: Feature matrix
            y: Not used (unsupervised learning)
            hyperparameters: Model hyperparameters
            contamination: Expected proportion of anomalies
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with trained model, metrics, and metadata
        """
        logger.info(f"Starting anomaly detection training with algorithm: {self.algorithm}")
        
        if self.algorithm != "isolation_forest":
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Create and train detector
        detector = AnomalyDetector(contamination=contamination)
        training_results = detector.train(X, hyperparameters=hyperparameters, **kwargs)
        
        # Store model
        self.model = detector
        
        # Calculate additional metrics
        scores = detector.predict_scores(X)
        predictions = detector.predict(X)
        
        results = {
            "model": detector,
            "feature_names": self.feature_names,
            "training_metrics": training_results,
            "algorithm": self.algorithm,
            "contamination": contamination,
            "score_distribution": {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "percentiles": {
                    "p5": float(np.percentile(scores, 5)),
                    "p25": float(np.percentile(scores, 25)),
                    "p50": float(np.percentile(scores, 50)),
                    "p75": float(np.percentile(scores, 75)),
                    "p95": float(np.percentile(scores, 95))
                }
            }
        }
        
        logger.info(f"Anomaly detection training completed. Anomaly rate: {training_results['anomaly_rate']:.2%}")
        
        return results

