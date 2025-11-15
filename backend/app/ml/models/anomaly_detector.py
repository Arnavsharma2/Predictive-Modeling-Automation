"""
Anomaly detection model implementation using Isolation Forest.
"""
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from app.ml.models.base import BaseMLModel
from app.core.logging import get_logger

logger = get_logger(__name__)


class AnomalyDetector(BaseMLModel):
    """Anomaly detection model using Isolation Forest."""
    
    def __init__(self, contamination: float = 0.1, **kwargs):
        """
        Initialize anomaly detector.
        
        Args:
            contamination: Expected proportion of anomalies (0-0.5)
            **kwargs: Additional parameters
        """
        super().__init__(model_type="anomaly_detection", **kwargs)
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.threshold = None  # Anomaly score threshold
    
    def train(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the anomaly detection model.
        
        Args:
            X: Feature matrix
            y: Not used for anomaly detection (unsupervised)
            hyperparameters: Model hyperparameters
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training metrics
        """
        logger.info("Training anomaly detection model...")
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create model with hyperparameters
        model_params = {
            "contamination": self.contamination,
            "random_state": 42,
            "n_estimators": 100
        }
        
        if hyperparameters:
            model_params.update(hyperparameters)
        
        # Train Isolation Forest
        self.model = IsolationForest(**model_params)
        self.model.fit(X_scaled)
        
        # Calculate anomaly scores for training data
        scores = self.model.score_samples(X_scaled)
        
        # Set threshold (negative scores indicate anomalies)
        # Use percentile-based threshold
        threshold_percentile = kwargs.get("threshold_percentile", 100 * self.contamination)
        self.threshold = np.percentile(scores, threshold_percentile)
        
        # Predict anomalies
        predictions = self.model.predict(X_scaled)
        anomalies = (predictions == -1).sum()
        normal = (predictions == 1).sum()
        
        self.is_trained = True
        
        # Store metadata
        self.metadata = {
            "contamination": self.contamination,
            "threshold": float(self.threshold),
            "n_samples": len(X),
            "n_anomalies_detected": int(anomalies),
            "n_normal": int(normal),
            "anomaly_rate": float(anomalies / len(X))
        }
        
        logger.info(f"Anomaly detection model trained. Detected {anomalies} anomalies out of {len(X)} samples.")
        
        return {
            "n_samples": len(X),
            "n_anomalies": int(anomalies),
            "n_normal": int(normal),
            "anomaly_rate": float(anomalies / len(X)),
            "threshold": float(self.threshold),
            "score_stats": {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores))
            }
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict anomalies.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions (-1 for anomaly, 1 for normal)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if self.model is None:
            raise ValueError("Model not initialized")
        
        # Ensure feature order matches training
        if self.feature_names:
            X = X[[col for col in self.feature_names if col in X.columns]]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        return predictions
    
    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get anomaly scores (lower scores = more anomalous).
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of anomaly scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if self.model is None:
            raise ValueError("Model not initialized")
        
        # Ensure feature order matches training
        if self.feature_names:
            X = X[[col for col in self.feature_names if col in X.columns]]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get scores
        scores = self.model.score_samples(X_scaled)
        return scores
    
    def predict_with_scores(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Predict anomalies with scores.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary with predictions and scores
        """
        predictions = self.predict(X)
        scores = self.predict_scores(X)
        
        return {
            "predictions": predictions,
            "scores": scores,
            "is_anomaly": (predictions == -1),
            "anomaly_probability": self._score_to_probability(scores)
        }
    
    def _score_to_probability(self, scores: np.ndarray) -> np.ndarray:
        """
        Convert anomaly scores to probabilities.
        
        Args:
            scores: Anomaly scores
            
        Returns:
            Probability array (0 = normal, 1 = anomaly)
        """
        if self.threshold is None:
            # Use contamination-based threshold
            threshold = np.percentile(scores, 100 * self.contamination)
        else:
            threshold = self.threshold
        
        # Normalize scores to probabilities
        # Lower scores = higher anomaly probability
        probabilities = 1 / (1 + np.exp(scores - threshold))
        return probabilities

