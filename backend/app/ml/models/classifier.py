"""
Classification model implementation.
"""
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from app.ml.models.base import BaseMLModel
from app.ml.trainers.classification_trainer import ClassificationTrainer
from app.core.logging import get_logger

logger = get_logger(__name__)


class Classifier(BaseMLModel):
    """Classification model for multi-class classification."""
    
    def __init__(self, algorithm: str = "random_forest", **kwargs):
        """
        Initialize classifier.
        
        Args:
            algorithm: Algorithm to use (random_forest, xgboost)
            **kwargs: Additional parameters
        """
        super().__init__(model_type="classification", **kwargs)
        self.algorithm = algorithm
        self.trainer = ClassificationTrainer(algorithm=algorithm)
        self.classes_ = None  # Class labels
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        hyperparameters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the classification model.
        
        Args:
            X: Feature matrix
            y: Target labels
            hyperparameters: Model hyperparameters
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training metrics
        """
        logger.info("Training classification model...")
        
        # Train using trainer
        results = self.trainer.train(X, y, hyperparameters=hyperparameters, **kwargs)
        
        # Store model and metadata
        self.model = results["model"]
        self.feature_names = results["feature_names"]
        self.classes_ = results.get("classes_")
        self.is_trained = True
        
        # Store metadata
        self.metadata = {
            "algorithm": results["algorithm"],
            "hyperparameters": results["hyperparameters"],
            "n_classes": len(self.classes_) if self.classes_ is not None else 0,
            "classes": self.classes_.tolist() if self.classes_ is not None else [],
            "train_metrics": results["train_metrics"],
            "test_metrics": results["test_metrics"],
            "cv_scores": results.get("cv_scores"),
            "feature_importance": results.get("feature_importance")
        }
        
        logger.info(f"Classification model training completed. Accuracy: {results['test_metrics']['accuracy']:.4f}")
        return {
            "train_metrics": results["train_metrics"],
            "test_metrics": results["test_metrics"],
            "cv_scores": results.get("cv_scores"),
            "feature_importance": results.get("feature_importance"),
            "classes": self.classes_.tolist() if self.classes_ is not None else []
        }
    
    def predict(self, X: pd.DataFrame, handle_missing_features: str = "error") -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            handle_missing_features: How to handle missing features ('error', 'warn', 'fill_zero', 'fill_mean')
            
        Returns:
            Predictions array (class labels)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions. Call train() first.")
        
        if self.model is None:
            raise ValueError("Model not initialized. Training may have failed.")
        
        # Check for missing features
        if self.feature_names:
            missing_features = [f for f in self.feature_names if f not in X.columns]
            extra_features = [f for f in X.columns if f not in self.feature_names]
            
            if missing_features:
                error_msg = (
                    f"Missing required features: {missing_features[:10]}" +
                    (f" (and {len(missing_features) - 10} more)" if len(missing_features) > 10 else "") +
                    f". Required features: {len(self.feature_names)}. Provided: {len(X.columns)}."
                )
                
                if handle_missing_features == "error":
                    raise ValueError(error_msg + " Set handle_missing_features='warn' or 'fill_zero' to continue.")
                elif handle_missing_features == "warn":
                    logger.warning(error_msg + " Filling with zeros. Predictions may be inaccurate.")
                    for feat in missing_features:
                        X[feat] = 0.0
                elif handle_missing_features in ["fill_zero", "fill_mean"]:
                    for feat in missing_features:
                        X[feat] = 0.0
            
            if extra_features:
                logger.debug(f"Ignoring {len(extra_features)} extra features not used during training")
            
            # Ensure feature order matches training
            X = X[[col for col in self.feature_names if col in X.columns]]
        
        try:
            predictions = self.model.predict(X)
            return predictions
        except Exception as e:
            error_msg = f"Prediction failed: {str(e)}"
            if "feature" in str(e).lower() or "column" in str(e).lower():
                error_msg += f". Expected {len(self.feature_names)} features: {self.feature_names[:5]}..."
            raise RuntimeError(error_msg) from e
    
    def predict_proba(self, X: pd.DataFrame, handle_missing_features: str = "error") -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix
            handle_missing_features: How to handle missing features ('error', 'warn', 'fill_zero', 'fill_mean')
            
        Returns:
            Probability array (shape: [n_samples, n_classes])
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions. Call train() first.")
        
        if self.model is None:
            raise ValueError("Model not initialized. Training may have failed.")
        
        # Check for missing features (reuse logic from predict)
        if self.feature_names:
            missing_features = [f for f in self.feature_names if f not in X.columns]
            if missing_features:
                if handle_missing_features == "error":
                    raise ValueError(
                        f"Missing required features: {missing_features[:10]}. "
                        "Set handle_missing_features='warn' or 'fill_zero' to continue."
                    )
                elif handle_missing_features in ["warn", "fill_zero", "fill_mean"]:
                    if handle_missing_features == "warn":
                        logger.warning(f"Missing features: {missing_features[:5]}... Filling with zeros.")
                    for feat in missing_features:
                        X[feat] = 0.0
            
            # Ensure feature order matches training
            X = X[[col for col in self.feature_names if col in X.columns]]
        
        if hasattr(self.model, 'predict_proba'):
            try:
                probabilities = self.model.predict_proba(X)
                return probabilities
            except Exception as e:
                error_msg = f"Probability prediction failed: {str(e)}"
                if "feature" in str(e).lower():
                    error_msg += f". Expected {len(self.feature_names)} features."
                raise RuntimeError(error_msg) from e
        else:
            raise ValueError(
                f"Model type {type(self.model).__name__} does not support probability predictions. "
                "Use predict() for class labels instead."
            )
    
    def predict_with_proba(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Make predictions with probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary with predictions and probabilities
        """
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        
        return {
            "predictions": predictions,
            "probabilities": probabilities,
            "classes": self.classes_.tolist() if self.classes_ is not None else []
        }

