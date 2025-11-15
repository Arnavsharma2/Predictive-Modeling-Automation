"""
Predictive regression model implementation.
"""
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from app.ml.models.base import BaseMLModel
from app.ml.trainers.regression_trainer import RegressionTrainer
from app.ml.evaluation.metrics import MetricsCalculator
from app.core.logging import get_logger

logger = get_logger(__name__)


class RegressionPredictor(BaseMLModel):
    """Regression model for predictive analytics."""
    
    def __init__(self, algorithm: str = "random_forest", **kwargs):
        """
        Initialize regression predictor.
        
        Args:
            algorithm: Algorithm to use (random_forest, xgboost)
            **kwargs: Additional parameters
        """
        super().__init__(model_type="regression", **kwargs)
        self.algorithm = algorithm
        self.trainer = RegressionTrainer(algorithm=algorithm)
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        hyperparameters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the regression model.
        
        Args:
            X: Feature matrix
            y: Target vector
            hyperparameters: Model hyperparameters
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training metrics
        """
        logger.info("Training regression model...")
        
        # Train using trainer
        results = self.trainer.train(X, y, hyperparameters=hyperparameters, **kwargs)
        
        # Store model and metadata
        self.model = results["model"]
        self.feature_names = results["feature_names"]
        self.is_trained = True
        
        # Store metadata
        self.metadata = {
            "algorithm": results["algorithm"],
            "hyperparameters": results["hyperparameters"],
            "train_metrics": results["train_metrics"],
            "test_metrics": results["test_metrics"],
            "cv_scores": results.get("cv_scores"),
            "feature_importance": results.get("feature_importance")
        }
        
        logger.info("Regression model training completed")
        return {
            "train_metrics": results["train_metrics"],
            "test_metrics": results["test_metrics"],
            "test_predictions": results.get("test_predictions"),
            "test_targets": results.get("test_targets"),
            "cv_scores": results.get("cv_scores"),
            "feature_importance": results.get("feature_importance")
        }
    
    def predict(self, X: pd.DataFrame, handle_missing_features: str = "error") -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            handle_missing_features: How to handle missing features ('error', 'warn', 'fill_zero', 'fill_mean')
            
        Returns:
            Predictions array
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
                    # Fill missing features with zeros
                    for feat in missing_features:
                        X[feat] = 0.0
                elif handle_missing_features == "fill_zero":
                    for feat in missing_features:
                        X[feat] = 0.0
                elif handle_missing_features == "fill_mean":
                    # This would require storing training means, which we don't have
                    # Fall back to fill_zero
                    logger.warning("fill_mean not available, using fill_zero instead")
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
    
    def predict_with_confidence(
        self,
        X: pd.DataFrame,
        confidence_level: float = 0.95,
        method: str = "tree_std"
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions with confidence intervals (if available).

        Args:
            X: Feature matrix
            confidence_level: Confidence level (0.95 = 95% interval)
            method: Method for confidence intervals ('tree_std', 'quantile', 'conformal')

        Returns:
            Dictionary with predictions and confidence intervals
        """
        predictions = self.predict(X)

        # Calculate z-score for confidence level
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence_level) / 2)

        # For tree-based models, we can estimate uncertainty using individual tree predictions
        if method == "tree_std" and hasattr(self.model, 'estimators_'):
            # Get predictions from each tree
            tree_predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
            std = np.std(tree_predictions, axis=0)
            
            # Handle edge case where std is zero
            std = np.maximum(std, np.abs(predictions) * 0.01)  # At least 1% of prediction

            return {
                "predictions": predictions,
                "std": std,
                "lower_bound": predictions - z_score * std,
                "upper_bound": predictions + z_score * std,
                "confidence_level": confidence_level,
                "method": "tree_std"
            }

        # For quantile regression (if available)
        elif method == "quantile" and hasattr(self.model, 'estimators_'):
            # Use percentiles from tree predictions
            tree_predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
            alpha = (1 - confidence_level) / 2
            lower_percentile = alpha * 100
            upper_percentile = (1 - alpha) * 100
            
            lower_bound = np.percentile(tree_predictions, lower_percentile, axis=0)
            upper_bound = np.percentile(tree_predictions, upper_percentile, axis=0)
            std = np.std(tree_predictions, axis=0)

            return {
                "predictions": predictions,
                "std": std,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "confidence_level": confidence_level,
                "method": "quantile"
            }

        # Fallback: use simple heuristic based on prediction magnitude
        else:
            # Estimate uncertainty as percentage of prediction (conservative)
            uncertainty_pct = 0.1  # 10% uncertainty
            std = np.abs(predictions) * uncertainty_pct
            std = np.maximum(std, np.abs(predictions).mean() * 0.05)  # At least 5% of mean

            return {
                "predictions": predictions,
                "std": std,
                "lower_bound": predictions - z_score * std,
                "upper_bound": predictions + z_score * std,
                "confidence_level": confidence_level,
                "method": "heuristic"
            }

