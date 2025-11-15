"""
Model calibration utilities for well-calibrated probability predictions.
"""
from typing import Dict, Any, Optional, Tuple
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

from app.core.logging import get_logger

logger = get_logger(__name__)


class ModelCalibrator:
    """
    Calibrate classification models to produce well-calibrated probabilities.
    """

    def __init__(
        self,
        method: str = "isotonic",
        cv: int = 5
    ):
        """
        Initialize model calibrator.

        Args:
            method: Calibration method ('isotonic', 'sigmoid' (Platt scaling), or 'auto')
            cv: Number of cross-validation folds for calibration
        """
        self.method = method.lower()
        self.cv = cv
        self.calibrator = None
        self.is_fitted = False

    def fit(self, model: Any, X: np.ndarray, y: np.ndarray) -> 'ModelCalibrator':
        """
        Fit the calibrator on a trained model.

        Args:
            model: Trained classification model with predict_proba method
            X: Feature matrix
            y: True labels

        Returns:
            Self
        """
        if not hasattr(model, 'predict_proba'):
            raise ValueError("Model must have predict_proba method for calibration")

        # Auto-select method based on dataset size
        if self.method == "auto":
            n_samples = len(X)
            # For small datasets, use sigmoid (Platt scaling)
            # For larger datasets, use isotonic (more flexible)
            if n_samples < 1000:
                self.method = "sigmoid"
                logger.info("Auto-selected sigmoid calibration for small dataset")
            else:
                self.method = "isotonic"
                logger.info("Auto-selected isotonic calibration for larger dataset")

        # Create calibrated classifier
        self.calibrator = CalibratedClassifierCV(
            base_estimator=model,
            method=self.method,
            cv=self.cv,
            n_jobs=-1
        )

        self.calibrator.fit(X, y)
        self.is_fitted = True

        logger.info(f"Model calibrated using {self.method} method with {self.cv}-fold CV")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict calibrated probabilities.

        Args:
            X: Feature matrix

        Returns:
            Calibrated probability array
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before predicting")

        return self.calibrator.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Feature matrix

        Returns:
            Predicted class labels
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before predicting")

        return self.calibrator.predict(X)


def calculate_calibration_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Any]:
    """
    Calculate calibration metrics.

    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities (for positive class)
        n_bins: Number of bins for calibration curve

    Returns:
        Dictionary with calibration metrics
    """
    # Ensure probabilities are for positive class
    if y_pred_proba.ndim > 1:
        if y_pred_proba.shape[1] == 2:
            y_pred_proba = y_pred_proba[:, 1]
        else:
            raise ValueError("For multi-class, provide probabilities for each class separately")

    # Calculate Brier score (lower is better)
    brier_score = brier_score_loss(y_true, y_pred_proba)

    # Calculate calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=n_bins, strategy='uniform'
    )

    # Calculate Expected Calibration Error (ECE)
    # ECE = sum(|accuracy - confidence| * bin_size)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_pred_proba[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    # Calculate Maximum Calibration Error (MCE)
    mce = np.max(np.abs(fraction_of_positives - mean_predicted_value))

    return {
        "brier_score": float(brier_score),
        "expected_calibration_error": float(ece),
        "max_calibration_error": float(mce),
        "calibration_curve": {
            "fraction_of_positives": fraction_of_positives.tolist(),
            "mean_predicted_value": mean_predicted_value.tolist()
        }
    }


def calibrate_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    method: str = "auto",
    cv: int = 5
) -> ModelCalibrator:
    """
    Convenience function to calibrate a model.

    Args:
        model: Trained classification model
        X_train: Training features
        y_train: Training labels
        method: Calibration method
        cv: Number of CV folds

    Returns:
        Fitted ModelCalibrator
    """
    calibrator = ModelCalibrator(method=method, cv=cv)
    calibrator.fit(model, X_train, y_train)
    return calibrator

