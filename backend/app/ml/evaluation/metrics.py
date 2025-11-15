"""
Model evaluation metrics.
"""
from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


class MetricsCalculator:
    """Calculate evaluation metrics for ML models."""
    
    @staticmethod
    def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate regression metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Additional metrics
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100  # Mean Absolute Percentage Error
        median_ae = np.median(np.abs(y_true - y_pred))
        
        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2_score": float(r2),
            "mape": float(mape),
            "median_ae": float(median_ae)
        }
    
    @staticmethod
    def calculate_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = "weighted"
    ) -> Dict[str, Any]:
        """
        Calculate classification metrics.
        
        Args:
            y_true: True target labels
            y_pred: Predicted labels
            average: Averaging strategy for multi-class metrics
            
        Returns:
            Dictionary of metrics
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=average, zero_division=0)
        recall = recall_score(y_true, y_pred, average=average, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": cm.tolist(),
            "classification_report": report
        }
    
    @staticmethod
    def calculate_all_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        task_type: str = "regression"
    ) -> Dict[str, Any]:
        """
        Calculate all relevant metrics based on task type.
        
        Args:
            y_true: True target values/labels
            y_pred: Predicted values/labels
            task_type: Type of task (regression or classification)
            
        Returns:
            Dictionary of metrics
        """
        if task_type == "regression":
            return MetricsCalculator.calculate_regression_metrics(y_true, y_pred)
        elif task_type == "classification":
            return MetricsCalculator.calculate_classification_metrics(y_true, y_pred)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

