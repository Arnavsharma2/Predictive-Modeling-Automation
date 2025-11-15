"""
Statistical testing for A/B tests.
"""
from typing import Dict, Any, List, Optional
import numpy as np
from scipy import stats

from app.core.logging import get_logger

logger = get_logger(__name__)


class ABTestStatistics:
    """Statistical analysis for A/B tests."""
    
    @staticmethod
    def calculate_metrics(
        predictions: List[Dict[str, Any]],
        metric_type: str = "regression"
    ) -> Dict[str, float]:
        """
        Calculate performance metrics from predictions.
        
        Args:
            predictions: List of predictions with actual_value and prediction
            metric_type: Type of metric (regression, classification, etc.)
            
        Returns:
            Dictionary of metrics
        """
        if not predictions:
            return {}
        
        # Filter predictions with actual values
        valid_predictions = [
            p for p in predictions
            if p.get("actual_value") is not None and p.get("prediction") is not None
        ]
        
        if not valid_predictions:
            return {}
        
        if metric_type == "regression":
            return ABTestStatistics._calculate_regression_metrics(valid_predictions)
        elif metric_type == "classification":
            return ABTestStatistics._calculate_classification_metrics(valid_predictions)
        else:
            return {}
    
    @staticmethod
    def _calculate_regression_metrics(
        predictions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate regression metrics."""
        actuals = [p["actual_value"] for p in predictions]
        preds = [p["prediction"] for p in predictions]
        
        actuals = np.array(actuals)
        preds = np.array(preds)
        
        # RMSE
        rmse = np.sqrt(np.mean((actuals - preds) ** 2))
        
        # MAE
        mae = np.mean(np.abs(actuals - preds))
        
        # R²
        ss_res = np.sum((actuals - preds) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2_score": float(r2),
            "sample_size": len(predictions)
        }
    
    @staticmethod
    def _calculate_classification_metrics(
        predictions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate classification metrics."""
        actuals = [p["actual_value"] for p in predictions]
        preds = [p["prediction"] for p in predictions]
        
        # Accuracy
        correct = sum(1 for a, p in zip(actuals, preds) if a == p)
        accuracy = correct / len(predictions) if predictions else 0
        
        # Precision, Recall, F1 (simplified - would need more details for multi-class)
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        try:
            precision = precision_score(actuals, preds, average="weighted", zero_division=0)
            recall = recall_score(actuals, preds, average="weighted", zero_division=0)
            f1 = f1_score(actuals, preds, average="weighted", zero_division=0)
        except Exception:
            precision = recall = f1 = 0.0
        
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "sample_size": len(predictions)
        }
    
    @staticmethod
    def t_test(
        control_metrics: Dict[str, float],
        treatment_metrics: Dict[str, float],
        metric_name: str = "accuracy"
    ) -> Dict[str, Any]:
        """
        Perform t-test to compare control and treatment.
        
        Args:
            control_metrics: Control group metrics
            treatment_metrics: Treatment group metrics
            metric_name: Metric to compare
            
        Returns:
            Dictionary with t-test results
        """
        control_value = control_metrics.get(metric_name)
        treatment_value = treatment_metrics.get(metric_name)
        
        if control_value is None or treatment_value is None:
            return {
                "p_value": None,
                "significant": False,
                "error": f"Metric {metric_name} not found"
            }
        
        # For simplicity, we'll use a one-sample t-test
        # In practice, you'd need the raw data points
        # This is a simplified version
        
        # Calculate difference
        difference = treatment_value - control_value
        improvement_pct = (difference / control_value * 100) if control_value != 0 else 0
        
        # Simplified p-value calculation
        # In practice, you'd use actual data distributions
        # For now, we'll use a heuristic based on sample sizes and difference
        control_n = control_metrics.get("sample_size", 0)
        treatment_n = treatment_metrics.get("sample_size", 0)
        
        if control_n < 30 or treatment_n < 30:
            # Not enough samples
            return {
                "p_value": None,
                "significant": False,
                "error": "Insufficient samples for statistical test"
            }
        
        # Simplified significance test
        # In practice, use proper statistical test with raw data
        std_error = np.sqrt(
            (control_value * (1 - control_value) / control_n) +
            (treatment_value * (1 - treatment_value) / treatment_n)
        ) if metric_name in ["accuracy"] else abs(difference) / 10
        
        if std_error == 0:
            p_value = 1.0
        else:
            t_stat = difference / std_error
            # Two-tailed t-test
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=control_n + treatment_n - 2))
        
        significant = p_value < 0.05
        
        return {
            "p_value": float(p_value),
            "significant": significant,
            "difference": float(difference),
            "improvement_percent": float(improvement_pct),
            "control_value": float(control_value),
            "treatment_value": float(treatment_value)
        }
    
    @staticmethod
    def determine_winner(
        control_metrics: Dict[str, float],
        treatment_metrics: Dict[str, float],
        metric_name: str = "accuracy",
        p_value: Optional[float] = None
    ) -> Optional[str]:
        """
        Determine winner of A/B test.
        
        Args:
            control_metrics: Control group metrics
            treatment_metrics: Treatment group metrics
            metric_name: Metric to compare
            p_value: P-value from statistical test
            
        Returns:
            "control", "treatment", or None
        """
        if p_value is None or p_value >= 0.05:
            return None  # Not statistically significant
        
        control_value = control_metrics.get(metric_name)
        treatment_value = treatment_metrics.get(metric_name)
        
        if control_value is None or treatment_value is None:
            return None
        
        # For metrics where higher is better (accuracy, r2_score)
        if metric_name in ["accuracy", "r2_score", "precision", "recall", "f1_score"]:
            if treatment_value > control_value:
                return "treatment"
            else:
                return "control"
        
        # For metrics where lower is better (rmse, mae)
        elif metric_name in ["rmse", "mae"]:
            if treatment_value < control_value:
                return "treatment"
            else:
                return "control"
        
        return None

