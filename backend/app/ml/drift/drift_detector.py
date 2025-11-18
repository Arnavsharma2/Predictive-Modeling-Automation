"""
Drift detection service.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone
from app.ml.drift.statistical_tests import StatisticalTests
from app.core.logging import get_logger

logger = get_logger(__name__)


class DriftDetector:
    """Service for detecting data drift and concept drift."""
    
    def __init__(self, significance_level: float = 0.05, psi_threshold: float = 0.25):
        """
        Initialize drift detector.
        
        Args:
            significance_level: Significance level for statistical tests (default: 0.05)
            psi_threshold: PSI threshold for drift detection (default: 0.25)
        """
        self.significance_level = significance_level
        self.psi_threshold = psi_threshold
        self.stats_tests = StatisticalTests()
    
    def detect_data_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        features: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Detect data drift between reference and current datasets.
        
        Args:
            reference_data: Reference dataset (training data)
            current_data: Current dataset (production data)
            features: List of features to check (None = all common features)
            
        Returns:
            Dictionary with drift detection results
        """
        try:
            # Get common features
            if features is None:
                features = list(set(reference_data.columns) & set(current_data.columns))
            
            if not features:
                return {
                    "drift_detected": False,
                    "message": "No common features found between reference and current data",
                    "feature_results": {}
                }
            
            feature_results = {}
            drift_detected = False
            drift_severity = "none"
            
            for feature in features:
                if feature not in reference_data.columns or feature not in current_data.columns:
                    continue
                
                ref_values = reference_data[feature].values
                curr_values = current_data[feature].values
                
                # Skip if all values are NaN
                if np.all(pd.isna(ref_values)) or np.all(pd.isna(curr_values)):
                    continue
                
                feature_result = self._detect_feature_drift(ref_values, curr_values, feature)
                feature_results[feature] = feature_result
                
                # Check if drift detected for this feature
                if feature_result.get("drift_detected", False):
                    drift_detected = True
                    # Update severity if this is more severe
                    if feature_result.get("severity") == "high":
                        drift_severity = "high"
                    elif feature_result.get("severity") == "medium" and drift_severity != "high":
                        drift_severity = "medium"
                    elif drift_severity == "none":
                        drift_severity = "low"
            
            return {
                "drift_detected": drift_detected,
                "drift_severity": drift_severity,
                "feature_results": feature_results,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reference_samples": len(reference_data),
                "current_samples": len(current_data),
                "features_checked": len(feature_results)
            }
        except Exception as e:
            logger.error(f"Error detecting data drift: {e}", exc_info=True)
            return {
                "drift_detected": False,
                "error": str(e),
                "feature_results": {}
            }
    
    def _detect_feature_drift(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        feature_name: str
    ) -> Dict[str, Any]:
        """
        Detect drift for a single feature.
        
        Args:
            reference: Reference values
            current: Current values
            feature_name: Name of the feature
            
        Returns:
            Dictionary with drift detection results for the feature
        """
        try:
            # Remove NaN values for calculations
            ref_clean = reference[~pd.isna(reference)]
            curr_clean = current[~pd.isna(current)]
            
            if len(ref_clean) == 0 or len(curr_clean) == 0:
                return {
                    "feature": feature_name,
                    "drift_detected": False,
                    "message": "Insufficient data for drift detection"
                }
            
            # Calculate PSI
            psi = self.stats_tests.calculate_psi(ref_clean, curr_clean)
            
            # Perform KS test
            ks_statistic, ks_pvalue = self.stats_tests.kolmogorov_smirnov_test(ref_clean, curr_clean)
            
            # Perform Chi-square test
            chi2_statistic, chi2_pvalue = self.stats_tests.chi_square_test(ref_clean, curr_clean)
            
            # Calculate Wasserstein distance
            wasserstein_dist = self.stats_tests.wasserstein_distance(ref_clean, curr_clean)
            
            # Calculate JS divergence
            js_divergence = self.stats_tests.jensen_shannon_divergence(ref_clean, curr_clean)
            
            # Determine drift status
            drift_detected = False
            severity = "none"
            drift_reasons = []
            
            # PSI-based detection
            if not np.isnan(psi):
                if psi >= self.psi_threshold:
                    drift_detected = True
                    drift_reasons.append(f"PSI={psi:.4f} >= {self.psi_threshold}")
                    if psi >= 0.5:
                        severity = "high"
                    elif psi >= 0.35:
                        severity = "medium"
                    else:
                        severity = "low"
            
            # KS test-based detection
            if not np.isnan(ks_pvalue) and ks_pvalue < self.significance_level:
                drift_detected = True
                drift_reasons.append(f"KS test p-value={ks_pvalue:.4f} < {self.significance_level}")
                if severity == "none":
                    severity = "medium"
            
            # Chi-square test-based detection
            if not np.isnan(chi2_pvalue) and chi2_pvalue < self.significance_level:
                drift_detected = True
                drift_reasons.append(f"Chi-square test p-value={chi2_pvalue:.4f} < {self.significance_level}")
                if severity == "none":
                    severity = "medium"
            
            # Calculate summary statistics
            ref_mean = float(np.mean(ref_clean)) if len(ref_clean) > 0 else np.nan
            curr_mean = float(np.mean(curr_clean)) if len(curr_clean) > 0 else np.nan
            ref_std = float(np.std(ref_clean)) if len(ref_clean) > 0 else np.nan
            curr_std = float(np.std(curr_clean)) if len(curr_clean) > 0 else np.nan
            
            return {
                "feature": feature_name,
                "drift_detected": drift_detected,
                "severity": severity,
                "drift_reasons": drift_reasons,
                "metrics": {
                    "psi": float(psi) if not (np.isnan(psi) or np.isinf(psi)) else None,
                    "ks_statistic": float(ks_statistic) if not (np.isnan(ks_statistic) or np.isinf(ks_statistic)) else None,
                    "ks_pvalue": float(ks_pvalue) if not (np.isnan(ks_pvalue) or np.isinf(ks_pvalue)) else None,
                    "chi2_statistic": float(chi2_statistic) if not (np.isnan(chi2_statistic) or np.isinf(chi2_statistic)) else None,
                    "chi2_pvalue": float(chi2_pvalue) if not (np.isnan(chi2_pvalue) or np.isinf(chi2_pvalue)) else None,
                    "wasserstein_distance": float(wasserstein_dist) if not (np.isnan(wasserstein_dist) or np.isinf(wasserstein_dist)) else None,
                    "js_divergence": float(js_divergence) if not (np.isnan(js_divergence) or np.isinf(js_divergence)) else None,
                },
                "statistics": {
                    "reference": {
                        "mean": ref_mean if not (np.isnan(ref_mean) or np.isinf(ref_mean)) else None,
                        "std": ref_std if not (np.isnan(ref_std) or np.isinf(ref_std)) else None,
                        "min": float(np.min(ref_clean)) if len(ref_clean) > 0 and not (np.isnan(np.min(ref_clean)) or np.isinf(np.min(ref_clean))) else None,
                        "max": float(np.max(ref_clean)) if len(ref_clean) > 0 and not (np.isnan(np.max(ref_clean)) or np.isinf(np.max(ref_clean))) else None,
                        "count": len(ref_clean)
                    },
                    "current": {
                        "mean": curr_mean if not (np.isnan(curr_mean) or np.isinf(curr_mean)) else None,
                        "std": curr_std if not (np.isnan(curr_std) or np.isinf(curr_std)) else None,
                        "min": float(np.min(curr_clean)) if len(curr_clean) > 0 and not (np.isnan(np.min(curr_clean)) or np.isinf(np.min(curr_clean))) else None,
                        "max": float(np.max(curr_clean)) if len(curr_clean) > 0 and not (np.isnan(np.max(curr_clean)) or np.isinf(np.max(curr_clean))) else None,
                        "count": len(curr_clean)
                    }
                }
            }
        except Exception as e:
            logger.error(f"Error detecting drift for feature {feature_name}: {e}", exc_info=True)
            return {
                "feature": feature_name,
                "drift_detected": False,
                "error": str(e)
            }
    
    def detect_concept_drift(
        self,
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray,
        reference_targets: Optional[np.ndarray] = None,
        current_targets: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Detect concept drift by comparing prediction distributions or model performance.
        
        Args:
            reference_predictions: Predictions on reference data
            current_predictions: Predictions on current data
            reference_targets: Actual targets for reference data (optional, for performance-based detection)
            current_targets: Actual targets for current data (optional, for performance-based detection)
            
        Returns:
            Dictionary with concept drift detection results
        """
        try:
            results = {
                "concept_drift_detected": False,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "methods": {}
            }
            
            # Method 1: Compare prediction distributions
            if len(reference_predictions) > 0 and len(current_predictions) > 0:
                pred_drift = self._detect_feature_drift(
                    reference_predictions,
                    current_predictions,
                    "predictions"
                )
                results["methods"]["prediction_distribution"] = pred_drift
                
                if pred_drift.get("drift_detected", False):
                    results["concept_drift_detected"] = True
            
            # Method 2: Compare model performance (if targets available)
            if reference_targets is not None and current_targets is not None:
                from sklearn.metrics import mean_squared_error, accuracy_score
                
                ref_rmse = np.sqrt(mean_squared_error(reference_targets, reference_predictions))
                curr_rmse = np.sqrt(mean_squared_error(current_targets, current_predictions))
                
                # Performance degradation threshold (20% increase in RMSE)
                performance_degradation = (curr_rmse - ref_rmse) / ref_rmse if ref_rmse > 0 else 0
                
                if performance_degradation > 0.2:
                    results["concept_drift_detected"] = True
                    results["methods"]["performance_degradation"] = {
                        "drift_detected": True,
                        "reference_rmse": float(ref_rmse),
                        "current_rmse": float(curr_rmse),
                        "degradation_percent": float(performance_degradation * 100)
                    }
            
            return results
        except Exception as e:
            logger.error(f"Error detecting concept drift: {e}", exc_info=True)
            return {
                "concept_drift_detected": False,
                "error": str(e)
            }

