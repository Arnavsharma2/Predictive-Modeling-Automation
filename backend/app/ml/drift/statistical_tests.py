"""
Statistical tests for drift detection.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from app.core.logging import get_logger

logger = get_logger(__name__)


class StatisticalTests:
    """Statistical tests for detecting data drift."""
    
    @staticmethod
    def calculate_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI) for a single feature.
        
        PSI measures the change in distribution of a variable between two samples.
        PSI < 0.1: No significant change
        0.1 <= PSI < 0.25: Minor change
        PSI >= 0.25: Significant change
        
        Args:
            expected: Reference distribution (training data)
            actual: Current distribution (production data)
            buckets: Number of bins for discretization
            
        Returns:
            PSI value
        """
        try:
            # Remove NaN values
            expected = expected[~np.isnan(expected)]
            actual = actual[~np.isnan(actual)]
            
            if len(expected) == 0 or len(actual) == 0:
                return np.nan
            
            # Create bins based on expected distribution
            min_val = min(np.min(expected), np.min(actual))
            max_val = max(np.max(expected), np.max(actual))
            
            # Handle edge case where all values are the same
            if min_val == max_val:
                return 0.0
            
            bins = np.linspace(min_val, max_val, buckets + 1)
            bins[0] = -np.inf
            bins[-1] = np.inf
            
            # Calculate expected and actual distributions
            expected_counts, _ = np.histogram(expected, bins=bins)
            actual_counts, _ = np.histogram(actual, bins=bins)
            
            # Normalize to probabilities
            expected_probs = expected_counts / len(expected)
            actual_probs = actual_counts / len(actual)
            
            # Add small epsilon to avoid division by zero
            epsilon = 1e-6
            expected_probs = expected_probs + epsilon
            actual_probs = actual_probs + epsilon
            
            # Normalize again
            expected_probs = expected_probs / expected_probs.sum()
            actual_probs = actual_probs / actual_probs.sum()
            
            # Calculate PSI
            psi = np.sum((actual_probs - expected_probs) * np.log(actual_probs / expected_probs))
            
            return float(psi)
        except Exception as e:
            logger.error(f"Error calculating PSI: {e}")
            return np.nan
    
    @staticmethod
    def kolmogorov_smirnov_test(expected: np.ndarray, actual: np.ndarray) -> Tuple[float, float]:
        """
        Perform Kolmogorov-Smirnov test for distribution comparison.
        
        Args:
            expected: Reference distribution
            actual: Current distribution
            
        Returns:
            Tuple of (statistic, p-value)
        """
        try:
            expected = expected[~np.isnan(expected)]
            actual = actual[~np.isnan(actual)]
            
            if len(expected) == 0 or len(actual) == 0:
                return (np.nan, np.nan)
            
            statistic, p_value = stats.ks_2samp(expected, actual)
            return (float(statistic), float(p_value))
        except Exception as e:
            logger.error(f"Error in KS test: {e}")
            return (np.nan, np.nan)
    
    @staticmethod
    def chi_square_test(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> Tuple[float, float]:
        """
        Perform Chi-square test for categorical or discretized features.
        
        Args:
            expected: Reference distribution
            actual: Current distribution
            bins: Number of bins for discretization (for continuous features)
            
        Returns:
            Tuple of (chi-square statistic, p-value)
        """
        try:
            expected = expected[~np.isnan(expected)]
            actual = actual[~np.isnan(actual)]
            
            if len(expected) == 0 or len(actual) == 0:
                return (np.nan, np.nan)
            
            # Check if data is categorical (string/object type)
            if expected.dtype == 'object' or actual.dtype == 'object':
                # Categorical data
                expected_unique = pd.Series(expected).value_counts()
                actual_unique = pd.Series(actual).value_counts()
                
                # Get all unique values
                all_values = set(expected_unique.index) | set(actual_unique.index)
                
                # Create aligned frequency arrays
                expected_freq = [expected_unique.get(val, 0) for val in all_values]
                actual_freq = [actual_unique.get(val, 0) for val in all_values]
            else:
                # Continuous data - discretize
                min_val = min(np.min(expected), np.min(actual))
                max_val = max(np.max(expected), np.max(actual))
                
                if min_val == max_val:
                    return (0.0, 1.0)
                
                bin_edges = np.linspace(min_val, max_val, bins + 1)
                bin_edges[0] = -np.inf
                bin_edges[-1] = np.inf
                
                expected_counts, _ = np.histogram(expected, bins=bin_edges)
                actual_counts, _ = np.histogram(actual, bins=bin_edges)
                
                expected_freq = expected_counts.tolist()
                actual_freq = actual_counts.tolist()
            
            # Perform chi-square test
            statistic, p_value = stats.chisquare(actual_freq, f_exp=expected_freq)
            
            return (float(statistic), float(p_value))
        except Exception as e:
            logger.error(f"Error in chi-square test: {e}")
            return (np.nan, np.nan)
    
    @staticmethod
    def wasserstein_distance(expected: np.ndarray, actual: np.ndarray) -> float:
        """
        Calculate Wasserstein distance (Earth Mover's Distance) between distributions.
        
        Args:
            expected: Reference distribution
            actual: Current distribution
            
        Returns:
            Wasserstein distance
        """
        try:
            expected = expected[~np.isnan(expected)]
            actual = actual[~np.isnan(actual)]
            
            if len(expected) == 0 or len(actual) == 0:
                return np.nan
            
            # Use scipy's wasserstein_distance
            distance = stats.wasserstein_distance(expected, actual)
            return float(distance)
        except Exception as e:
            logger.error(f"Error calculating Wasserstein distance: {e}")
            return np.nan
    
    @staticmethod
    def jensen_shannon_divergence(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
        """
        Calculate Jensen-Shannon divergence between distributions.
        
        Args:
            expected: Reference distribution
            actual: Current distribution
            bins: Number of bins for discretization
            
        Returns:
            JS divergence (0 to 1, where 0 = identical, 1 = completely different)
        """
        try:
            from scipy.spatial.distance import jensenshannon
            
            expected = expected[~np.isnan(expected)]
            actual = actual[~np.isnan(actual)]
            
            if len(expected) == 0 or len(actual) == 0:
                return np.nan
            
            # Discretize continuous data
            min_val = min(np.min(expected), np.min(actual))
            max_val = max(np.max(expected), np.max(actual))
            
            if min_val == max_val:
                return 0.0
            
            bin_edges = np.linspace(min_val, max_val, bins + 1)
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf
            
            expected_hist, _ = np.histogram(expected, bins=bin_edges)
            actual_hist, _ = np.histogram(actual, bins=bin_edges)
            
            # Normalize to probabilities
            expected_probs = expected_hist / expected_hist.sum() if expected_hist.sum() > 0 else expected_hist
            actual_probs = actual_hist / actual_hist.sum() if actual_hist.sum() > 0 else actual_hist
            
            # Add small epsilon to avoid zero probabilities
            epsilon = 1e-10
            expected_probs = expected_probs + epsilon
            actual_probs = actual_probs + epsilon
            expected_probs = expected_probs / expected_probs.sum()
            actual_probs = actual_probs / actual_probs.sum()
            
            # Calculate JS divergence
            js_div = jensenshannon(expected_probs, actual_probs)
            
            return float(js_div)
        except Exception as e:
            logger.error(f"Error calculating JS divergence: {e}")
            return np.nan

