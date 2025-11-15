"""
Outlier detection and handling utilities.
"""
from typing import List, Optional, Literal
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class OutlierDetector:
    """Detect and handle outliers in data."""

    def __init__(
        self,
        method: Literal['iqr', 'zscore', 'isolation_forest', 'lof', 'elliptic'] = 'iqr',
        threshold: float = 3.0,
        contamination: float = 0.1
    ):
        """
        Initialize outlier detector.

        Args:
            method: Detection method
            threshold: Threshold for IQR or Z-score methods
            contamination: Expected proportion of outliers (for ML methods)
        """
        self.method = method
        self.threshold = threshold
        self.contamination = contamination
        self.detector = None
        self.scaler = None
        self.feature_stats = {}

    def fit(self, X: pd.DataFrame, numeric_columns: Optional[List[str]] = None) -> 'OutlierDetector':
        """
        Fit the outlier detector.

        Args:
            X: Input dataframe
            numeric_columns: List of numeric columns to check for outliers

        Returns:
            Self
        """
        if numeric_columns is None:
            numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            # Filter out datetime and non-numeric columns
            numeric_columns = [
                col for col in numeric_columns 
                if col in X.columns and pd.api.types.is_numeric_dtype(X[col])
            ]

        X_numeric = X[numeric_columns]

        if self.method == 'iqr':
            # Calculate IQR statistics
            for col in numeric_columns:
                q1 = X_numeric[col].quantile(0.25)
                q3 = X_numeric[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - self.threshold * iqr
                upper_bound = q3 + self.threshold * iqr
                self.feature_stats[col] = {
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }

        elif self.method == 'zscore':
            # Calculate mean and std
            for col in numeric_columns:
                mean = X_numeric[col].mean()
                std = X_numeric[col].std()
                self.feature_stats[col] = {
                    'mean': mean,
                    'std': std,
                    'lower_bound': mean - self.threshold * std,
                    'upper_bound': mean + self.threshold * std
                }

        elif self.method == 'isolation_forest':
            # Fit Isolation Forest
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_numeric.fillna(0))
            self.detector = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
            self.detector.fit(X_scaled)

        elif self.method == 'lof':
            # Fit Local Outlier Factor
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_numeric.fillna(0))
            self.detector = LocalOutlierFactor(
                contamination=self.contamination,
                novelty=True  # For prediction on new data
            )
            self.detector.fit(X_scaled)

        elif self.method == 'elliptic':
            # Fit Elliptic Envelope
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_numeric.fillna(0))
            self.detector = EllipticEnvelope(
                contamination=self.contamination,
                random_state=42
            )
            self.detector.fit(X_scaled)

        return self

    def detect(self, X: pd.DataFrame, numeric_columns: Optional[List[str]] = None) -> pd.Series:
        """
        Detect outliers in data.

        Args:
            X: Input dataframe
            numeric_columns: List of numeric columns to check for outliers

        Returns:
            Boolean series indicating outliers (True = outlier)
        """
        if numeric_columns is None:
            numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            # Filter out datetime and non-numeric columns
            numeric_columns = [
                col for col in numeric_columns 
                if col in X.columns and pd.api.types.is_numeric_dtype(X[col])
            ]

        X_numeric = X[numeric_columns]

        if self.method in ['iqr', 'zscore']:
            # Detect using statistical methods
            outlier_mask = pd.Series([False] * len(X), index=X.index)

            for col in numeric_columns:
                if col not in self.feature_stats:
                    continue

                lower = self.feature_stats[col]['lower_bound']
                upper = self.feature_stats[col]['upper_bound']
                outlier_mask |= (X_numeric[col] < lower) | (X_numeric[col] > upper)

            return outlier_mask

        elif self.method in ['isolation_forest', 'lof', 'elliptic']:
            # Detect using ML methods
            X_scaled = self.scaler.transform(X_numeric.fillna(0))
            predictions = self.detector.predict(X_scaled)
            # -1 for outliers, 1 for inliers
            return pd.Series(predictions == -1, index=X.index)

        return pd.Series([False] * len(X), index=X.index)

    def handle_outliers(
        self,
        X: pd.DataFrame,
        strategy: Literal['remove', 'cap', 'flag', 'transform'] = 'cap',
        numeric_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Handle outliers in data.

        Args:
            X: Input dataframe
            strategy: Handling strategy ('remove', 'cap', 'flag', 'transform')
            numeric_columns: List of numeric columns to handle

        Returns:
            Dataframe with outliers handled
        """
        if numeric_columns is None:
            numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            # Filter out datetime and non-numeric columns
            numeric_columns = [
                col for col in numeric_columns 
                if col in X.columns and pd.api.types.is_numeric_dtype(X[col])
            ]

        X_result = X.copy()

        if strategy == 'remove':
            # Remove outlier rows
            outlier_mask = self.detect(X, numeric_columns)
            X_result = X_result[~outlier_mask]

        elif strategy == 'cap':
            # Cap outliers to bounds (winsorization)
            if self.method in ['iqr', 'zscore']:
                for col in numeric_columns:
                    if col not in self.feature_stats:
                        continue
                    # Skip datetime columns
                    if not pd.api.types.is_numeric_dtype(X_result[col]):
                        continue

                    lower = self.feature_stats[col]['lower_bound']
                    upper = self.feature_stats[col]['upper_bound']
                    X_result[col] = X_result[col].clip(lower=lower, upper=upper)
            else:
                # For ML methods, cap to percentiles
                for col in numeric_columns:
                    # Skip datetime columns
                    if not pd.api.types.is_numeric_dtype(X_result[col]):
                        continue
                    lower = X_result[col].quantile(self.contamination / 2)
                    upper = X_result[col].quantile(1 - self.contamination / 2)
                    X_result[col] = X_result[col].clip(lower=lower, upper=upper)

        elif strategy == 'flag':
            # Add binary flag column for outliers
            outlier_mask = self.detect(X, numeric_columns)
            X_result['is_outlier'] = outlier_mask.astype(int)

        elif strategy == 'transform':
            # Apply log transformation to reduce skewness
            for col in numeric_columns:
                # Skip datetime columns
                if not pd.api.types.is_numeric_dtype(X_result[col]):
                    continue
                if (X_result[col] > 0).all():
                    X_result[f'{col}_transformed'] = np.log(X_result[col])
                elif (X_result[col] >= 0).all():
                    X_result[f'{col}_transformed'] = np.log1p(X_result[col])
                else:
                    # For negative values, use sqrt of absolute value with sign
                    X_result[f'{col}_transformed'] = np.sign(X_result[col]) * np.sqrt(np.abs(X_result[col]))

        return X_result

    def fit_detect_handle(
        self,
        X: pd.DataFrame,
        strategy: Literal['remove', 'cap', 'flag', 'transform'] = 'cap',
        numeric_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fit, detect, and handle outliers in one step.

        Args:
            X: Input dataframe
            strategy: Handling strategy
            numeric_columns: List of numeric columns

        Returns:
            Dataframe with outliers handled
        """
        self.fit(X, numeric_columns)
        return self.handle_outliers(X, strategy, numeric_columns)


def detect_and_handle_outliers(
    df: pd.DataFrame,
    numeric_columns: Optional[List[str]] = None,
    method: str = 'iqr',
    strategy: str = 'cap',
    threshold: float = 3.0
) -> pd.DataFrame:
    """
    Convenience function to detect and handle outliers.

    Args:
        df: Input dataframe
        numeric_columns: List of numeric columns (None = all numeric)
        method: Detection method ('iqr', 'zscore', 'isolation_forest', 'lof')
        strategy: Handling strategy ('remove', 'cap', 'flag', 'transform')
        threshold: Threshold for statistical methods

    Returns:
        Dataframe with outliers handled
    """
    detector = OutlierDetector(method=method, threshold=threshold)
    return detector.fit_detect_handle(df, strategy=strategy, numeric_columns=numeric_columns)
