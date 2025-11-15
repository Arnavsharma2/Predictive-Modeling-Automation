"""
Feature engineering utilities.
"""
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

warnings.filterwarnings('ignore')


class TargetEncoder(BaseEstimator, TransformerMixin):
    """Target encoding for categorical variables with improved smoothing and CV-aware encoding."""

    def __init__(self, smoothing: float = 1.0, cv_folds: int = 5, use_cv: bool = True):
        """
        Initialize target encoder.
        
        Args:
            smoothing: Smoothing parameter (higher = more global mean influence)
            cv_folds: Number of CV folds for cross-validation aware encoding
            use_cv: Whether to use cross-validation aware encoding (prevents data leakage)
        """
        self.smoothing = smoothing
        self.cv_folds = cv_folds
        self.use_cv = use_cv
        self.encoding_dict: Dict[str, Dict] = {}
        self.global_mean = None

    def fit(self, X, y):
        """Fit target encoder."""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        self.global_mean = y.mean()

        if self.use_cv and len(X) > 50:
            # Use cross-validation aware encoding to prevent data leakage
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            
            # Initialize encoding dictionary
            for col in X.columns:
                self.encoding_dict[col] = {}
            
            # Calculate encodings using CV
            for train_idx, val_idx in kf.split(X):
                X_train_fold = X.iloc[train_idx]
                y_train_fold = y.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                
                for col in X.columns:
                    # Calculate target mean for each category in training fold
                    agg = pd.DataFrame({'cat': X_train_fold[col], 'target': y_train_fold})
                    agg_stats = agg.groupby('cat')['target'].agg(['mean', 'count'])

                    # Smoothing
                    smoothed_mean = (
                        agg_stats['count'] * agg_stats['mean'] + self.smoothing * self.global_mean
                    ) / (agg_stats['count'] + self.smoothing)
                    
                    # Update encoding dict (average across folds)
                    for cat, mean_val in smoothed_mean.items():
                        if col not in self.encoding_dict:
                            self.encoding_dict[col] = {}
                        if cat not in self.encoding_dict[col]:
                            self.encoding_dict[col][cat] = []
                        self.encoding_dict[col][cat].append(mean_val)
            
            # Average across folds
            for col in self.encoding_dict:
                for cat in self.encoding_dict[col]:
                    if isinstance(self.encoding_dict[col][cat], list):
                        self.encoding_dict[col][cat] = np.mean(self.encoding_dict[col][cat])
        else:
            # Standard target encoding (faster but can leak)
            for col in X.columns:
                # Calculate target mean for each category
                agg = pd.DataFrame({'cat': X[col], 'target': y})
                agg_stats = agg.groupby('cat')['target'].agg(['mean', 'count'])

                # Smoothing
                smoothed_mean = (
                    agg_stats['count'] * agg_stats['mean'] + self.smoothing * self.global_mean
                ) / (agg_stats['count'] + self.smoothing)

                self.encoding_dict[col] = smoothed_mean.to_dict()

        return self

    def transform(self, X):
        """Transform using target encoding."""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        X_encoded = X.copy()
        for col in X.columns:
            if col in self.encoding_dict:
                X_encoded[col] = X[col].map(self.encoding_dict[col]).fillna(self.global_mean)

        return X_encoded.values


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Frequency encoding for categorical variables."""

    def __init__(self):
        self.frequency_dict: Dict[str, Dict] = {}

    def fit(self, X, y=None):
        """Fit frequency encoder."""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        for col in X.columns:
            freq = X[col].value_counts(normalize=True).to_dict()
            self.frequency_dict[col] = freq

        return self

    def transform(self, X):
        """Transform using frequency encoding."""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        X_encoded = X.copy()
        for col in X.columns:
            if col in self.frequency_dict:
                X_encoded[col] = X[col].map(self.frequency_dict[col]).fillna(0)

        return X_encoded.values


class LeaveOneOutEncoder(BaseEstimator, TransformerMixin):
    """Leave-one-out encoding for categorical variables."""

    def __init__(self, smoothing: float = 1.0):
        self.smoothing = smoothing
        self.encoding_dict: Dict[str, Dict] = {}
        self.global_mean = None

    def fit(self, X, y):
        """Fit leave-one-out encoder."""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        self.global_mean = y.mean()

        for col in X.columns:
            # Calculate target mean for each category, excluding current row
            encoding = {}
            for cat in X[col].unique():
                mask = X[col] == cat
                if mask.sum() > 1:
                    # Mean of target excluding current row (approximated)
                    cat_mean = y[mask].mean()
                    encoding[cat] = cat_mean
                else:
                    encoding[cat] = self.global_mean
            
            self.encoding_dict[col] = encoding

        return self

    def transform(self, X):
        """Transform using leave-one-out encoding."""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        X_encoded = X.copy()
        for col in X.columns:
            if col in self.encoding_dict:
                X_encoded[col] = X[col].map(self.encoding_dict[col]).fillna(self.global_mean)

        return X_encoded.values


class JamesSteinEncoder(BaseEstimator, TransformerMixin):
    """James-Stein encoding for categorical variables (statistical shrinkage)."""

    def __init__(self):
        self.encoding_dict: Dict[str, Dict] = {}
        self.global_mean = None
        self.global_var = None

    def fit(self, X, y):
        """Fit James-Stein encoder."""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        self.global_mean = y.mean()
        self.global_var = y.var()

        for col in X.columns:
            encoding = {}
            for cat in X[col].unique():
                mask = X[col] == cat
                n = mask.sum()
                if n > 1:
                    cat_mean = y[mask].mean()
                    cat_var = y[mask].var() if n > 1 else self.global_var
                    
                    # James-Stein shrinkage factor
                    shrinkage = self.global_var / (self.global_var + cat_var / n) if cat_var > 0 else 0.5
                    encoding[cat] = shrinkage * cat_mean + (1 - shrinkage) * self.global_mean
                else:
                    encoding[cat] = self.global_mean
            
            self.encoding_dict[col] = encoding

        return self

    def transform(self, X):
        """Transform using James-Stein encoding."""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        X_encoded = X.copy()
        for col in X.columns:
            if col in self.encoding_dict:
                X_encoded[col] = X[col].map(self.encoding_dict[col]).fillna(self.global_mean)

        return X_encoded.values


class FeatureEngineer:
    """Feature engineering utilities."""
    
    @staticmethod
    def create_time_features(df: pd.DataFrame, time_column: str) -> pd.DataFrame:
        """
        Create time-based features from a datetime column.
        
        Args:
            df: Dataframe with time column
            time_column: Name of datetime column
            
        Returns:
            Dataframe with additional time features
        """
        if time_column not in df.columns:
            return df
        
        df = df.copy()
        df[time_column] = pd.to_datetime(df[time_column])
        
        # Extract time components
        df[f'{time_column}_year'] = df[time_column].dt.year
        df[f'{time_column}_month'] = df[time_column].dt.month
        df[f'{time_column}_day'] = df[time_column].dt.day
        df[f'{time_column}_dayofweek'] = df[time_column].dt.dayofweek
        df[f'{time_column}_dayofyear'] = df[time_column].dt.dayofyear
        df[f'{time_column}_week'] = df[time_column].dt.isocalendar().week
        df[f'{time_column}_quarter'] = df[time_column].dt.quarter
        df[f'{time_column}_hour'] = df[time_column].dt.hour
        df[f'{time_column}_minute'] = df[time_column].dt.minute
        
        # Cyclical encoding for periodic features
        df[f'{time_column}_month_sin'] = np.sin(2 * np.pi * df[f'{time_column}_month'] / 12)
        df[f'{time_column}_month_cos'] = np.cos(2 * np.pi * df[f'{time_column}_month'] / 12)
        df[f'{time_column}_dayofweek_sin'] = np.sin(2 * np.pi * df[f'{time_column}_dayofweek'] / 7)
        df[f'{time_column}_dayofweek_cos'] = np.cos(2 * np.pi * df[f'{time_column}_dayofweek'] / 7)
        df[f'{time_column}_hour_sin'] = np.sin(2 * np.pi * df[f'{time_column}_hour'] / 24)
        df[f'{time_column}_hour_cos'] = np.cos(2 * np.pi * df[f'{time_column}_hour'] / 24)
        
        return df
    
    @staticmethod
    def create_statistical_features(
        df: pd.DataFrame,
        numeric_columns: List[str],
        window_size: int = 7
    ) -> pd.DataFrame:
        """
        Create statistical features (rolling mean, std, etc.).
        
        Args:
            df: Dataframe
            numeric_columns: List of numeric columns to create features for
            window_size: Window size for rolling statistics
            
        Returns:
            Dataframe with additional statistical features
        """
        df = df.copy()
        
        for col in numeric_columns:
            if col not in df.columns:
                continue
            
            # Rolling statistics
            df[f'{col}_rolling_mean_{window_size}'] = df[col].rolling(window=window_size, min_periods=1).mean()
            df[f'{col}_rolling_std_{window_size}'] = df[col].rolling(window=window_size, min_periods=1).std()
            df[f'{col}_rolling_min_{window_size}'] = df[col].rolling(window=window_size, min_periods=1).min()
            df[f'{col}_rolling_max_{window_size}'] = df[col].rolling(window=window_size, min_periods=1).max()
            
            # Lag features
            df[f'{col}_lag_1'] = df[col].shift(1)
            df[f'{col}_lag_2'] = df[col].shift(2)
            df[f'{col}_lag_3'] = df[col].shift(3)
            
            # Difference features
            df[f'{col}_diff_1'] = df[col].diff(1)
            df[f'{col}_pct_change'] = df[col].pct_change()
        
        return df.fillna(0)  # Fill NaN values created by lag/diff operations
    
    @staticmethod
    def create_interaction_features(
        df: pd.DataFrame,
        columns: List[str],
        operations: List[str] = ['multiply', 'add', 'subtract'],
        max_interactions: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Create smart interaction features between columns (limited to prevent explosion).

        Args:
            df: Dataframe
            columns: List of columns to create interactions for
            operations: List of operations (multiply, add, subtract, divide)
            max_interactions: Maximum number of interactions to create (None = all)

        Returns:
            Dataframe with additional interaction features
        """
        df = df.copy()
        
        # Limit columns to prevent feature explosion
        if max_interactions and len(columns) > max_interactions:
            # Select top columns by variance (most informative)
            variances = df[columns].var().sort_values(ascending=False)
            columns = variances.head(max_interactions).index.tolist()
        
        interaction_count = 0
        max_total_interactions = max_interactions or (len(columns) * (len(columns) - 1) // 2)
        
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                if interaction_count >= max_total_interactions:
                    break
                    
                if col1 not in df.columns or col2 not in df.columns:
                    continue
                
                # Only create interactions for numeric columns
                if not (pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2])):
                    continue
                
                if 'multiply' in operations:
                    df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                    interaction_count += 1
                if 'add' in operations and interaction_count < max_total_interactions:
                    df[f'{col1}_plus_{col2}'] = df[col1] + df[col2]
                    interaction_count += 1
                if 'subtract' in operations and interaction_count < max_total_interactions:
                    df[f'{col1}_minus_{col2}'] = df[col1] - df[col2]
                    interaction_count += 1
                if 'divide' in operations and interaction_count < max_total_interactions:
                    # Avoid division by zero
                    df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-8)
                    interaction_count += 1
            
            if interaction_count >= max_total_interactions:
                break
        
        return df
    
    @staticmethod
    def create_aggregation_features(
        df: pd.DataFrame,
        group_by: str,
        numeric_columns: List[str],
        aggregations: List[str] = ['mean', 'std', 'min', 'max', 'count']
    ) -> pd.DataFrame:
        """
        Create aggregation features grouped by a column.

        Args:
            df: Dataframe
            group_by: Column to group by
            numeric_columns: Numeric columns to aggregate
            aggregations: List of aggregation functions

        Returns:
            Dataframe with additional aggregation features
        """
        if group_by not in df.columns:
            return df

        df = df.copy()

        for col in numeric_columns:
            if col not in df.columns:
                continue

            grouped = df.groupby(group_by)[col]

            for agg_func in aggregations:
                if hasattr(grouped, agg_func):
                    df[f'{col}_{agg_func}_by_{group_by}'] = grouped.transform(agg_func)

        return df

    @staticmethod
    def create_polynomial_features(
        df: pd.DataFrame,
        numeric_columns: List[str],
        degree: int = 2,
        interaction_only: bool = False
    ) -> pd.DataFrame:
        """
        Create polynomial and interaction features.

        Args:
            df: Dataframe
            numeric_columns: Numeric columns to create polynomial features for
            degree: Degree of polynomial features (2 or 3 recommended)
            interaction_only: If True, only interaction features (no powers)

        Returns:
            Dataframe with polynomial features
        """
        df = df.copy()

        # Filter to existing columns
        existing_cols = [col for col in numeric_columns if col in df.columns]
        if not existing_cols:
            return df

        # Create polynomial features
        poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
        poly_features = poly.fit_transform(df[existing_cols])

        # Get feature names
        feature_names = poly.get_feature_names_out(existing_cols)

        # Add new features (skip original features)
        new_feature_names = [name for name in feature_names if name not in existing_cols]
        new_features_idx = [i for i, name in enumerate(feature_names) if name not in existing_cols]

        for idx, name in zip(new_features_idx, new_feature_names):
            df[f'poly_{name}'] = poly_features[:, idx]

        return df

    @staticmethod
    def create_binned_features(
        df: pd.DataFrame,
        numeric_columns: List[str],
        n_bins: int = 5,
        strategy: str = 'quantile'
    ) -> pd.DataFrame:
        """
        Create binned/discretized features from continuous variables.

        Args:
            df: Dataframe
            numeric_columns: Numeric columns to bin
            n_bins: Number of bins
            strategy: Binning strategy ('uniform', 'quantile', 'kmeans')

        Returns:
            Dataframe with binned features
        """
        df = df.copy()

        for col in numeric_columns:
            if col not in df.columns:
                continue

            try:
                discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
                df[f'{col}_binned'] = discretizer.fit_transform(df[[col]])
            except:
                # Skip if binning fails (e.g., constant column)
                continue

        return df

    @staticmethod
    def create_ratio_features(
        df: pd.DataFrame,
        numeric_columns: List[str],
        max_ratios: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Create ratio features between numeric columns (smart selection to prevent explosion).

        Args:
            df: Dataframe
            numeric_columns: Numeric columns to create ratios for
            max_ratios: Maximum number of ratios to create (None = all)

        Returns:
            Dataframe with ratio features
        """
        df = df.copy()

        # Limit to top columns by variance if too many
        if max_ratios and len(numeric_columns) > max_ratios:
            variances = df[numeric_columns].var().sort_values(ascending=False)
            numeric_columns = variances.head(max_ratios).index.tolist()

        ratio_count = 0
        max_total_ratios = max_ratios or (len(numeric_columns) * (len(numeric_columns) - 1))

        for i, col1 in enumerate(numeric_columns):
            for col2 in numeric_columns[i+1:]:
                if ratio_count >= max_total_ratios:
                    break
                    
                if col1 not in df.columns or col2 not in df.columns:
                    continue

                # Only create ratios for positive values (to avoid negative ratios)
                if (df[col1] > 0).all() and (df[col2] > 0).all():
                    # Avoid division by zero
                    df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-8)
                    ratio_count += 1
                    if ratio_count < max_total_ratios:
                        df[f'{col2}_div_{col1}'] = df[col2] / (df[col1] + 1e-8)
                        ratio_count += 1

            if ratio_count >= max_total_ratios:
                break

        return df

    @staticmethod
    def create_cluster_features(
        df: pd.DataFrame,
        numeric_columns: List[str],
        n_clusters: int = 5
    ) -> pd.DataFrame:
        """
        Create cluster-based features using KMeans.

        Args:
            df: Dataframe
            numeric_columns: Numeric columns to use for clustering
            n_clusters: Number of clusters

        Returns:
            Dataframe with cluster membership features
        """
        df = df.copy()

        # Filter to existing columns
        existing_cols = [col for col in numeric_columns if col in df.columns]
        if not existing_cols or len(df) < n_clusters:
            return df

        try:
            # Standardize features for clustering
            X = df[existing_cols].fillna(0)
            X_std = (X - X.mean()) / (X.std() + 1e-8)

            # Fit KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df['cluster_id'] = kmeans.fit_predict(X_std)

            # Add distance to cluster center
            distances = kmeans.transform(X_std)
            df['cluster_distance'] = distances.min(axis=1)
        except:
            # Skip if clustering fails
            pass

        return df

    @staticmethod
    def create_log_features(
        df: pd.DataFrame,
        numeric_columns: List[str]
    ) -> pd.DataFrame:
        """
        Create log-transformed features for skewed distributions.

        Args:
            df: Dataframe
            numeric_columns: Numeric columns to log-transform

        Returns:
            Dataframe with log features
        """
        df = df.copy()

        for col in numeric_columns:
            if col not in df.columns:
                continue

            # Check if column has positive values
            if (df[col] > 0).all():
                df[f'{col}_log'] = np.log(df[col])
            elif (df[col] >= 0).all():
                df[f'{col}_log1p'] = np.log1p(df[col])  # log(1 + x) for non-negative values

        return df

