"""
Feature selection utilities for dimensionality reduction and model improvement.
"""
from typing import List, Optional, Literal, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
    chi2,
    RFE,
    RFECV,
    VarianceThreshold
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

warnings.filterwarnings('ignore')


class FeatureSelector:
    """Select most important features using various methods."""

    def __init__(
        self,
        method: Literal['univariate', 'model_based', 'rfe', 'variance', 'correlation'] = 'model_based',
        n_features: Optional[int] = None,
        task: Literal['classification', 'regression'] = 'regression',
        threshold: float = 0.01
    ):
        """
        Initialize feature selector.

        Args:
            method: Selection method
            n_features: Number of features to select (None = auto)
            task: Type of ML task
            threshold: Threshold for variance/correlation methods
        """
        self.method = method
        self.n_features = n_features
        self.task = task
        self.threshold = threshold
        self.selector = None
        self.selected_features = None
        self.feature_scores = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FeatureSelector':
        """
        Fit the feature selector.

        Args:
            X: Input features
            y: Target variable

        Returns:
            Self
        """
        X_values = X.values if isinstance(X, pd.DataFrame) else X
        y_values = y.values if isinstance(y, pd.Series) else y

        if self.method == 'univariate':
            # Univariate statistical tests
            if self.task == 'classification':
                # Use mutual information for classification
                score_func = mutual_info_classif
            else:
                # Use F-test for regression
                score_func = f_regression

            k = self.n_features if self.n_features else 'all'
            self.selector = SelectKBest(score_func=score_func, k=k)
            self.selector.fit(X_values, y_values)
            self.feature_scores = self.selector.scores_

            # Get selected features
            if self.n_features:
                feature_mask = self.selector.get_support()
                self.selected_features = X.columns[feature_mask].tolist() if isinstance(X, pd.DataFrame) else list(range(X_values.shape[1]))[feature_mask]
            else:
                # Select features above median score
                median_score = np.median(self.feature_scores)
                feature_mask = self.feature_scores > median_score
                self.selected_features = X.columns[feature_mask].tolist() if isinstance(X, pd.DataFrame) else list(np.where(feature_mask)[0])

        elif self.method == 'model_based':
            # Model-based feature importance
            if self.task == 'classification':
                model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)

            model.fit(X_values, y_values)
            self.feature_scores = model.feature_importances_

            # Select top features
            if self.n_features:
                top_indices = np.argsort(self.feature_scores)[-self.n_features:]
            else:
                # Select features above mean importance
                threshold = self.feature_scores.mean()
                top_indices = np.where(self.feature_scores > threshold)[0]

            if isinstance(X, pd.DataFrame):
                self.selected_features = X.columns[top_indices].tolist()
            else:
                self.selected_features = list(top_indices)

            self.selector = model

        elif self.method == 'rfe':
            # Recursive Feature Elimination
            if self.task == 'classification':
                estimator = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
            else:
                estimator = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)

            n_features_to_select = self.n_features if self.n_features else max(1, X_values.shape[1] // 2)

            self.selector = RFE(
                estimator=estimator,
                n_features_to_select=n_features_to_select,
                step=1
            )
            self.selector.fit(X_values, y_values)

            feature_mask = self.selector.get_support()
            if isinstance(X, pd.DataFrame):
                self.selected_features = X.columns[feature_mask].tolist()
            else:
                self.selected_features = list(np.where(feature_mask)[0])

            self.feature_scores = self.selector.ranking_

        elif self.method == 'variance':
            # Variance threshold
            self.selector = VarianceThreshold(threshold=self.threshold)
            self.selector.fit(X_values)

            feature_mask = self.selector.get_support()
            if isinstance(X, pd.DataFrame):
                self.selected_features = X.columns[feature_mask].tolist()
            else:
                self.selected_features = list(np.where(feature_mask)[0])

            self.feature_scores = self.selector.variances_

        elif self.method == 'correlation':
            # Correlation-based selection
            if isinstance(X, pd.DataFrame):
                # Calculate correlation matrix
                corr_matrix = X.corr().abs()

                # Find highly correlated features
                upper_triangle = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )

                # Drop features with correlation > threshold
                to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > (1 - self.threshold))]

                self.selected_features = [col for col in X.columns if col not in to_drop]
                self.feature_scores = None  # No scores for correlation method
            else:
                # For numpy arrays, use all features
                self.selected_features = list(range(X_values.shape[1]))

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by selecting features.

        Args:
            X: Input features

        Returns:
            Dataframe with selected features
        """
        if self.selected_features is None:
            raise ValueError("Selector must be fitted before transforming")

        if isinstance(X, pd.DataFrame):
            return X[self.selected_features]
        else:
            return X[:, self.selected_features]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance scores.

        Returns:
            Dataframe with feature names and scores
        """
        if self.feature_scores is None:
            return None

        return pd.DataFrame({
            'feature': self.selected_features,
            'score': self.feature_scores[self.selected_features] if isinstance(self.feature_scores, np.ndarray) else self.feature_scores
        }).sort_values('score', ascending=False)


class CombinedFeatureSelector(BaseEstimator, TransformerMixin):
    """Combine multiple feature selection methods."""

    def __init__(
        self,
        methods: List[str] = ['univariate', 'model_based'],
        task: Literal['classification', 'regression'] = 'regression',
        voting: Literal['union', 'intersection'] = 'intersection',
        n_features: Optional[int] = None
    ):
        """
        Initialize combined feature selector.

        Args:
            methods: List of methods to combine
            task: Type of ML task
            voting: How to combine results ('union' or 'intersection')
            n_features: Target number of features
        """
        self.methods = methods
        self.task = task
        self.voting = voting
        self.n_features = n_features
        self.selectors = []
        self.selected_features = None

    def fit(self, X, y):
        """Fit all selectors."""
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        y_series = pd.Series(y) if not isinstance(y, pd.Series) else y

        self.selectors = []
        all_selected = []

        for method in self.methods:
            selector = FeatureSelector(method=method, task=self.task, n_features=self.n_features)
            try:
                selector.fit(X_df, y_series)
                self.selectors.append(selector)
                all_selected.append(set(selector.selected_features))
            except Exception as e:
                print(f"Warning: {method} selection failed: {e}")
                continue

        # Combine results
        if not all_selected:
            raise ValueError("All feature selection methods failed")

        if self.voting == 'intersection':
            # Features selected by all methods
            self.selected_features = list(set.intersection(*all_selected))
        else:  # union
            # Features selected by any method
            self.selected_features = list(set.union(*all_selected))

        # If no features selected or too many, fall back to top n_features from first selector
        if not self.selected_features or (self.n_features and len(self.selected_features) > self.n_features * 2):
            self.selected_features = self.selectors[0].selected_features[:self.n_features] if self.n_features else self.selectors[0].selected_features

        return self

    def transform(self, X):
        """Transform by selecting features."""
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        return X_df[self.selected_features]

    def fit_transform(self, X, y):
        """Fit and transform."""
        self.fit(X, y)
        return self.transform(X)


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    method: str = 'model_based',
    n_features: Optional[int] = None,
    task: str = 'regression'
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Convenience function for feature selection.

    Args:
        X: Input features
        y: Target variable
        method: Selection method
        n_features: Number of features to select
        task: Type of ML task

    Returns:
        Tuple of (selected features dataframe, list of selected feature names)
    """
    selector = FeatureSelector(method=method, n_features=n_features, task=task)
    X_selected = selector.fit_transform(X, y)
    return X_selected, selector.selected_features
