"""
Feature importance analysis module.

Provides various methods for calculating and analyzing feature importance:
- Model-based importance (tree models, linear models)
- Permutation importance
- Drop-column importance
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from sklearn.inspection import permutation_importance
import joblib
from pathlib import Path


class FeatureImportanceAnalyzer:
    """
    Analyzer for feature importance using multiple methods.
    """

    def __init__(
        self,
        model_path: str,
        feature_names: Optional[List[str]] = None
    ):
        """
        Initialize feature importance analyzer.

        Args:
            model_path: Path to the trained model
            feature_names: List of feature names
        """
        self.model_path = Path(model_path)
        self.model = self._load_model()
        self.feature_names = feature_names

    def _load_model(self):
        """Load the trained model from disk."""
        return joblib.load(self.model_path)

    def get_model_importance(self) -> Optional[Dict[str, float]]:
        """
        Get built-in feature importance from the model (if available).

        Works for:
        - Tree-based models (Random Forest, XGBoost, LightGBM, CatBoost)
        - Linear models with coefficients

        Returns:
            Dictionary mapping feature names to importance scores,
            or None if model doesn't support built-in importance
        """
        importance_values = None

        # Tree-based models
        if hasattr(self.model, 'feature_importances_'):
            importance_values = self.model.feature_importances_

        # Linear models
        elif hasattr(self.model, 'coef_'):
            # Use absolute values of coefficients
            coef = self.model.coef_
            if coef.ndim > 1:
                # Multi-class: average importance across classes
                importance_values = np.mean(np.abs(coef), axis=0)
            else:
                importance_values = np.abs(coef)

        if importance_values is None:
            return None

        # Create feature names if not provided
        if self.feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importance_values))]
        else:
            feature_names = self.feature_names

        # Create and sort dictionary
        importance_dict = dict(zip(feature_names, importance_values.tolist()))
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

    def get_permutation_importance(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        n_repeats: int = 10,
        random_state: int = 42,
        scoring: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate permutation importance.

        Permutation importance measures the decrease in model performance
        when a single feature's values are randomly shuffled.

        Args:
            X: Feature matrix
            y: Target values
            n_repeats: Number of times to permute each feature
            random_state: Random seed
            scoring: Scoring metric (None = use default)

        Returns:
            Dictionary containing:
            - importance_mean: Mean importance for each feature
            - importance_std: Standard deviation of importance
        """
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            if self.feature_names is None:
                self.feature_names = X.columns.tolist()
            X_array = X.values
        else:
            X_array = X

        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y

        # Calculate permutation importance
        result = permutation_importance(
            self.model,
            X_array,
            y_array,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring=scoring
        )

        # Create feature names if not provided
        if self.feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X_array.shape[1])]
        else:
            feature_names = self.feature_names

        # Create dictionaries
        importance_mean = dict(zip(feature_names, result.importances_mean.tolist()))
        importance_std = dict(zip(feature_names, result.importances_std.tolist()))

        # Sort by mean importance
        importance_mean = dict(sorted(importance_mean.items(), key=lambda x: x[1], reverse=True))

        return {
            'importance_mean': importance_mean,
            'importance_std': importance_std,
            'feature_names': feature_names
        }

    def get_drop_column_importance(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        scoring_func: callable,
        n_jobs: int = -1
    ) -> Dict[str, float]:
        """
        Calculate drop-column importance.

        Measures the decrease in model performance when each feature is removed.
        More accurate but computationally expensive than permutation importance.

        Args:
            X: Feature matrix
            y: Target values
            scoring_func: Function to score model (higher is better)
            n_jobs: Number of parallel jobs

        Returns:
            Dictionary mapping feature names to importance scores
        """
        from sklearn.base import clone

        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            if self.feature_names is None:
                self.feature_names = X.columns.tolist()
            X_array = X.values
        else:
            X_array = X

        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y

        # Get baseline score
        baseline_score = scoring_func(self.model, X_array, y_array)

        # Create feature names if not provided
        if self.feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X_array.shape[1])]
        else:
            feature_names = self.feature_names

        # Calculate importance for each feature
        importance_dict = {}

        for i, feature_name in enumerate(feature_names):
            # Create dataset without this feature
            X_dropped = np.delete(X_array, i, axis=1)

            # Clone and retrain model
            model_clone = clone(self.model)
            model_clone.fit(X_dropped, y_array)

            # Get score without this feature
            score_without = scoring_func(model_clone, X_dropped, y_array)

            # Importance is the decrease in score
            importance_dict[feature_name] = baseline_score - score_without

        # Sort by importance
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

    def get_comprehensive_importance(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        include_permutation: bool = True,
        n_repeats: int = 10
    ) -> Dict[str, Dict[str, float]]:
        """
        Get comprehensive feature importance using multiple methods.

        Args:
            X: Feature matrix
            y: Target values
            include_permutation: Whether to include permutation importance
            n_repeats: Number of repeats for permutation importance

        Returns:
            Dictionary containing importance from different methods:
            - model_importance: Built-in model importance (if available)
            - permutation_importance: Permutation-based importance
        """
        results = {}

        # Model-based importance
        model_imp = self.get_model_importance()
        if model_imp is not None:
            results['model_importance'] = model_imp

        # Permutation importance
        if include_permutation:
            perm_imp = self.get_permutation_importance(X, y, n_repeats=n_repeats)
            results['permutation_importance'] = perm_imp['importance_mean']
            results['permutation_std'] = perm_imp['importance_std']

        return results

    def get_top_features(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        n_features: int = 10,
        method: str = 'model'
    ) -> List[str]:
        """
        Get top N most important features.

        Args:
            X: Feature matrix
            y: Target values
            n_features: Number of top features to return
            method: 'model' or 'permutation'

        Returns:
            List of top feature names
        """
        if method == 'model':
            importance = self.get_model_importance()
            if importance is None:
                # Fall back to permutation if model importance not available
                method = 'permutation'

        if method == 'permutation':
            importance_result = self.get_permutation_importance(X, y)
            importance = importance_result['importance_mean']

        # Get top N features
        top_features = list(importance.keys())[:n_features]
        return top_features
