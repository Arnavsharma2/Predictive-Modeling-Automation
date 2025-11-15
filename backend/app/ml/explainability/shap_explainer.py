"""
SHAP (SHapley Additive exPlanations) explainer for model predictions.

SHAP provides a unified measure of feature importance based on game theory,
offering both global and local explanations for model predictions.
"""
import shap
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
import joblib
from pathlib import Path


class ShapExplainer:
    """
    SHAP explainer for machine learning models.

    Supports various model types:
    - Tree-based models (XGBoost, LightGBM, CatBoost, Random Forest)
    - Linear models
    - General models (using KernelExplainer)
    """

    def __init__(self, model_path: str, feature_names: Optional[List[str]] = None):
        """
        Initialize SHAP explainer.

        Args:
            model_path: Path to the trained model
            feature_names: List of feature names (optional, will be loaded from model if available)
        """
        self.model_path = Path(model_path)
        self.model = self._load_model()
        self.feature_names = feature_names
        self.explainer = None
        self._initialize_explainer()

    def _load_model(self):
        """Load the trained model from disk."""
        return joblib.load(self.model_path)

    def _initialize_explainer(self):
        """Initialize the appropriate SHAP explainer based on model type."""
        model_type = type(self.model).__name__

        # Tree-based models
        if 'XGB' in model_type or 'LGB' in model_type or 'CatBoost' in model_type:
            self.explainer = shap.TreeExplainer(self.model)
        # Random Forest and other sklearn tree models
        elif 'Forest' in model_type or 'Tree' in model_type:
            self.explainer = shap.TreeExplainer(self.model)
        # Linear models
        elif 'Linear' in model_type or 'Ridge' in model_type or 'Lasso' in model_type:
            self.explainer = shap.LinearExplainer(self.model, shap.maskers.Independent(None))
        # General models (slower but works for any model)
        else:
            # For general models, we'll need a background dataset
            # This will be set later when explain() is called
            self.explainer = None

    def explain(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        background_data: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        max_evals: int = 100
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanations for the given data.

        Args:
            X: Input data to explain (single instance or multiple instances)
            background_data: Background dataset for KernelExplainer (optional)
            max_evals: Maximum evaluations for KernelExplainer

        Returns:
            Dictionary containing SHAP values and related information
        """
        # Convert to numpy if pandas DataFrame
        if isinstance(X, pd.DataFrame):
            if self.feature_names is None:
                self.feature_names = X.columns.tolist()
            X_array = X.values
        else:
            X_array = X

        # Initialize KernelExplainer if needed
        if self.explainer is None:
            if background_data is None:
                raise ValueError("background_data required for KernelExplainer")
            if isinstance(background_data, pd.DataFrame):
                background_data = background_data.values
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
                background_data,
                max_evals=max_evals
            )

        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X_array)

        # Handle multi-class classification
        if isinstance(shap_values, list):
            # For binary classification, use positive class
            shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]

        # Safely extract base_value
        base_val = self.explainer.expected_value
        if isinstance(base_val, np.ndarray):
            if base_val.ndim == 0:
                base_val = float(base_val.item())
            elif len(base_val) > 0:
                # For multi-class, use the first class or average
                base_val = float(base_val[0] if len(base_val) == 1 else np.mean(base_val))
            else:
                base_val = 0.0
        elif isinstance(base_val, (list, tuple)):
            base_val = float(base_val[0] if len(base_val) == 1 else np.mean(base_val))
        else:
            base_val = float(base_val)

        return {
            'shap_values': shap_values.tolist() if isinstance(shap_values, np.ndarray) else shap_values,
            'base_values': base_val,
            'feature_names': self.feature_names,
            'data': X_array.tolist()
        }

    def explain_instance(
        self,
        instance: Union[pd.DataFrame, np.ndarray, List],
        background_data: Optional[Union[pd.DataFrame, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanation for a single instance.

        Args:
            instance: Single instance to explain
            background_data: Background dataset for KernelExplainer (optional)

        Returns:
            Dictionary containing:
            - shap_values: SHAP values for each feature
            - base_value: Expected value (baseline prediction)
            - prediction: Model prediction for this instance
            - feature_contributions: List of (feature_name, shap_value) tuples sorted by importance
        """
        # Convert instance to proper format
        if isinstance(instance, list):
            instance = np.array(instance).reshape(1, -1)
        elif isinstance(instance, pd.DataFrame):
            if self.feature_names is None:
                self.feature_names = instance.columns.tolist()
            instance_array = instance.values
        elif isinstance(instance, np.ndarray):
            if instance.ndim == 1:
                instance_array = instance.reshape(1, -1)
            else:
                instance_array = instance

        # Get explanation
        explanation = self.explain(instance_array, background_data)

        # Get prediction - handle both regression and classification
        if isinstance(instance, pd.DataFrame):
            pred_array = self.model.predict(instance)
        else:
            pred_array = self.model.predict(instance_array)
        
        # Extract prediction value
        if isinstance(pred_array, np.ndarray):
            if pred_array.ndim == 0:
                prediction = float(pred_array.item())
            else:
                prediction = float(pred_array[0])
        elif isinstance(pred_array, (list, tuple)):
            prediction = float(pred_array[0])
        else:
            prediction = float(pred_array)

        # Create feature contributions list
        shap_vals = explanation['shap_values']
        # Handle different shap_values formats
        if isinstance(shap_vals, list):
            if len(shap_vals) > 0 and isinstance(shap_vals[0], (list, np.ndarray)):
                # Multi-dimensional: take first instance
                shap_vals = shap_vals[0]
            elif len(shap_vals) > 0:
                shap_vals = shap_vals[0]
        elif isinstance(shap_vals, np.ndarray):
            if shap_vals.ndim > 1:
                # Multi-dimensional: take first instance
                shap_vals = shap_vals[0]
            shap_vals = shap_vals.tolist() if isinstance(shap_vals, np.ndarray) else shap_vals
        
        # Ensure we have feature names
        if not self.feature_names:
            self.feature_names = [f'feature_{i}' for i in range(len(shap_vals))]
        
        # Ensure feature_names and shap_vals have same length
        if len(self.feature_names) != len(shap_vals):
            # Pad or truncate to match
            min_len = min(len(self.feature_names), len(shap_vals))
            self.feature_names = self.feature_names[:min_len]
            shap_vals = shap_vals[:min_len] if isinstance(shap_vals, list) else shap_vals[:min_len]
        
        feature_contributions = list(zip(self.feature_names, shap_vals))
        feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)

        return {
            'shap_values': shap_vals,
            'base_value': explanation['base_values'],
            'prediction': prediction,
            'feature_contributions': [
                {'feature': name, 'shap_value': float(value)}
                for name, value in feature_contributions
            ],
            'feature_names': self.feature_names
        }

    def get_feature_importance(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        background_data: Optional[Union[pd.DataFrame, np.ndarray]] = None
    ) -> Dict[str, float]:
        """
        Get global feature importance based on mean absolute SHAP values.

        Args:
            X: Dataset to calculate importance on
            background_data: Background dataset for KernelExplainer (optional)

        Returns:
            Dictionary mapping feature names to importance scores
        """
        explanation = self.explain(X, background_data)
        shap_values = np.array(explanation['shap_values'])

        # Calculate mean absolute SHAP value for each feature
        # Handle multi-dimensional SHAP values (e.g., multi-class classification)
        if shap_values.ndim > 2:
            # For multi-class: average across all classes
            importance_values = np.mean(np.abs(shap_values), axis=(0, -1))
        elif shap_values.ndim == 2:
            # 2D: (samples, features) - average across samples
            importance_values = np.mean(np.abs(shap_values), axis=0)
        else:
            # 1D: already aggregated
            importance_values = np.abs(shap_values)

        # Ensure we have a 1D array
        if importance_values.ndim > 1:
            importance_values = np.mean(importance_values, axis=-1)

        if self.feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importance_values))]
        else:
            feature_names = self.feature_names

        # Create dictionary and ensure all values are floats (not lists)
        importance_dict = {}
        for name, val in zip(feature_names, importance_values):
            # Ensure value is a scalar float
            if isinstance(val, (list, np.ndarray)):
                importance_dict[name] = float(np.mean(np.abs(val)))
            else:
                importance_dict[name] = float(val)
        
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

    def get_waterfall_data(
        self,
        instance: Union[pd.DataFrame, np.ndarray, List],
        background_data: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        max_display: int = 10
    ) -> Dict[str, Any]:
        """
        Get waterfall plot data for a single instance.

        Args:
            instance: Single instance to explain
            background_data: Background dataset
            max_display: Maximum number of features to display

        Returns:
            Dictionary with waterfall plot data
        """
        explanation = self.explain_instance(instance, background_data)

        # Get top features
        contributions = explanation['feature_contributions'][:max_display]

        # Prepare waterfall data
        waterfall_data = {
            'base_value': explanation['base_value'],
            'prediction': explanation['prediction'],
            'contributions': contributions,
            'feature_names': [c['feature'] for c in contributions],
            'shap_values': [c['shap_value'] for c in contributions]
        }

        return waterfall_data

    def summary(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        background_data: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        max_features: int = 20
    ) -> Dict[str, Any]:
        """
        Generate summary of SHAP values across dataset.

        Args:
            X: Dataset to summarize
            background_data: Background dataset
            max_features: Maximum features to include

        Returns:
            Summary dictionary with aggregated SHAP information
        """
        explanation = self.explain(X, background_data)
        feature_importance = self.get_feature_importance(X, background_data)

        # Get top features
        top_features = list(feature_importance.keys())[:max_features]

        return {
            'feature_importance': feature_importance,
            'top_features': top_features,
            'num_samples': len(X),
            'num_features': len(self.feature_names) if self.feature_names else X.shape[1],
            'base_value': explanation['base_values']
        }
