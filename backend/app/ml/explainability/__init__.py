"""
Model explainability and interpretability module.

This module provides tools for explaining model predictions using various techniques:
- SHAP (SHapley Additive exPlanations)
- LIME (Local Interpretable Model-agnostic Explanations)
- Feature importance analysis
- Partial dependence plots
"""

from .shap_explainer import ShapExplainer
from .lime_explainer import LimeExplainer
from .feature_importance import FeatureImportanceAnalyzer

__all__ = [
    'ShapExplainer',
    'LimeExplainer',
    'FeatureImportanceAnalyzer'
]
