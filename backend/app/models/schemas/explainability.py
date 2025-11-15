"""
Pydantic schemas for model explainability API.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union


class FeatureContribution(BaseModel):
    """Feature contribution in an explanation."""
    feature: str = Field(..., description="Feature name")
    value: Union[float, str] = Field(..., description="SHAP value or LIME weight")
    description: Optional[str] = Field(None, description="Description of the feature condition (LIME)")


class ShapExplanationRequest(BaseModel):
    """Request for SHAP explanation."""
    model_id: int = Field(..., description="Model ID")
    features: Optional[Dict[str, Any]] = Field(None, description="Feature values in original column format (before preprocessing). If None, generates global explanation from training data.")
    background_samples: Optional[int] = Field(100, description="Number of background samples for KernelExplainer")
    sample_size: Optional[int] = Field(100, description="Number of training samples to use for global explanation (if features not provided)")


class ShapExplanationResponse(BaseModel):
    """Response containing SHAP explanation."""
    model_id: int
    shap_values: List[float] = Field(..., description="SHAP values for each feature")
    base_value: float = Field(..., description="Expected value (baseline prediction)")
    prediction: float = Field(..., description="Model prediction for this instance")
    feature_contributions: List[Dict[str, Union[str, float]]] = Field(
        ...,
        description="List of feature contributions sorted by importance"
    )
    feature_names: List[str]


class ShapWaterfallData(BaseModel):
    """Waterfall plot data for SHAP."""
    model_id: int
    base_value: float
    prediction: float
    contributions: List[Dict[str, Union[str, float]]]
    feature_names: List[str]
    shap_values: List[float]


class LimeExplanationRequest(BaseModel):
    """Request for LIME explanation."""
    model_id: int = Field(..., description="Model ID")
    features: Optional[Dict[str, Any]] = Field(None, description="Feature values in original column format (before preprocessing). If None, generates global explanation from training data.")
    num_features: int = Field(10, description="Number of features to include in explanation")
    num_samples: int = Field(5000, description="Number of samples for local model")
    sample_size: Optional[int] = Field(100, description="Number of training samples to use for global explanation (if features not provided)")


class LimeExplanationResponse(BaseModel):
    """Response containing LIME explanation."""
    model_id: int
    prediction: Union[float, List[float]] = Field(..., description="Model prediction")
    predicted_class: Optional[int] = Field(None, description="Predicted class (classification only)")
    local_prediction: Optional[float] = Field(None, description="Local model prediction")
    feature_contributions: List[Dict[str, Any]] = Field(
        ...,
        description="Feature contributions with descriptions (description is optional)"
    )
    intercept: float = Field(..., description="Local model intercept")
    score: Optional[float] = Field(None, description="R² score of local model")
    feature_names: List[str]
    mode: str = Field(..., description="'classification' or 'regression'")


class FeatureImportanceRequest(BaseModel):
    """Request for feature importance."""
    model_id: int = Field(..., description="Model ID")
    method: str = Field(
        "model",
        description="Importance method: 'model', 'permutation', or 'comprehensive'"
    )
    n_repeats: Optional[int] = Field(10, description="Number of repeats for permutation importance")


class FeatureImportanceResponse(BaseModel):
    """Response containing feature importance."""
    model_id: int
    method: str
    importance: Dict[str, float] = Field(..., description="Feature importance scores")
    importance_std: Optional[Dict[str, float]] = Field(
        None,
        description="Standard deviation (for permutation importance)"
    )
    feature_names: List[str]


class ComprehensiveImportanceResponse(BaseModel):
    """Response containing comprehensive feature importance from multiple methods."""
    model_id: int
    model_importance: Optional[Dict[str, float]] = Field(None, description="Built-in model importance")
    permutation_importance: Optional[Dict[str, float]] = Field(None, description="Permutation importance")
    permutation_std: Optional[Dict[str, float]] = Field(None, description="Permutation std dev")


class ExplanationSummaryRequest(BaseModel):
    """Request for explanation summary."""
    model_id: int = Field(..., description="Model ID")
    dataset_id: Optional[int] = Field(None, description="Dataset ID to analyze (optional)")
    max_features: int = Field(20, description="Maximum features to include in summary")
    sample_size: Optional[int] = Field(100, description="Number of samples to analyze")


class ExplanationSummaryResponse(BaseModel):
    """Response containing explanation summary."""
    model_id: int
    dataset_id: Optional[int]
    feature_importance: Dict[str, float]
    top_features: List[str]
    num_samples: int
    num_features: int
    base_value: float
    method: str = Field(..., description="Method used: 'shap' or 'lime'")
