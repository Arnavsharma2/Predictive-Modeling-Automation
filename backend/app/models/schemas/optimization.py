"""
Pydantic schemas for hyperparameter optimization API.
"""
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List


class HyperparameterSpace(BaseModel):
    """Hyperparameter search space definition."""
    type: str = Field(..., description="Type: 'int', 'float', or 'categorical'")
    low: Optional[float] = Field(None, description="Lower bound (for int/float)")
    high: Optional[float] = Field(None, description="Upper bound (for int/float)")
    choices: Optional[List[Any]] = Field(None, description="Choices (for categorical)")
    log: Optional[bool] = Field(False, description="Use log scale (for int/float)")
    step: Optional[float] = Field(None, description="Step size (for float)")


class OptimizationRequest(BaseModel):
    """Request to start hyperparameter optimization."""
    model_id: Optional[int] = Field(None, description="Model ID (if optimizing existing model)")
    algorithm: str = Field(..., description="Algorithm to optimize")
    search_space: Optional[Dict[str, Dict[str, Any]]] = Field(None, description="Custom search space")
    metric_name: str = Field("rmse", description="Metric to optimize")
    direction: str = Field("minimize", pattern="^(minimize|maximize)$", description="Optimization direction")
    n_trials: int = Field(100, ge=1, le=1000, description="Number of trials")
    timeout: Optional[float] = Field(None, ge=0, description="Timeout in seconds")
    study_name: Optional[str] = Field(None, description="Study name")


class OptimizationResponse(BaseModel):
    """Response from hyperparameter optimization."""
    study_name: str
    best_params: Dict[str, Any]
    best_value: float
    best_trial_number: int
    n_trials: int
    status: str = Field(..., description="Status: 'completed', 'timeout', 'failed'")


class StudySummaryResponse(BaseModel):
    """Study summary response."""
    study_name: str
    n_trials: int
    best_value: float
    best_params: Dict[str, Any]
    best_trial_number: int
    direction: str

