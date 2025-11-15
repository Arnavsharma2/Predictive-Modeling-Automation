"""
Pydantic schemas for experiment tracking API.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


class ExperimentCreate(BaseModel):
    """Request to create an experiment."""
    name: str = Field(..., description="Experiment name")
    tags: Optional[Dict[str, str]] = Field(None, description="Experiment tags")


class ExperimentResponse(BaseModel):
    """Experiment information."""
    experiment_id: str
    name: str
    tags: Dict[str, str]
    artifact_location: Optional[str] = None


class RunCreate(BaseModel):
    """Request to create a run."""
    experiment_name: str = Field(..., description="Experiment name")
    run_name: Optional[str] = Field(None, description="Run name")
    tags: Optional[Dict[str, str]] = Field(None, description="Run tags")


class RunResponse(BaseModel):
    """Run information."""
    run_id: str
    run_name: Optional[str]
    experiment_id: str
    status: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metrics: Dict[str, float]
    params: Dict[str, str]
    tags: Dict[str, str]


class RunListResponse(BaseModel):
    """List of runs."""
    runs: List[RunResponse]
    total: int


class MetricLog(BaseModel):
    """Metric to log."""
    name: str = Field(..., description="Metric name")
    value: float = Field(..., description="Metric value")
    step: Optional[int] = Field(None, description="Step number")


class ParamLog(BaseModel):
    """Parameter to log."""
    name: str = Field(..., description="Parameter name")
    value: str = Field(..., description="Parameter value")


class LogMetricsRequest(BaseModel):
    """Request to log metrics."""
    run_id: str = Field(..., description="Run ID")
    metrics: List[MetricLog] = Field(..., description="Metrics to log")


class LogParamsRequest(BaseModel):
    """Request to log parameters."""
    run_id: str = Field(..., description="Run ID")
    params: List[ParamLog] = Field(..., description="Parameters to log")


class RunSearchRequest(BaseModel):
    """Request to search runs."""
    experiment_name: str = Field(..., description="Experiment name")
    filter_string: Optional[str] = Field(None, description="Filter string (e.g., 'metrics.rmse < 0.5')")
    max_results: int = Field(100, ge=1, le=1000, description="Maximum number of results")


class RunComparisonResponse(BaseModel):
    """Run comparison results."""
    runs: List[RunResponse]
    metric_name: str
    best_run_id: Optional[str] = None
    best_metric_value: Optional[float] = None

