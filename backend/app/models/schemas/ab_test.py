"""
A/B testing schemas.
"""
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field
from app.models.database.ab_tests import ABTestStatus


class ABTestCreate(BaseModel):
    """Schema for creating A/B test."""
    name: str = Field(..., min_length=1, max_length=200)
    model_id: int
    control_version_id: int
    treatment_version_id: int
    control_traffic_percentage: float = Field(50.0, ge=0, le=100)
    treatment_traffic_percentage: float = Field(50.0, ge=0, le=100)
    routing_strategy: str = Field("random", description="Routing strategy: random, user_id, session_id")
    min_samples: int = Field(1000, ge=1)
    description: Optional[str] = None


class ABTestResponse(BaseModel):
    """Schema for A/B test response."""
    id: int
    name: str
    description: Optional[str]
    status: str
    model_id: int
    control_version_id: Optional[int]
    treatment_version_id: Optional[int]
    control_traffic_percentage: float
    treatment_traffic_percentage: float
    routing_strategy: str
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    min_samples: int
    control_metrics: Optional[Dict[str, Any]]
    treatment_metrics: Optional[Dict[str, Any]]
    statistical_significance: Optional[float]
    winner: Optional[str]
    created_at: datetime
    updated_at: Optional[datetime]
    
    class Config:
        from_attributes = True


class ABTestListResponse(BaseModel):
    """Schema for A/B test list response."""
    tests: List[ABTestResponse]
    total: int


class ABTestResultsResponse(BaseModel):
    """Schema for A/B test results."""
    test_id: int
    test_name: str
    status: str
    control_metrics: Dict[str, Any]
    treatment_metrics: Dict[str, Any]
    statistical_test: Dict[str, Any]
    winner: Optional[str]
    control_samples: int
    treatment_samples: int

