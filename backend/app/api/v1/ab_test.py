"""
A/B testing API endpoints.
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.database.ab_tests import ABTest, ABTestStatus
from app.models.schemas.ab_test import (
    ABTestCreate, ABTestResponse, ABTestResultsResponse, ABTestListResponse
)
from app.services.ab_testing.ab_test_manager import ABTestManager

router = APIRouter(prefix="/ab-test", tags=["A/B Testing"])


@router.post("/create", response_model=ABTestResponse, status_code=status.HTTP_201_CREATED)
async def create_ab_test(
    test: ABTestCreate,
    db: AsyncSession = Depends(get_db)
):
    """Create A/B test."""
    ab_test = await ABTestManager.create_test(
        db=db,
        name=test.name,
        model_id=test.model_id,
        control_version_id=test.control_version_id,
        treatment_version_id=test.treatment_version_id,
        control_traffic_percentage=test.control_traffic_percentage,
        treatment_traffic_percentage=test.treatment_traffic_percentage,
        routing_strategy=test.routing_strategy,
        min_samples=test.min_samples,
        description=test.description
    )
    
    return ABTestResponse.model_validate(ab_test)


@router.get("/", response_model=ABTestListResponse)
async def list_ab_tests(
    status: Optional[str] = None,
    model_id: Optional[int] = None,
    limit: int = 100,
    offset: int = 0,
    db: AsyncSession = Depends(get_db)
):
    """List A/B tests."""
    from sqlalchemy import select, and_
    
    query = select(ABTest)
    conditions = []
    
    if status:
        conditions.append(ABTest.status == ABTestStatus(status))
    if model_id:
        conditions.append(ABTest.model_id == model_id)
    
    if conditions:
        query = query.where(and_(*conditions))
    
    query = query.order_by(ABTest.created_at.desc()).limit(limit).offset(offset)
    
    result = await db.execute(query)
    tests = result.scalars().all()
    
    return ABTestListResponse(
        tests=[ABTestResponse.model_validate(t) for t in tests],
        total=len(tests)
    )


@router.post("/{test_id}/start", response_model=ABTestResponse)
async def start_ab_test(
    test_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Start A/B test."""
    test = await ABTestManager.start_test(db=db, test_id=test_id)
    return ABTestResponse.model_validate(test)


@router.post("/{test_id}/stop", response_model=ABTestResponse)
async def stop_ab_test(
    test_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Stop A/B test."""
    test = await ABTestManager.stop_test(db=db, test_id=test_id)
    return ABTestResponse.model_validate(test)


@router.get("/{test_id}/results", response_model=ABTestResultsResponse)
async def get_ab_test_results(
    test_id: int,
    metric_name: str = "accuracy",
    db: AsyncSession = Depends(get_db)
):
    """Get A/B test results."""
    results = await ABTestManager.get_results(
        db=db,
        test_id=test_id,
        metric_name=metric_name
    )
    
    return ABTestResultsResponse(**results)

