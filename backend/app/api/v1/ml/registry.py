"""
Model registry API endpoints.
"""
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.core.logging import get_logger
from app.models.database.ml_models import MLModel, ModelType, ModelStatus
from app.models.database.model_versions import ModelVersion
from app.models.schemas.ml import (
    ModelRegistryResponse,
    ModelRegistryListResponse,
    MLModelResponse,
    ModelVersionResponse
)
from app.ml.versioning.version_manager import VersionManager

logger = get_logger(__name__)

router = APIRouter(prefix="/registry", tags=["Model Registry"])


@router.get("", response_model=ModelRegistryListResponse)
async def list_registry(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    model_type: Optional[ModelType] = None,
    status: Optional[ModelStatus] = None,
    tag: Optional[str] = None,
    search: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    List all models in the registry with metadata.
    """
    # Build query
    query = select(MLModel)
    
    if model_type:
        query = query.where(MLModel.type == model_type)
    
    if status:
        query = query.where(MLModel.status == status)
    
    if search:
        query = query.where(
            or_(
                MLModel.name.ilike(f"%{search}%"),
                MLModel.description.ilike(f"%{search}%")
            )
        )
    
    # Get total count
    count_query = select(func.count()).select_from(MLModel)
    if model_type:
        count_query = count_query.where(MLModel.type == model_type)
    if status:
        count_query = count_query.where(MLModel.status == status)
    if search:
        count_query = count_query.where(
            or_(
                MLModel.name.ilike(f"%{search}%"),
                MLModel.description.ilike(f"%{search}%")
            )
        )
    
    total_result = await db.execute(count_query)
    total = total_result.scalar()
    
    # Get models with eager loading to avoid N+1 queries
    query = query.options(selectinload(MLModel.versions))
    query = query.order_by(MLModel.created_at.desc()).offset(skip).limit(limit)
    result = await db.execute(query)
    models = result.scalars().all()

    # Build registry responses
    registry_entries = []
    for model in models:
        # Filter versions in Python to avoid additional queries
        non_archived_versions = [v for v in model.versions if not v.is_archived]
        active_version = next((v for v in non_archived_versions if v.is_active), None)

        # Use pre-loaded versions instead of querying again
        all_versions = non_archived_versions
        
        # Get tags from active version or model
        tags = []
        if active_version and active_version.tags:
            tags = active_version.tags
        elif all_versions:
            # Collect tags from all versions
            for v in all_versions:
                if v.tags:
                    tags.extend(v.tags)
            tags = list(set(tags))  # Remove duplicates
        
        # Filter by tag if specified
        if tag and tag not in tags:
            continue
        
        registry_entries.append(ModelRegistryResponse(
            model=MLModelResponse.model_validate(model),
            active_version=ModelVersionResponse.model_validate(active_version) if active_version else None,
            total_versions=len(all_versions),
            tags=tags if tags else None
        ))
    
    return ModelRegistryListResponse(
        models=registry_entries,
        total=total
    )


@router.get("/{model_id}", response_model=ModelRegistryResponse)
async def get_registry_entry(
    model_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get detailed registry entry for a model.
    """
    model = await db.get(MLModel, model_id)
    
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )
    
    # Get active version
    active_version = await VersionManager.get_active_version(db, model_id)
    
    # Get all versions
    all_versions = await VersionManager.get_versions_by_model(db, model_id, include_archived=False)
    
    # Get tags
    tags = []
    if active_version and active_version.tags:
        tags = active_version.tags
    elif all_versions:
        for v in all_versions:
            if v.tags:
                tags.extend(v.tags)
        tags = list(set(tags))
    
    return ModelRegistryResponse(
        model=MLModelResponse.model_validate(model),
        active_version=ModelVersionResponse.model_validate(active_version) if active_version else None,
        total_versions=len(all_versions),
        tags=tags if tags else None
    )

