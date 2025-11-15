"""
Model version management utilities.
"""
from typing import Optional, Dict, Any, List
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_
from sqlalchemy.orm import selectinload

from app.models.database.ml_models import MLModel
from app.models.database.model_versions import ModelVersion
from app.core.logging import get_logger

logger = get_logger(__name__)


class VersionManager:
    """Manage model versions."""
    
    @staticmethod
    async def create_version(
        db: AsyncSession,
        model_id: int,
        version: str,
        model_path: str,
        performance_metrics: Optional[Dict[str, Any]] = None,
        features: Optional[List[str]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        feature_importance: Optional[Dict[str, float]] = None,
        dataset_size: Optional[int] = None,
        training_duration_seconds: Optional[float] = None,
        notes: Optional[str] = None,
        tags: Optional[List[str]] = None,
        make_active: bool = True
    ) -> ModelVersion:
        """
        Create a new model version.
        
        Args:
            db: Database session
            model_id: ID of the model
            version: Version string (e.g., "1.0.0")
            model_path: Path to model artifact
            performance_metrics: Performance metrics
            features: List of feature names
            hyperparameters: Model hyperparameters
            training_config: Training configuration
            feature_importance: Feature importance scores
            dataset_size: Size of training dataset
            training_duration_seconds: Training duration
            notes: Version notes
            tags: Tags for filtering
            make_active: Whether to make this version active
            
        Returns:
            Created model version
        """
        # Deactivate other versions if making this active
        if make_active:
            await VersionManager.deactivate_all_versions(db, model_id)
        
        # Get model size (would need to fetch from storage)
        model_size_bytes = None  # Could be calculated from model_path
        
        # Create version
        model_version = ModelVersion(
            model_id=model_id,
            version=version,
            model_path=model_path,
            performance_metrics=performance_metrics,
            features=features,
            hyperparameters=hyperparameters,
            training_config=training_config,
            feature_importance=feature_importance,
            dataset_size=dataset_size,
            training_duration_seconds=training_duration_seconds,
            training_date=datetime.utcnow(),
            is_active=make_active,
            notes=notes,
            tags=tags or []
        )
        
        db.add(model_version)
        await db.commit()
        await db.refresh(model_version)
        
        logger.info(f"Created model version {version} for model {model_id}")
        return model_version
    
    @staticmethod
    async def get_version(db: AsyncSession, version_id: int) -> Optional[ModelVersion]:
        """Get model version by ID."""
        return await db.get(ModelVersion, version_id)
    
    @staticmethod
    async def get_versions_by_model(
        db: AsyncSession,
        model_id: int,
        include_archived: bool = False
    ) -> List[ModelVersion]:
        """Get all versions for a model."""
        query = select(ModelVersion).where(ModelVersion.model_id == model_id)
        
        if not include_archived:
            query = query.where(ModelVersion.is_archived == False)
        
        query = query.order_by(ModelVersion.created_at.desc())
        
        result = await db.execute(query)
        return list(result.scalars().all())
    
    @staticmethod
    async def get_active_version(
        db: AsyncSession,
        model_id: int
    ) -> Optional[ModelVersion]:
        """Get the active version for a model."""
        query = select(ModelVersion).where(
            and_(
                ModelVersion.model_id == model_id,
                ModelVersion.is_active == True
            )
        )
        
        result = await db.execute(query)
        return result.scalar_one_or_none()
    
    @staticmethod
    async def deactivate_all_versions(db: AsyncSession, model_id: int) -> None:
        """Deactivate all versions for a model."""
        await db.execute(
            update(ModelVersion)
            .where(ModelVersion.model_id == model_id)
            .values(is_active=False)
        )
        await db.commit()
    
    @staticmethod
    async def activate_version(
        db: AsyncSession,
        version_id: int
    ) -> ModelVersion:
        """
        Activate a specific version (deactivates others).
        
        Args:
            db: Database session
            version_id: ID of version to activate
            
        Returns:
            Activated version
        """
        version = await db.get(ModelVersion, version_id)
        
        if not version:
            raise ValueError(f"Version {version_id} not found")
        
        # Deactivate all other versions for this model
        await VersionManager.deactivate_all_versions(db, version.model_id)
        
        # Activate this version
        version.is_active = True
        await db.commit()
        await db.refresh(version)
        
        logger.info(f"Activated version {version.version} for model {version.model_id}")
        return version
    
    @staticmethod
    async def archive_version(
        db: AsyncSession,
        version_id: int
    ) -> ModelVersion:
        """Archive a version."""
        version = await db.get(ModelVersion, version_id)
        
        if not version:
            raise ValueError(f"Version {version_id} not found")
        
        version.is_archived = True
        if version.is_active:
            version.is_active = False
        
        await db.commit()
        await db.refresh(version)
        
        logger.info(f"Archived version {version.version} for model {version.model_id}")
        return version
    
    @staticmethod
    async def update_version_metadata(
        db: AsyncSession,
        version_id: int,
        notes: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> ModelVersion:
        """Update version metadata."""
        version = await db.get(ModelVersion, version_id)
        
        if not version:
            raise ValueError(f"Version {version_id} not found")
        
        if notes is not None:
            version.notes = notes
        
        if tags is not None:
            version.tags = tags
        
        await db.commit()
        await db.refresh(version)
        
        return version

