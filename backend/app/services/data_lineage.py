"""
Data lineage tracking service.
"""
from typing import Optional, Dict, Any, List
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from sqlalchemy.orm import selectinload

from app.models.database.data_lineage import DataLineage
from app.models.database.data_sources import DataSource
from app.models.database.ml_models import MLModel
from app.models.database.training_jobs import TrainingJob
from app.core.logging import get_logger

logger = get_logger(__name__)


class DataLineageService:
    """Service for tracking data lineage."""
    
    async def create_lineage(
        self,
        db: AsyncSession,
        source_id: int,
        target_type: str,  # "data_source", "model", "training_job"
        target_id: int,
        transformation: Optional[Dict[str, Any]] = None,
        created_by: Optional[int] = None
    ) -> DataLineage:
        """
        Create a lineage record.
        
        Args:
            db: Database session
            source_id: Source data source ID
            target_type: Type of target (data_source, model, training_job)
            target_id: Target ID
            transformation: Transformation details
            created_by: User ID
            
        Returns:
            DataLineage object
        """
        try:
            lineage = DataLineage(
                source_data_source_id=source_id,
                target_type=target_type,
                target_id=target_id,
                transformation=transformation,
                created_by=created_by
            )
            
            db.add(lineage)
            await db.commit()
            await db.refresh(lineage)
            
            return lineage
        except Exception as e:
            logger.error(f"Error creating lineage: {e}", exc_info=True)
            raise
    
    async def get_lineage_for_source(
        self,
        db: AsyncSession,
        data_source_id: int
    ) -> List[DataLineage]:
        """Get all lineage records for a data source."""
        result = await db.execute(
            select(DataLineage)
            .where(DataLineage.source_data_source_id == data_source_id)
            .order_by(DataLineage.created_at.desc())
        )
        return list(result.scalars().all())
    
    async def get_lineage_for_target(
        self,
        db: AsyncSession,
        target_type: str,
        target_id: int
    ) -> List[DataLineage]:
        """Get all lineage records for a target."""
        result = await db.execute(
            select(DataLineage)
            .where(
                and_(
                    DataLineage.target_type == target_type,
                    DataLineage.target_id == target_id
                )
            )
            .order_by(DataLineage.created_at.desc())
        )
        return list(result.scalars().all())
    
    async def get_full_lineage_tree(
        self,
        db: AsyncSession,
        data_source_id: int,
        max_depth: int = 5
    ) -> Optional[Dict[str, Any]]:
        """
        Get full lineage tree starting from a data source.
        
        Args:
            db: Database session
            data_source_id: Starting data source ID
            max_depth: Maximum depth to traverse
            
        Returns:
            Lineage tree structure
        """
        async def build_tree(source_id: int, depth: int = 0) -> Optional[Dict[str, Any]]:
            if depth >= max_depth:
                return None
            
            # Get source
            source_result = await db.execute(
                select(DataSource).where(DataSource.id == source_id)
            )
            source = source_result.scalar_one_or_none()
            
            if not source:
                return None
            
            tree = {
                "id": source.id,
                "name": source.name,
                "type": "data_source",
                "children": []
            }
            
            # Get all targets
            lineage_records = await self.get_lineage_for_source(db, source_id)
            
            for lineage in lineage_records:
                child = None
                
                if lineage.target_type == "data_source":
                    target_result = await db.execute(
                        select(DataSource).where(DataSource.id == lineage.target_id)
                    )
                    target = target_result.scalar_one_or_none()
                    if target:
                        child = {
                            "id": target.id,
                            "name": target.name,
                            "type": "data_source",
                            "transformation": lineage.transformation,
                            "children": []
                        }
                        # Recursively build child tree
                        child_tree = await build_tree(target.id, depth + 1)
                        if child_tree:
                            child["children"] = [child_tree]
                
                elif lineage.target_type == "model":
                    target_result = await db.execute(
                        select(MLModel).where(MLModel.id == lineage.target_id)
                    )
                    target = target_result.scalar_one_or_none()
                    if target:
                        child = {
                            "id": target.id,
                            "name": target.name,
                            "type": "model",
                            "transformation": lineage.transformation
                        }
                
                elif lineage.target_type == "training_job":
                    target_result = await db.execute(
                        select(TrainingJob).where(TrainingJob.id == lineage.target_id)
                    )
                    target = target_result.scalar_one_or_none()
                    if target:
                        child = {
                            "id": target.id,
                            "name": f"Training Job {target.id}",
                            "type": "training_job",
                            "transformation": lineage.transformation
                        }
                
                if child:
                    tree["children"].append(child)
            
            return tree
        
        return await build_tree(data_source_id, 0)

