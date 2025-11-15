"""
Load tasks for ETL pipeline.
"""
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession

from prefect import task
from app.core.logging import get_logger
from app.core.database import AsyncSessionLocal, get_db_session
from app.models.database.data_points import DataPoint
from app.models.database.etl_jobs import ETLJob, ETLJobStatus

logger = get_logger(__name__)


def convert_to_json_serializable(obj: Any) -> Any:
    """
    Convert numpy/pandas types to native Python types for JSON serialization.
    
    Args:
        obj: Object that may contain numpy/pandas types
        
    Returns:
        Object with all numpy/pandas types converted to native Python types
    """
    # Handle None first
    if obj is None:
        return None
    
    # Handle numpy arrays before checking pd.isna (which can return arrays)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    
    # Handle pandas types
    if isinstance(obj, pd.Series):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    
    # Handle numpy scalar types
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    
    # Handle NaN/NaT values (check after numpy/pandas types to avoid array ambiguity)
    try:
        if pd.isna(obj):
            return None
    except (ValueError, TypeError):
        # pd.isna can raise ValueError for arrays or TypeError for unsupported types
        pass
    
    # Handle datetime objects
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    
    # Handle collections recursively
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    
    # Return as-is for native Python types
    return obj


@task(name="load_to_database")
async def load_to_database(
    df: pd.DataFrame,
    source_id: int,
    batch_size: int = 1000
) -> Dict[str, Any]:
    """
    Load transformed data into database.
    
    Args:
        df: Transformed DataFrame
        source_id: Data source ID
        batch_size: Batch size for insertion
        
    Returns:
        Dictionary with load statistics
    """
    logger.info(f"Loading {len(df)} rows to database for source {source_id}")
    
    records_processed = 0
    records_failed = 0
    
    try:
        # Use get_db_session to ensure proper event loop handling in Prefect tasks
        async with get_db_session() as session:
            # Convert DataFrame to records
            records = df.to_dict(orient="records")
            
            # Insert in batches
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                data_points = []
                
                for record in batch:
                    try:
                        # Convert to JSON-serializable format
                        serializable_record = convert_to_json_serializable(record)

                        # Create data point
                        data_point = DataPoint(
                            source_id=source_id,
                            data=serializable_record,
                            timestamp=datetime.utcnow()
                        )
                        data_points.append(data_point)
                    except Exception as e:
                        logger.warning(f"Error creating data point: {e}")
                        records_failed += 1
                
                # Bulk insert batch
                if data_points:
                    try:
                        session.add_all(data_points)
                        await session.commit()
                        records_processed += len(data_points)
                        logger.info(f"Inserted batch {i // batch_size + 1}: {len(data_points)} records")
                    except Exception as e:
                        await session.rollback()
                        logger.error(f"Error inserting batch: {e}")
                        records_failed += len(data_points)
            
            logger.info(f"Load complete: {records_processed} processed, {records_failed} failed")
            
            return {
                "records_processed": records_processed,
                "records_failed": records_failed,
                "total_records": len(records)
            }
            
    except Exception as e:
        logger.error(f"Error loading data to database: {e}")
        raise


@task(name="update_etl_job_status")
async def update_etl_job_status(
    job_id: int,
    status: ETLJobStatus,
    records_processed: int = 0,
    records_failed: int = 0,
    error_message: str = None,
    metadata: Dict[str, Any] = None
) -> None:
    """
    Update ETL job status in database.
    
    Args:
        job_id: ETL job ID
        status: New status
        records_processed: Number of records processed
        records_failed: Number of records failed
        error_message: Error message (if failed)
        metadata: Additional metadata
    """
    logger.info(f"Updating ETL job {job_id} status to {status}")
    
    try:
        # Use get_db_session to ensure proper event loop handling in Prefect tasks
        async with get_db_session() as session:
            job = await session.get(ETLJob, job_id)
            if not job:
                logger.warning(f"ETL job {job_id} not found")
                return
            
            job.status = status
            job.records_processed = records_processed
            job.records_failed = records_failed
            job.error_message = error_message
            if metadata:
                # Convert numpy/pandas types to native Python types for JSON serialization
                job.meta_data = convert_to_json_serializable(metadata)
            
            if status == ETLJobStatus.RUNNING and not job.started_at:
                job.started_at = datetime.utcnow()
            elif status in [ETLJobStatus.COMPLETED, ETLJobStatus.FAILED, ETLJobStatus.CANCELLED]:
                job.completed_at = datetime.utcnow()
            
            await session.commit()
            logger.info(f"ETL job {job_id} status updated to {status}")
            
    except Exception as e:
        logger.error(f"Error updating ETL job status: {e}")
        raise

