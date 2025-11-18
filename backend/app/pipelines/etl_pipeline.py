"""
ETL pipeline using Prefect.
"""
from typing import Dict, Any
from datetime import datetime, timezone

from prefect import flow, get_run_logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.logging import get_logger
from app.core.database import get_db_session
from app.models.database.data_sources import DataSource, DataSourceType, DataSourceStatus
from app.models.database.etl_jobs import ETLJob, ETLJobStatus
from app.pipelines.tasks.extract import extract_data
from app.pipelines.tasks.transform import transform_data
from app.pipelines.tasks.load import load_to_database, update_etl_job_status

logger = get_logger(__name__)


@flow(name="etl_pipeline", retries=1, retry_delay_seconds=30)
async def etl_pipeline(
    source_id: int,
    etl_config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Main ETL pipeline flow.
    
    Args:
        source_id: Data source ID
        etl_config: ETL configuration (transform, validation, etc.)
        
    Returns:
        Dictionary with pipeline execution results
    """
    flow_logger = get_run_logger()
    flow_logger.info(f"Starting ETL pipeline for source {source_id}")
    
    etl_config = etl_config or {}
    job_id = None
    prefect_flow_run_id = None
    
    try:
        # Get flow run ID (Prefect 2.x compatible)
        try:
            from prefect import runtime
            if hasattr(runtime, 'flow_run') and runtime.flow_run:
                prefect_flow_run_id = str(runtime.flow_run.id)
        except (ImportError, AttributeError):
            # If runtime is not available, flow run ID will be None
            prefect_flow_run_id = None
        
        # Get data source
        # Use get_db_session to ensure proper event loop handling in Prefect tasks
        async with get_db_session() as session:
            source = await session.get(DataSource, source_id)
            if not source:
                raise ValueError(f"Data source {source_id} not found")
            
            # Check if ETL job with this flow run ID already exists (handles retries)
            job = None
            if prefect_flow_run_id:
                result = await session.execute(
                    select(ETLJob).where(ETLJob.prefect_flow_run_id == prefect_flow_run_id)
                )
                job = result.scalar_one_or_none()
            
            if job:
                # Reuse existing job
                flow_logger.info(f"Reusing existing ETL job {job.id} for flow run {prefect_flow_run_id}")
                job_id = job.id
                # Update job status to RUNNING if it was in a failed/pending state
                if job.status != ETLJobStatus.RUNNING:
                    job.status = ETLJobStatus.RUNNING
                    job.started_at = datetime.now(timezone.utc)
                    job.error_message = None
                    await session.commit()
            else:
                # Create new ETL job
                job = ETLJob(
                    source_id=source_id,
                    status=ETLJobStatus.RUNNING,
                    prefect_flow_run_id=prefect_flow_run_id,
                    started_at=datetime.now(timezone.utc),
                    metadata={"etl_config": etl_config}
                )
                session.add(job)
                await session.flush()  # Flush to get the ID without committing
                job_id = job.id
                await session.commit()
                flow_logger.info(f"Created ETL job {job_id} for source {source_id}")
            
            # Update source status
            source.status = DataSourceStatus.PROCESSING
            await session.commit()
        
        # Extract
        flow_logger.info("Extracting data...")
        raw_data = await extract_data(
            source_type=source.type,
            source_id=source_id,
            config=source.config or {}
        )
        
        # Transform
        flow_logger.info("Transforming data...")
        df, validation_result = await transform_data(
            raw_data=raw_data,
            source_type=source.type.value,
            transform_config=etl_config.get("transform", {})
        )
        
        # Load
        flow_logger.info("Loading data to database...")
        load_result = await load_to_database(
            df=df,
            source_id=source_id,
            batch_size=etl_config.get("batch_size", 1000)
        )
        
        # Update job status
        await update_etl_job_status(
            job_id=job_id,
            status=ETLJobStatus.COMPLETED,
            records_processed=load_result["records_processed"],
            records_failed=load_result["records_failed"],
            metadata={
                "etl_config": etl_config,
                "validation": validation_result,
                "load_result": load_result
            }
        )
        
        # Update source status
        # Use get_db_session to ensure proper event loop handling in Prefect tasks
        async with get_db_session() as session:
            source = await session.get(DataSource, source_id)
            if source:
                source.status = DataSourceStatus.ACTIVE
                await session.commit()
        
        flow_logger.info(f"ETL pipeline completed successfully for source {source_id}")
        
        return {
            "success": True,
            "job_id": job_id,
            "prefect_flow_run_id": prefect_flow_run_id,
            "records_processed": load_result["records_processed"],
            "records_failed": load_result["records_failed"],
            "validation": validation_result
        }
        
    except Exception as e:
        flow_logger.error(f"ETL pipeline failed: {e}")
        
        # Update job status to failed
        if job_id:
            await update_etl_job_status(
                job_id=job_id,
                status=ETLJobStatus.FAILED,
                error_message=str(e)
            )
        
        # Update source status
        # Use get_db_session to ensure proper event loop handling in Prefect tasks
        try:
            async with get_db_session() as session:
                source = await session.get(DataSource, source_id)
                if source:
                    source.status = DataSourceStatus.ERROR
                    await session.commit()
        except Exception:
            pass
        
        raise

