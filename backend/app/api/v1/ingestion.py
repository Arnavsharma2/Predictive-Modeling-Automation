"""
Data ingestion API endpoints.
"""
from typing import List
from datetime import datetime
from pathlib import Path
import uuid

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_db
from app.core.logging import get_logger
from app.models.database.data_sources import DataSource, DataSourceType, DataSourceStatus
from app.models.database.etl_jobs import ETLJob, ETLJobStatus
from app.models.schemas.ingestion import (
    CSVUploadRequest,
    CSVUploadResponse,
    APISourceRequest,
    APISourceResponse,
    ScrapingRequest,
    ScrapingResponse,
    DataSourceResponse,
    DataSourceListResponse,
    ETLJobResponse,
    ETLJobListResponse,
)
from app.processors.csv_processor import CSVProcessor
from app.storage.cloud_storage import cloud_storage
from app.pipelines.etl_pipeline import etl_pipeline
import aiofiles.os

logger = get_logger(__name__)

router = APIRouter(prefix="/ingestion", tags=["Ingestion"])


@router.post("/upload", response_model=CSVUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_csv(
    file: UploadFile = File(...),
    name: str = Form(None),
    description: str = Form(None),
    auto_process: bool = Form(True),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload CSV file for data ingestion.
    """
    try:
        # Read file content
        file_content = await file.read()
        
        # Validate file
        processor = CSVProcessor()
        validation = processor.validate_file(file_content, file.filename)
        
        if not validation["valid"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=validation["message"]
            )
        
        # Generate unique file path
        file_id = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix
        file_path = f"uploads/{file_id}{file_extension}"
        
        # Upload to cloud storage or save locally
        try:
            storage_path = await cloud_storage.upload_file(
                file_content=file_content,
                file_path=file_path,
                content_type=file.content_type or "text/csv"
            )
        except Exception as e:
            logger.warning(f"Could not upload to cloud storage, saving locally: {e}")
            # Save file locally
            uploads_dir = Path("uploads")
            uploads_dir.mkdir(exist_ok=True)
            local_file_path = uploads_dir / f"{file_id}{file_extension}"
            with open(local_file_path, "wb") as f:
                f.write(file_content)
            storage_path = str(local_file_path)
            logger.info(f"File saved locally at {storage_path}")
        
        # Create data source
        # Use provided name if it's not empty, otherwise use filename
        source_name = name.strip() if name and name.strip() else file.filename
        
        source = DataSource(
            name=source_name,
            type=DataSourceType.CSV,
            status=DataSourceStatus.PROCESSING,
            description=description.strip() if description else None,
            config={
                "file_path": storage_path,
                "filename": file.filename,
                "encoding": validation.get("encoding", "utf-8"),
                "file_size": validation.get("file_size")
            }
        )
        db.add(source)
        await db.commit()
        await db.refresh(source)
        
        # Trigger ETL pipeline if auto_process is True
        if auto_process:
            try:
                # Run ETL pipeline asynchronously (Prefect 2.x - flows are async functions)
                import asyncio
                asyncio.create_task(
                    etl_pipeline.with_options(name=f"etl_csv_{source.id}")(
                        source_id=source.id,
                        etl_config={}
                    )
                )
            except Exception as e:
                logger.error(f"Error triggering ETL pipeline: {e}")
                source.status = DataSourceStatus.ERROR
                await db.commit()
        
        return CSVUploadResponse(
            source_id=source.id,
            name=source.name,
            status=source.status,
            message="File uploaded successfully",
            file_path=storage_path
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading CSV file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading file: {str(e)}"
        )


@router.post("/api-source", response_model=APISourceResponse, status_code=status.HTTP_201_CREATED)
async def create_api_source(
    request: APISourceRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Configure external API data source.
    """
    try:
        # Create data source
        source = DataSource(
            name=request.name,
            type=DataSourceType.API,
            status=DataSourceStatus.ACTIVE,
            description=request.description,
            config={
                "url": str(request.url),
                "method": request.method,
                "headers": request.headers or {},
                "params": request.params or {},
                "body": request.body,
                "auth": request.auth.dict() if request.auth else None,
                "rate_limit": request.rate_limit,
                "retry_count": request.retry_count,
                "retry_delay": request.retry_delay,
                "polling_interval": request.polling_interval
            }
        )
        db.add(source)
        await db.commit()
        await db.refresh(source)
        
        # Trigger ETL pipeline if auto_process is True
        if request.auto_process:
            try:
                # Run ETL pipeline asynchronously (Prefect 2.x - flows are async functions)
                import asyncio
                asyncio.create_task(
                    etl_pipeline.with_options(name=f"etl_api_{source.id}")(
                        source_id=source.id,
                        etl_config={}
                    )
                )
                source.status = DataSourceStatus.PROCESSING
                await db.commit()
            except Exception as e:
                logger.error(f"Error triggering ETL pipeline: {e}")
        
        return APISourceResponse(
            source_id=source.id,
            name=source.name,
            status=source.status,
            message="API source configured successfully"
        )
        
    except Exception as e:
        logger.error(f"Error creating API source: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating API source: {str(e)}"
        )


@router.post("/scrape", response_model=ScrapingResponse, status_code=status.HTTP_201_CREATED)
async def create_scraping_job(
    request: ScrapingRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Configure web scraping job.
    """
    try:
        # Create data source
        source = DataSource(
            name=request.name,
            type=DataSourceType.WEB_SCRAPE,
            status=DataSourceStatus.ACTIVE,
            description=request.description,
            config={
                "url": str(request.config.url),
                "selectors": request.config.selectors,
                "respect_robots_txt": request.config.respect_robots_txt,
                "rate_limit": request.config.rate_limit,
                "max_pages": request.config.max_pages,
                "pagination": request.config.pagination
            }
        )
        db.add(source)
        await db.commit()
        await db.refresh(source)
        
        # Trigger ETL pipeline if auto_process is True
        if request.auto_process:
            try:
                # Run ETL pipeline asynchronously (Prefect 2.x - flows are async functions)
                import asyncio
                asyncio.create_task(
                    etl_pipeline.with_options(name=f"etl_scrape_{source.id}")(
                        source_id=source.id,
                        etl_config={}
                    )
                )
                source.status = DataSourceStatus.PROCESSING
                await db.commit()
            except Exception as e:
                logger.error(f"Error triggering ETL pipeline: {e}")
        
        return ScrapingResponse(
            source_id=source.id,
            name=source.name,
            status=source.status,
            message="Scraping job configured successfully"
        )
        
    except Exception as e:
        logger.error(f"Error creating scraping job: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating scraping job: {str(e)}"
        )


@router.get("/sources", response_model=DataSourceListResponse)
async def list_data_sources(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
):
    """
    List all data sources.
    """
    try:
        result = await db.execute(
            select(DataSource)
            .offset(skip)
            .limit(limit)
            .order_by(DataSource.created_at.desc())
        )
        sources = result.scalars().all()
        
        total_result = await db.execute(select(DataSource))
        total = len(total_result.scalars().all())
        
        return DataSourceListResponse(
            sources=[DataSourceResponse.model_validate(s) for s in sources],
            total=total
        )
    except Exception as e:
        logger.error(f"Error listing data sources: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing data sources: {str(e)}"
        )


@router.get("/sources/{source_id}", response_model=DataSourceResponse)
async def get_data_source(
    source_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get data source by ID.
    """
    source = await db.get(DataSource, source_id)
    if not source:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Data source {source_id} not found"
        )
    return DataSourceResponse.model_validate(source)


@router.get("/jobs", response_model=ETLJobListResponse)
async def list_etl_jobs(
    source_id: int = None,
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
):
    """
    List ETL jobs.
    """
    try:
        query = select(ETLJob)
        if source_id:
            query = query.where(ETLJob.source_id == source_id)
        
        result = await db.execute(
            query.offset(skip).limit(limit).order_by(ETLJob.created_at.desc())
        )
        jobs = result.scalars().all()
        
        total_result = await db.execute(query)
        total = len(total_result.scalars().all())
        
        return ETLJobListResponse(
            jobs=[ETLJobResponse.model_validate(j) for j in jobs],
            total=total
        )
    except Exception as e:
        logger.error(f"Error listing ETL jobs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing ETL jobs: {str(e)}"
        )


@router.get("/jobs/{job_id}", response_model=ETLJobResponse)
async def get_etl_job(
    job_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get ETL job by ID.
    """
    job = await db.get(ETLJob, job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"ETL job {job_id} not found"
        )
    return ETLJobResponse.model_validate(job)


@router.delete("/sources/{source_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_data_source(
    source_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a data source and all its associated data.
    This will also delete:
    - All data points (CASCADE)
    - All ETL jobs (CASCADE)
    - Source files from storage
    """
    source = await db.get(DataSource, source_id)
    
    if not source:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Data source {source_id} not found"
        )
    
    try:
        # Delete source file from storage if it exists
        if source.config and "file_path" in source.config:
            file_path = source.config.get("file_path")
            if file_path:
                try:
                    # Try cloud storage deletion (handles S3, Azure, and local)
                    deleted = await cloud_storage.delete_file(file_path)
                    if deleted:
                        logger.info(f"Deleted file from storage: {file_path}")
                    else:
                        # If cloud storage deletion returned False, try direct local file deletion
                        # Handle both absolute and relative paths
                        local_path = None
                        if Path(file_path).is_absolute() and Path(file_path).exists():
                            local_path = Path(file_path)
                        elif not Path(file_path).is_absolute():
                            # Try relative path in uploads directory
                            uploads_path = Path("uploads") / Path(file_path).name
                            if uploads_path.exists():
                                local_path = uploads_path
                        
                        if local_path:
                            try:
                                await aiofiles.os.remove(local_path)
                                logger.info(f"Deleted local file: {local_path}")
                            except Exception as e:
                                logger.warning(f"Failed to delete local file {local_path}: {e}")
                except Exception as e:
                    logger.warning(f"Failed to delete file {file_path}: {e}")
        
        # Delete the data source (CASCADE will handle data points, ETL jobs, etc.)
        await db.delete(source)
        await db.commit()
        
        logger.info(f"Successfully deleted data source {source_id} ({source.name})")
        
    except Exception as e:
        logger.error(f"Error deleting data source {source_id}: {e}", exc_info=True)
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting data source: {str(e)}"
        )


@router.delete("/sources/bulk", status_code=status.HTTP_204_NO_CONTENT)
async def bulk_delete_data_sources(
    source_ids: str = Query(..., description="Comma-separated list of data source IDs to delete"),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete multiple data sources and all their associated data.
    """
    try:
        if not source_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No data source IDs provided"
            )
        
        # Parse comma-separated IDs
        try:
            id_list = [int(id_str.strip()) for id_str in source_ids.split(',') if id_str.strip()]
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid data source IDs format. Expected comma-separated integers."
            )
        
        if not id_list:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid data source IDs provided"
            )
        
        # Get all sources
        result = await db.execute(
            select(DataSource)
            .where(DataSource.id.in_(id_list))
        )
        sources_to_delete = result.scalars().all()
        
        if len(sources_to_delete) != len(id_list):
            found_ids = {s.id for s in sources_to_delete}
            missing_ids = set(id_list) - found_ids
            logger.warning(f"Some data sources not found: {missing_ids}")
        
        # Delete source files for each source
        for source in sources_to_delete:
            try:
                # Delete source file from storage if it exists
                if source.config and "file_path" in source.config:
                    file_path = source.config.get("file_path")
                    if file_path:
                        try:
                            # Try cloud storage deletion (handles S3, Azure, and local)
                            deleted = await cloud_storage.delete_file(file_path)
                            if deleted:
                                logger.info(f"Deleted file from storage: {file_path}")
                            else:
                                # If cloud storage deletion returned False, try direct local file deletion
                                local_path = None
                                if Path(file_path).is_absolute() and Path(file_path).exists():
                                    local_path = Path(file_path)
                                elif not Path(file_path).is_absolute():
                                    # Try relative path in uploads directory
                                    uploads_path = Path("uploads") / Path(file_path).name
                                    if uploads_path.exists():
                                        local_path = uploads_path
                                
                                if local_path:
                                    try:
                                        await aiofiles.os.remove(local_path)
                                        logger.info(f"Deleted local file: {local_path}")
                                    except Exception as e:
                                        logger.warning(f"Failed to delete local file {local_path}: {e}")
                        except Exception as e:
                            logger.warning(f"Failed to delete file {file_path}: {e}")
            except Exception as e:
                logger.warning(f"Error deleting files for source {source.id}: {e}")
        
        # Delete all sources (CASCADE will handle data points, ETL jobs, etc.)
        for source in sources_to_delete:
            await db.delete(source)
        
        await db.commit()
        
        logger.info(f"Successfully deleted {len(sources_to_delete)} data sources")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error bulk deleting data sources: {e}", exc_info=True)
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting data sources: {str(e)}"
        )

